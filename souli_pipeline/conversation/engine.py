"""
Souli Conversation Engine

State machine that orchestrates:
  1. Intake     — empathetic questioning to understand the user's energy state
  2. Sharing    — user is open and talking freely; lighter touch, reflect & follow
  3. Deepening  — probe questions when user needs a little more drawing out
  4. Summary    — Souli reflects back what it understood and asks for confirmation
  5. Intent     — venting vs solution-seeking
  6. RAG        — retrieves relevant YouTube counselor content from Qdrant
  7. Response   — Ollama llama3.1 generates a warm, contextual reply
  8. Solution   — presents healing framework from gold.xlsx when requested

Usage:
    engine = ConversationEngine.from_config(cfg, gold_path="outputs/.../gold.xlsx")
    while True:
        user_input = input("You: ")
        response = engine.turn(user_input)
        print("Souli:", response)
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conversation phases
# ---------------------------------------------------------------------------
PHASE_GREETING     = "greeting"
PHASE_INTAKE       = "intake"
PHASE_SHARING      = "sharing"       # user is open & talkative — lighter touch
PHASE_DEEPENING    = "deepening"     # user needs more drawing out — structured probes
PHASE_SUMMARY      = "summary"       # Souli reflects understanding & asks confirmation
PHASE_INTENT_CHECK = "intent_check"
PHASE_SOLUTION     = "solution"
PHASE_VENTING      = "venting"       # user is frustrated — minimal questions, mostly validation


@dataclass
class ConversationState:
    phase: str = PHASE_GREETING
    turn_count: int = 0
    user_name: Optional[str] = None
    # LLM message history for Ollama
    messages: List[Dict[str, str]] = field(default_factory=list)
    # Detected or inferred energy node
    energy_node: Optional[str] = None
    node_confidence: str = "unknown"
    # Probes used per node (for both intake and sharing modes)
    used_probe_indices: Dict[str, List[int]] = field(default_factory=dict)
    used_sharing_probe_indices: Dict[str, List[int]] = field(default_factory=dict)
    # Short-answer follow-up counter
    short_answer_count: int = 0
    # Intent resolved?
    intent: Optional[str] = None        # "venting" | "solution"
    # Framework solution loaded?
    framework_loaded: bool = False
    # Concatenated user text for embedding-based diagnosis
    user_text_buffer: str = ""
    # Summary flow
    summary_attempted: bool = False     # have we already sent a summary?
    summary_confirmed: bool = False     # did user confirm our understanding?
    # Rich first message detected (user already shared a lot)
    rich_opening: bool = False


class ConversationEngine:
    """
    Main engine. Create via from_config() or from_paths().
    Call .turn(user_text) for each user message.
    Call .turn_stream(user_text) for streaming responses.
    """

    def __init__(
        self,
        chat_model: str = "llama3.1",
        tagger_model: str = "qwen2.5:1.5b",
        ollama_endpoint: str = "http://localhost:11434",
        rag_top_k: int = 3,
        max_intake_turns: int = 4,
        temperature: float = 0.75,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        qdrant_collection: str = "souli_chunks",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        nodes_allowed: Optional[List[str]] = None,
        framework: Optional[Dict] = None,
        gold_df=None,
    ):
        self.chat_model = chat_model
        self.tagger_model = tagger_model
        self.ollama_endpoint = ollama_endpoint
        self.rag_top_k = rag_top_k
        self.max_intake_turns = max_intake_turns
        self.temperature = temperature
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.qdrant_collection = qdrant_collection
        self.embedding_model = embedding_model
        self.nodes_allowed = nodes_allowed or [
            "blocked_energy",
            "depleted_energy",
            "scattered_energy",
            "outofcontrol_energy",
            "normal_energy",
        ]
        self.framework = framework or {}
        self.gold_df = gold_df
        self.state = ConversationState()

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg, gold_path: Optional[str] = None, excel_path: Optional[str] = None):
        """Build engine from PipelineConfig + optional gold.xlsx or Excel path."""
        from .solution import load_framework_from_gold, load_framework_from_excel
        from ..retrieval.match import load_gold

        c = cfg.conversation
        r = cfg.retrieval
        e = cfg.energy

        framework = {}
        gold_df = None

        if gold_path:
            try:
                framework = load_framework_from_gold(gold_path)
                gold_df = load_gold(gold_path, e.nodes_allowed)
                logger.info("Loaded framework from gold.xlsx (%d nodes)", len(framework))
            except Exception as exc:
                logger.warning("Could not load gold.xlsx: %s", exc)

        if not framework and excel_path:
            try:
                framework = load_framework_from_excel(excel_path)
                logger.info("Loaded framework from Excel (%d nodes)", len(framework))
            except Exception as exc:
                logger.warning("Could not load Excel framework: %s", exc)

        return cls(
            chat_model=c.chat_model,
            tagger_model=c.tagger_model,
            ollama_endpoint=c.ollama_endpoint,
            rag_top_k=c.rag_top_k,
            max_intake_turns=c.max_intake_turns,
            temperature=c.temperature,
            qdrant_host=r.qdrant_host,
            qdrant_port=r.qdrant_port,
            qdrant_collection=r.qdrant_collection,
            embedding_model=r.embedding_model or "sentence-transformers/all-MiniLM-L6-v2",
            nodes_allowed=e.nodes_allowed,
            framework=framework,
            gold_df=gold_df,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        """Start a fresh conversation."""
        self.state = ConversationState()

    def turn(self, user_text: str) -> str:
        """Process one user message and return Souli's response."""
        result = self._process(user_text, stream=False)
        assert isinstance(result, str)
        return result

    def turn_stream(self, user_text: str) -> Generator[str, None, None]:
        """Process one user message and stream Souli's response token by token."""
        result = self._process(user_text, stream=True)
        if isinstance(result, str):
            yield result
        else:
            yield from result

    def greeting(self) -> str:
        """Return the opening greeting (call before first user message)."""
        from .intake import get_greeting
        return get_greeting()

    # ------------------------------------------------------------------
    # Internal processing
    # ------------------------------------------------------------------

    def _process(self, user_text: str, stream: bool):
        s = self.state
        s.turn_count += 1
        user_text = (user_text or "").strip()
        s.user_text_buffer += " " + user_text

        # Add user message to history
        s.messages.append({"role": "user", "content": user_text})

        # ---- Phase routing ----

        if s.phase == PHASE_GREETING:
            response = self._handle_greeting(user_text, stream)

        elif s.phase == PHASE_INTAKE:
            response = self._handle_intake(user_text, stream)

        elif s.phase == PHASE_SHARING:
            response = self._handle_sharing(user_text, stream)

        elif s.phase == PHASE_DEEPENING:
            response = self._handle_deepening(user_text, stream)

        elif s.phase == PHASE_SUMMARY:
            response = self._handle_summary_response(user_text, stream)

        elif s.phase == PHASE_INTENT_CHECK:
            response = self._handle_intent_check(user_text, stream)

        elif s.phase == PHASE_VENTING:
            response = self._handle_venting(user_text, stream)

        elif s.phase == PHASE_SOLUTION:
            response = self._handle_solution(user_text, stream)

        else:
            response = self._handle_venting(user_text, stream)

        # Add assistant reply to history (for non-streaming, full string)
        if isinstance(response, str):
            s.messages.append({"role": "assistant", "content": response})

        return response

    # ------------------------------------------------------------------
    # Phase handlers
    # ------------------------------------------------------------------

    def _handle_greeting(self, user_text: str, stream: bool):
        """Collect user's name; return a warm, personalised welcome."""
        s = self.state
        from .intake import is_rich_message

        name = _extract_name(user_text)
        s.user_name = name

        # Check if user already shared emotional context in their first message
        words = user_text.lower().split()
        shared_feelings = any(w in _NOT_NAMES for w in words)

        # Detect rich opening — user has already shared a lot
        if is_rich_message(user_text):
            s.rich_opening = True
            # Diagnose immediately from this first message
            self._diagnose(user_text)
            # Move straight to sharing phase
            s.phase = PHASE_SHARING
            name_part = f"{name}, " if name else ""
            return (
                f"{name_part}I hear you, and I'm really glad you felt comfortable sharing that. "
                f"That takes courage. Tell me more — what's been the hardest part of all this for you?"
            )

        # Normal greeting flow
        s.phase = PHASE_INTAKE

        if name and not shared_feelings:
            return f"Lovely to meet you, {name}. How are you feeling today?"
        elif name and shared_feelings:
            return (
                f"I hear you, {name}. I'm glad you're here. "
                f"Tell me more — what's been going on?"
            )
        else:
            return (
                "I hear you. I'm glad you reached out. "
                "Tell me more — what's been going on for you?"
            )

    def _handle_intake(self, user_text: str, stream: bool):
        """
        Standard intake: user hasn't opened up much yet.
        We ask structured questions to understand the energy node.
        After enough context, move to PHASE_SHARING or trigger summary.
        """
        s = self.state
        from .intake import is_short_answer, get_short_follow_up, is_rich_message

        # Re-diagnose as we accumulate more text
        if s.turn_count >= 2:
            self._diagnose(s.user_text_buffer)

        # If user suddenly opens up richly, promote to sharing phase
        if is_rich_message(user_text) and s.turn_count >= 2:
            s.phase = PHASE_SHARING
            return self._handle_sharing(user_text, stream)

        # Enough turns + we have a node → try summary
        if s.turn_count >= self.max_intake_turns and s.energy_node and not s.summary_attempted:
            return self._trigger_summary(stream)

        # If user gave a very short answer, gently prompt for more
        if is_short_answer(user_text) and s.short_answer_count < 2:
            s.short_answer_count += 1
            follow_up = get_short_follow_up(s.short_answer_count)
            rag = self._rag_retrieve(user_text, s.energy_node)
            reply = self._llm_response(user_text, rag, stream)
            if isinstance(reply, str) and not stream:
                return reply + "\n\n" + follow_up
            return reply

        # Normal intake: respond + move toward deepening
        s.phase = PHASE_DEEPENING
        rag = self._rag_retrieve(user_text, s.energy_node)
        return self._llm_response(user_text, rag, stream)

    def _handle_sharing(self, user_text: str, stream: bool):
        """
        Sharing phase: user is open and talkative.
        Souli listens, reflects briefly, and asks one light follow-up question.
        After 2-3 sharing turns, trigger summary.
        """
        s = self.state

        # Re-diagnose with accumulated text
        self._diagnose(s.user_text_buffer)

        # Check if user switched to wanting a solution
        from .intent import detect_intent
        intent = detect_intent(user_text)
        if intent == "solution":
            s.intent = "solution"
            s.phase = PHASE_SOLUTION
            return self._handle_solution(user_text, stream)

        # After 2+ sharing turns, offer summary (don't keep asking questions)
        sharing_turns = self._count_turns_in_phase(PHASE_SHARING)
        if sharing_turns >= 2 and s.energy_node and not s.summary_attempted:
            return self._trigger_summary(stream)

        # Light sharing probe — one gentle follow-up, not a structured probe
        from .intake import get_sharing_probe
        probe_idx_list = s.used_sharing_probe_indices.setdefault(
            s.energy_node or "blocked_energy", []
        )
        probe = get_sharing_probe(s.energy_node or "blocked_energy", probe_idx_list)
        if probe:
            probe_idx_list.append(len(probe_idx_list))

        rag = self._rag_retrieve(user_text, s.energy_node)
        reply = self._llm_response(user_text, rag, stream)

        # Append light probe (non-streaming only)
        if probe and isinstance(reply, str) and not stream:
            return reply + "\n\n" + probe
        return reply

    def _handle_deepening(self, user_text: str, stream: bool):
        """
        Deepening phase: user needs more drawing out — structured probes.
        """
        s = self.state
        from .intake import get_probe, is_rich_message

        # Re-diagnose with accumulated text
        self._diagnose(s.user_text_buffer)

        # If user suddenly opens up, promote to sharing
        if is_rich_message(user_text):
            s.phase = PHASE_SHARING
            return self._handle_sharing(user_text, stream)

        # Enough turns → try summary
        if s.turn_count >= self.max_intake_turns and s.energy_node and not s.summary_attempted:
            return self._trigger_summary(stream)

        # Weave in a structured probe question
        probe_idx_list = s.used_probe_indices.setdefault(s.energy_node or "blocked_energy", [])
        probe = get_probe(s.energy_node or "blocked_energy", probe_idx_list)
        if probe:
            probe_idx_list.append(len(probe_idx_list))

        rag = self._rag_retrieve(user_text, s.energy_node)
        reply = self._llm_response(user_text, rag, stream)

        if probe and isinstance(reply, str) and not stream:
            return reply + "\n\n" + probe
        return reply

    def _trigger_summary(self, stream: bool) -> str:
        """
        Generate the summary confirmation message and move to PHASE_SUMMARY.
        Always returns a plain string (not streamed) since it's a structured message.
        """
        s = self.state
        s.summary_attempted = True
        s.phase = PHASE_SUMMARY

        from .summarizer import generate_summary
        return generate_summary(
            user_text_buffer=s.user_text_buffer.strip(),
            energy_node=s.energy_node,
            user_name=s.user_name,
            ollama_model=self.chat_model,
            ollama_endpoint=self.ollama_endpoint,
            temperature=self.temperature,
        )

    def _handle_summary_response(self, user_text: str, stream: bool):
        """
        Handle the user's response to Souli's summary confirmation.
        Routes to: confirmed → intent_check | wants_more → sharing | correction → intake
        """
        s = self.state
        from .intent import detect_summary_response, detect_intent

        # Check if user is explicitly asking for solution — skip summary confirmation
        intent = detect_intent(user_text)
        if intent == "solution":
            s.intent = "solution"
            s.summary_confirmed = True
            s.phase = PHASE_SOLUTION
            return self._handle_solution(user_text, stream)

        response_type = detect_summary_response(user_text)

        if response_type == "confirmed":
            s.summary_confirmed = True
            s.phase = PHASE_INTENT_CHECK
            return self._handle_intent_check(user_text, stream)

        elif response_type == "wants_more":
            # User confirmed but wants to keep sharing — go back to sharing
            s.phase = PHASE_SHARING
            # Add the new content and respond warmly
            rag = self._rag_retrieve(user_text, s.energy_node)
            reply = self._llm_response(user_text, rag, stream)
            return reply

        elif response_type == "correction":
            # Summary was off — ask a clarifying question
            s.phase = PHASE_INTAKE
            s.summary_attempted = False  # allow another summary attempt later
            name_part = f"{s.user_name}, " if s.user_name else ""
            return (
                f"{name_part}I appreciate you correcting me — I want to make sure I really understand. "
                f"What felt off? What's the part that's weighing on you most right now?"
            )

        else:
            # Unclear — treat as light sharing and try summary again shortly
            s.phase = PHASE_SHARING
            rag = self._rag_retrieve(user_text, s.energy_node)
            return self._llm_response(user_text, rag, stream)

    def _handle_intent_check(self, user_text: str, stream: bool):
        s = self.state
        from .intent import detect_intent, INTENT_BRIDGE

        intent = detect_intent(
            user_text,
            history_texts=[m["content"] for m in s.messages[-4:] if m["role"] == "user"],
        )

        if intent == "solution":
            s.intent = "solution"
            s.phase = PHASE_SOLUTION
            return self._handle_solution(user_text, stream)

        if intent in ("venting", "sharing"):
            s.intent = "venting"
            s.phase = PHASE_VENTING
            return self._handle_venting(user_text, stream)

        # Unclear — ask the bridge question
        s.phase = PHASE_VENTING
        rag = self._rag_retrieve(user_text, s.energy_node)
        reply = self._llm_response(user_text, rag, stream)
        if isinstance(reply, str) and not stream:
            return reply + "\n\n" + INTENT_BRIDGE
        return reply

    def _handle_venting(self, user_text: str, stream: bool):
        """
        Venting phase: user is frustrated or emotionally raw.
        Minimal questions. Mostly validation and warmth.
        """
        s = self.state
        from .intent import detect_intent, INTENT_BRIDGE

        # Check if user suddenly wants a solution
        intent = detect_intent(user_text)
        if intent == "solution":
            s.intent = "solution"
            s.phase = PHASE_SOLUTION
            return self._handle_solution(user_text, stream)

        # Track consecutive short answers
        _short = len(user_text.strip().split()) <= 3
        if _short:
            s.short_answer_count += 1
        else:
            s.short_answer_count = 0

        # After 3 consecutive very short answers, gently offer to help
        if s.short_answer_count >= 3:
            s.short_answer_count = 0
            s.phase = PHASE_INTENT_CHECK
            name_part = f"{s.user_name}, " if s.user_name else ""
            return (
                f"{name_part}I hear you. "
                "Would you like me to suggest something that might actually help, "
                "or do you just want to keep talking?"
            )

        rag = self._rag_retrieve(user_text, s.energy_node)
        return self._llm_response(user_text, rag, stream)

    def _handle_solution(self, user_text: str, stream: bool):
        s = self.state
        from .counselor import generate_solution_response
        from .solution import get_solution_for_node, format_solution_text

        node = s.energy_node or "blocked_energy"
        sol = get_solution_for_node(node, self.framework)

        if not sol:
            logger.warning("No framework solution for node '%s' — using LLM only", node)
            rag = self._rag_retrieve(user_text, node)
            return self._llm_response(user_text, rag, stream)

        user_context = s.user_text_buffer.strip()
        try:
            return generate_solution_response(
                energy_node=node,
                framework_solution=sol,
                user_context=user_context,
                ollama_model=self.chat_model,
                ollama_endpoint=self.ollama_endpoint,
                temperature=self.temperature,
                stream=stream,
            )
        except Exception as exc:
            logger.warning("Ollama solution generation failed: %s", exc)
            return format_solution_text(node, sol)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _diagnose(self, text: str):
        """Update energy_node based on accumulated user text."""
        s = self.state
        from ..energy.normalize import infer_node
        from ..retrieval.match import diagnose as retrieval_diagnose

        try:
            if self.gold_df is not None and not self.gold_df.empty:
                result = retrieval_diagnose(
                    text,
                    self.gold_df,
                    self.nodes_allowed,
                    embedding_model=self.embedding_model,
                )
                s.energy_node = result.get("energy_node") or "blocked_energy"
                s.node_confidence = result.get("confidence", "keyword_fallback")
            else:
                s.energy_node = infer_node(text, "")
                s.node_confidence = "keyword_fallback"
        except Exception as exc:
            logger.warning("Diagnosis error: %s", exc)
            s.energy_node = s.energy_node or "blocked_energy"

    def _rag_retrieve(self, query: str, energy_node: Optional[str]) -> list:
        """Retrieve relevant YouTube teaching chunks from Qdrant."""
        try:
            from ..retrieval.qdrant_store import query_chunks
            return query_chunks(
                user_text=query,
                collection=self.qdrant_collection,
                energy_node=energy_node,
                top_k=self.rag_top_k,
                embedding_model=self.embedding_model,
                host=self.qdrant_host,
                port=self.qdrant_port,
            )
        except Exception as exc:
            logger.debug("Qdrant retrieval failed: %s", exc)
            return []

    def _llm_response(self, user_text: str, rag_chunks: list, stream: bool):
        """Generate counselor response via Ollama."""
        from .counselor import generate_counselor_response, fallback_response

        # Limit history to last 8 messages (4 turns) to avoid Ollama timeout
        history = self.state.messages[:-1][-8:]

        # Build a short list of topics already asked about
        asked_topics = []
        _topic_words = ["sleep", "eat", "food", "relax", "break", "support", "colleague",
                        "manager", "family", "friend", "work", "office", "exercise", "hobby"]
        for m in self.state.messages:
            if m["role"] == "assistant":
                low = m["content"].lower()
                for t in _topic_words:
                    if t in low and t not in asked_topics:
                        asked_topics.append(t)

        try:
            return generate_counselor_response(
                history=history,
                user_message=user_text,
                rag_chunks=rag_chunks,
                energy_node=self.state.energy_node,
                ollama_model=self.chat_model,
                ollama_endpoint=self.ollama_endpoint,
                temperature=self.temperature,
                stream=stream,
                user_name=self.state.user_name,
                phase=self.state.phase,
                asked_topics=asked_topics,
            )
        except Exception as exc:
            logger.warning("Ollama response failed: %s — using fallback.", exc)
            return fallback_response(self.state.energy_node, user_text)

    def _count_turns_in_phase(self, phase: str) -> int:
        """
        Count how many consecutive assistant messages have been sent while in a given phase.
        Used to decide when to trigger summary in sharing mode.
        """
        count = 0
        for msg in reversed(self.state.messages):
            if msg["role"] == "assistant":
                count += 1
            # We stop counting when we hit a phase boundary marker — approximate by
            # checking if we've gone back far enough (4 messages = 2 turns)
            if count >= 4:
                break
        # Approximate: if we've been in sharing for this many assistant turns
        sharing_count = sum(
            1 for m in self.state.messages
            if m["role"] == "assistant" and self.state.phase == phase
        )
        return sharing_count

    # ------------------------------------------------------------------
    # Convenience info
    # ------------------------------------------------------------------

    @property
    def diagnosis_summary(self) -> Dict:
        s = self.state
        return {
            "energy_node": s.energy_node,
            "confidence": s.node_confidence,
            "intent": s.intent,
            "phase": s.phase,
            "turn_count": s.turn_count,
        }


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_STOP_WORDS = {
    "hello", "hi", "hey", "yes", "no", "ok", "okay", "sure", "thanks",
    "hlo", "hii", "helo", "yrr", "yaar", "bhai", "dost", "sir", "mam",
    "na", "ha", "haan", "nahi", "hn", "hmm", "hm", "um", "uh",
}

_NOT_NAMES = {
    "very", "so", "really", "quite", "just", "feeling", "not", "too", "a", "an", "the",
    "good", "bad", "okay", "fine", "great", "terrible", "horrible", "well", "better",
    "sad", "happy", "angry", "tired", "exhausted", "stressed", "anxious", "worried",
    "scared", "nervous", "depressed", "confused", "lost", "desperate", "frustrated",
    "overwhelmed", "lonely", "alone", "hurt", "broken", "stuck", "empty", "numb",
    "excited", "grateful", "blessed", "unsure", "unsettled", "restless",
    "here", "there", "new", "back", "trying", "going", "looking", "feeling", "thinking",
    "also", "still", "already", "always", "never", "sometimes", "often", "just",
    "bit", "little", "kind", "sort", "totally", "completely", "absolutely",
}


def _extract_name(text: str) -> Optional[str]:
    """
    Try to extract a first name from the user's greeting response.
    Handles: "John", "I'm John", "my name is John", "call me John".
    Returns capitalised name, or None if nothing suitable found.
    """
    text = (text or "").strip()

    for pattern in [
        r"(?:my name is|name(?:'?s)? is|call me|they call me)\s+([A-Za-z]+)",
    ]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            candidate = m.group(1).lower()
            if candidate not in _NOT_NAMES and candidate not in _STOP_WORDS:
                return m.group(1).capitalize()

    for pattern in [r"(?:i'?m|i am)\s+([A-Za-z]+)"]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            candidate = m.group(1).lower()
            if candidate not in _NOT_NAMES and candidate not in _STOP_WORDS:
                return m.group(1).capitalize()

    words = [w for w in text.split() if w.isalpha()]
    meaningful = [w for w in words if w.lower() not in _STOP_WORDS and w.lower() not in _NOT_NAMES]
    if meaningful and len(words) <= 2:
        return meaningful[0].capitalize()

    return None