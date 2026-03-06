"""
Counselor response generator.

Uses Ollama llama3.1 + RAG context from Qdrant to generate responses
that mirror the warm, grounded style of the Souli video counselor.

All inference is local. No data leaves the machine.
"""
from __future__ import annotations

import logging
from typing import Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt — defines counselor personality
# ---------------------------------------------------------------------------

_COUNSELOR_SYSTEM_BASE = """\
You are Souli, a warm and deeply empathetic inner wellness companion.
Your name "Souli" means sitting with someone, understanding their soul and emotions, and walking alongside them.
You speak like a trusted, close friend who truly listens — calm, non-judgmental, and genuinely caring.
You never give medical advice. You never diagnose. You never prescribe medication.

Core approach:
1. ALWAYS match the person's energy first. If they are casual or happy, respond warmly and lightly.
   Do NOT assume they are in distress. Do NOT project sadness or heaviness onto them.
2. If they say "hello", "hi", or share something positive — respond naturally and warmly, like a friend would.
3. Let the conversation deepen gradually and naturally across several turns. Never rush into emotional depth.
4. Make the person feel truly heard and understood before exploring anything deeper.
5. When relevant and natural, weave in wisdom from the teaching content provided to you.
6. Speak naturally — no bullet points, no lists unless asked. Flowing, warm sentences.
7. Keep responses concise (2–4 sentences) unless the person has shared a lot.
8. Never push solutions. Follow the person's lead.
9. Use Indian cultural context sensitively — you understand family pressure, role expectations,
   emotional labor, and social timelines that many Indian people face.

Conversation progression (follow this naturally — don't rush):
- Early turns: Be present, warm, and curious. Ask gentle open questions about how they're feeling.
- Middle turns: Gently explore what's underneath — feelings, patterns, what's on their heart.
- Later turns: When you sense their energy state, reflect it back softly and ask if they'd like
  to explore practices, or simply continue talking. Let them choose.

Energy framework (for internal awareness only — never label the person):
- blocked_energy: withdrawal, numbness, feeling stuck, disconnected
- depleted_energy: exhausted, low self-worth, giving too much, fear of failure
- scattered_energy: overwhelmed, burnout, anxious, mind running in all directions
- outofcontrol_energy: strong emotions, anger, restlessness, reacting intensely
- normal_energy: stable, curious about growth and purpose

When teaching content from counselor videos is provided, reflect those insights naturally —
as if you are that counselor speaking in that same warm, grounded voice.
"""


def _build_counselor_system(user_name: Optional[str] = None, phase: Optional[str] = None) -> str:
    system = _COUNSELOR_SYSTEM_BASE
    if user_name:
        system = f"The person's name is {user_name}. Address them by name occasionally, warmly.\n\n" + system
    if phase in ("intake", "deepening"):
        system += (
            "\n\nCurrent phase: early conversation. Stay light and curious. "
            "Do not dive into deep emotional framing yet. Just be present and warm."
        )
    return system

_SOLUTION_SYSTEM = """\
You are Souli, a warm and practical inner wellness guide.
The person has asked for guidance. Provide it with warmth and clarity.

Present the practices gently — not as prescriptions, but as invitations.
Format: 2–3 short paragraphs. No numbered lists unless presenting multiple practices.
Ground everything in what the person shared — make it personal, not generic.
"""


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_rag_context(chunks: List[Dict]) -> str:
    if not chunks:
        return ""
    lines = ["[Relevant teaching from Souli counselor videos:]"]
    for i, c in enumerate(chunks[:3], 1):
        text = (c.get("text") or "").strip()
        if text:
            lines.append(f"{i}. {text[:400]}")
    return "\n".join(lines)


def _build_chat_messages(
    history: List[Dict[str, str]],
    user_message: str,
    rag_chunks: List[Dict],
    energy_node: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Build the messages list for Ollama chat.
    Injects RAG context as a system-level assistant hint before the user message.
    """
    messages = list(history)  # copy existing history

    # Inject RAG context as a contextual hint (injected as assistant pre-context)
    rag_text = _build_rag_context(rag_chunks)
    if rag_text:
        messages.append({"role": "assistant", "content": rag_text})

    messages.append({"role": "user", "content": user_message})
    return messages


def _build_solution_prompt(
    energy_node: str,
    framework_solution: Dict,
    user_context: str,
) -> str:
    node_label = energy_node.replace("_", " ").title()

    practices = framework_solution.get("primary_practices ( 7 min quick relief)", "")
    healing = framework_solution.get("primary_healing_principles", "")
    deeper = framework_solution.get("deeper_meditations_program ( 7 day quick recovery)", "")

    prompt = (
        f"The person is experiencing {node_label}.\n\n"
        f"What they shared: {user_context[:600]}\n\n"
        f"Healing principles: {healing[:400]}\n\n"
        f"Quick relief practices (7 min): {practices[:300]}\n\n"
        f"Deeper recovery program: {deeper[:300]}\n\n"
        f"Write a warm, personal response presenting this guidance to them."
    )
    return prompt


# ---------------------------------------------------------------------------
# Main response functions
# ---------------------------------------------------------------------------

def generate_counselor_response(
    history: List[Dict[str, str]],
    user_message: str,
    rag_chunks: List[Dict],
    energy_node: Optional[str] = None,
    ollama_model: str = "llama3.1",
    ollama_endpoint: str = "http://localhost:11434",
    temperature: float = 0.75,
    stream: bool = False,
    user_name: Optional[str] = None,
    phase: Optional[str] = None,
) -> str | Generator[str, None, None]:
    """
    Generate an empathetic counselor response using Ollama llama3.1 + RAG.

    stream=True returns a generator of text chunks.
    stream=False returns the full response string.
    """
    from ..llm.ollama import OllamaLLM

    llm = OllamaLLM(
        model=ollama_model,
        endpoint=ollama_endpoint,
        temperature=temperature,
        num_ctx=4096,
    )

    messages = _build_chat_messages(history, user_message, rag_chunks, energy_node)
    system = _build_counselor_system(user_name=user_name, phase=phase)

    if stream:
        return llm.chat_stream(messages, system=system, temperature=temperature)
    else:
        return llm.chat(messages, system=system, temperature=temperature)


def generate_solution_response(
    energy_node: str,
    framework_solution: Dict,
    user_context: str,
    ollama_model: str = "llama3.1",
    ollama_endpoint: str = "http://localhost:11434",
    temperature: float = 0.65,
    stream: bool = False,
) -> str | Generator[str, None, None]:
    """
    Generate a solution-mode response: warm presentation of practices + meditations.
    """
    from ..llm.ollama import OllamaLLM

    llm = OllamaLLM(
        model=ollama_model,
        endpoint=ollama_endpoint,
        temperature=temperature,
        num_ctx=4096,
    )

    prompt = _build_solution_prompt(energy_node, framework_solution, user_context)
    messages = [{"role": "user", "content": prompt}]

    if stream:
        return llm.chat_stream(messages, system=_SOLUTION_SYSTEM, temperature=temperature)
    else:
        return llm.chat(messages, system=_SOLUTION_SYSTEM, temperature=temperature)


def fallback_response(energy_node: Optional[str]) -> str:
    """Simple fallback when Ollama is unavailable."""
    node_responses = {
        "blocked_energy": (
            "I can feel that you're carrying something really heavy right now. "
            "It's okay to just be where you are. You don't have to have it all figured out. "
            "I'm here with you."
        ),
        "depleted_energy": (
            "It sounds like you've been giving so much — to everyone except yourself. "
            "That tiredness you feel is real, and it's telling you something important. "
            "You matter too."
        ),
        "scattered_energy": (
            "It sounds like everything is pulling at you from all directions. "
            "That overwhelm is exhausting. You deserve to breathe and just be for a moment."
        ),
        "outofcontrol_energy": (
            "I can sense there's a lot of intensity inside right now. "
            "Those feelings are valid — they're not weakness. "
            "Let's take this one breath at a time."
        ),
        "normal_energy": (
            "It sounds like you're in a reflective space, looking for something deeper. "
            "That curiosity about growth is a beautiful starting point."
        ),
    }
    return node_responses.get(
        energy_node or "",
        "Thank you for sharing that with me. I'm here and I'm listening.",
    )
