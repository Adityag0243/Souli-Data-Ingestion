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
You are Souli — a warm, real friend who listens and cares.
You talk like a close friend: simple words, short sentences, zero jargon.
Never use formal or heavy language. Keep it natural and human.

Rules:
- Max 2-3 short sentences per reply. Never write paragraphs.
- ONE question per reply — short and direct.
- Match the person's energy. If they're casual, be casual. Don't be dramatic or therapeutic.
- If they share something painful, acknowledge it briefly and ask one gentle question.
- Never repeat back what they just said. Never say "It sounds like..." more than once.
- Use simple everyday words. Avoid: "It sounds like", "I can sense", "It seems", "It appears".
- If they ask for a solution, give it — don't keep asking more questions.
- You understand Indian family pressure, relationship stress, work stress very well.
- Never give medical advice.

When teaching content is provided, use it naturally — like a friend sharing something useful.
"""


def _build_counselor_system(user_name: Optional[str] = None, phase: Optional[str] = None) -> str:
    system = _COUNSELOR_SYSTEM_BASE
    if user_name:
        system = f"The person's name is {user_name}. Address them by name occasionally, warmly.\n\n" + system
    if phase in ("intake", "deepening"):
        system += "\n\nSTRICT: 1-2 sentences only. One short question at the end."
    elif phase == "intent_check":
        system += "\n\nSTRICT: 2 sentences max. Ask if they want practical help or just to talk."
    elif phase == "venting":
        system += "\n\nSTRICT: 2 sentences max. Be present. One warm question."
    else:
        system += "\n\nSTRICT: 2-3 sentences max. Be direct and warm."
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
        num_ctx=2048,
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


def fallback_response(energy_node: Optional[str], user_text: str = "") -> str:
    """Simple fallback when Ollama is unavailable. Varies based on user_text length."""
    import hashlib
    # Pick a variant based on user text so it doesn't repeat
    variant = int(hashlib.md5((user_text or "x").encode()).hexdigest(), 16) % 3

    node_responses = {
        "blocked_energy": [
            "That sounds really heavy. What part of it is weighing on you the most right now?",
            "I hear you. It's okay to not have it figured out. What feels hardest in this moment?",
            "You don't have to carry this alone. What would feel like a small relief right now?",
        ],
        "depleted_energy": [
            "You've been giving a lot, haven't you. What's draining you the most?",
            "That tiredness is real. When did you last feel like yourself?",
            "It sounds like you've been running on empty. What does your day usually look like?",
        ],
        "scattered_energy": [
            "Everything seems to be pulling at you at once. What's the loudest thing on your mind?",
            "That overwhelm makes sense. Which part of this feels hardest to handle?",
            "You're juggling a lot. What would help you feel even slightly more settled?",
        ],
        "outofcontrol_energy": [
            "There's a lot of intensity in what you're carrying. What triggered this the most?",
            "Those feelings are valid. What's been building up inside you?",
            "I'm here with you. What do you most need right now — to be heard, or to find a way through?",
        ],
        "normal_energy": [
            "That's interesting — what's been on your mind lately?",
            "I'm curious — what brought you here today?",
            "Tell me more — what's been sitting with you?",
        ],
    }
    options = node_responses.get(energy_node or "", [
        "I'm here with you. Tell me more about what's going on.",
        "That makes sense. What's been the hardest part?",
        "I hear you. What would feel helpful right now?",
    ])
    return options[variant % len(options)]
