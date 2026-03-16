"""
Summarizer — generates a concise empathetic summary of what Souli has
understood about the user's situation and asks for confirmation.

Called when ConversationEngine decides enough context has been gathered
to wrap up the intake/sharing phase and move toward intent/solution.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Summary confirmation template
# ---------------------------------------------------------------------------

SUMMARY_TEMPLATE = (
    "So from what you've shared, it sounds like {summary_body}. "
    "Have I understood that right? "
    "If yes, I'd love to help you find something that might ease this — "
    "or if you'd like to share more first, that's completely okay too."
)

# Fallback when LLM is unavailable
_NODE_SUMMARY_STUBS = {
    "blocked_energy": (
        "you're feeling stuck and disconnected, like you're just going through the motions "
        "and something inside has shut down"
    ),
    "depleted_energy": (
        "you're running on empty — exhausted, undervalued, and finding it hard to complete "
        "things or feel motivated"
    ),
    "scattered_energy": (
        "you're overwhelmed with too much happening at once, feeling anxious and unable to "
        "find your footing no matter how hard you try"
    ),
    "outofcontrol_energy": (
        "there's a lot of intense emotion building up inside — anger, restlessness, or "
        "reactions that feel bigger than you'd like them to be"
    ),
    "normal_energy": (
        "you're in a relatively stable place but looking for more meaning, growth, or "
        "a deeper sense of fulfilment"
    ),
}


# ---------------------------------------------------------------------------
# LLM-based summary (Ollama)
# ---------------------------------------------------------------------------

def generate_summary(
    user_text_buffer: str,
    energy_node: Optional[str],
    user_name: Optional[str] = None,
    ollama_model: str = "llama3.1",
    ollama_endpoint: str = "http://localhost:11434",
    temperature: float = 0.5,
) -> str:
    """
    Generate a personalised summary of what the user has shared so far.
    Falls back to a keyword-based stub if Ollama is unavailable.
    Returns the full confirmation message string.
    """
    try:
        from ..llm.ollama import OllamaLLM

        llm = OllamaLLM(
            model=ollama_model,
            endpoint=ollama_endpoint,
            temperature=temperature,
            num_ctx=2048,
        )

        if not llm.is_available():
            return _fallback_summary(energy_node, user_name)

        name_part = f"The user's name is {user_name}. " if user_name else ""

        system = (
            "You are Souli, a warm empathetic companion. "
            "Your task is to write ONE concise summary sentence (max 40 words) "
            "capturing the core of what the user has shared — their situation, "
            "feelings, and what seems to be bothering them most. "
            "Write in second person (you are..., you feel..., you've been...). "
            "Do NOT offer advice. Do NOT ask a question. Just summarise warmly."
        )

        prompt = (
            f"{name_part}"
            f"Here is what the user has shared so far:\n\n"
            f"\"\"\"\n{user_text_buffer[:1200].strip()}\n\"\"\"\n\n"
            f"Write the summary sentence now (no preamble, just the sentence):"
        )

        summary_body = llm.generate(prompt=prompt, system=system, temperature=temperature)
        summary_body = summary_body.strip().strip('"').strip("'").rstrip(".")

        name_addr = f"{user_name}, " if user_name else ""
        return (
            f"{name_addr}so from everything you've shared — {summary_body}. "
            f"Have I understood that right? "
            f"If yes, I'd love to explore something that might actually help you — "
            f"or if you want to share more first, we can absolutely do that too."
        )

    except Exception as exc:
        logger.warning("Summary generation failed (%s) — using fallback.", exc)
        return _fallback_summary(energy_node, user_name)


def _fallback_summary(energy_node: Optional[str], user_name: Optional[str]) -> str:
    """Keyword-based fallback summary when Ollama is unavailable."""
    stub = _NODE_SUMMARY_STUBS.get(energy_node or "", _NODE_SUMMARY_STUBS["blocked_energy"])
    name_addr = f"{user_name}, " if user_name else ""
    return (
        f"{name_addr}from what you've shared it sounds like {stub}. "
        f"Have I understood that right? "
        f"If yes, I'd love to explore something that might actually help you — "
        f"or if you want to share more first, we can absolutely do that too."
    )