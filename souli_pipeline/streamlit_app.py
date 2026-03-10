"""
Souli Streamlit App — Text chat + Voice chat + Debug sidebar
Run: streamlit run souli_pipeline/streamlit_app.py
"""
from __future__ import annotations

import os
import tempfile
import logging
from pathlib import Path

import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────

CONFIG_PATH = os.environ.get(
    "SOULI_CONFIG_PATH",
    str(Path(__file__).parent.parent / "configs" / "pipeline.gcp.yaml"),
)
GOLD_PATH  = os.environ.get("SOULI_GOLD_PATH", None)
EXCEL_PATH = os.environ.get("SOULI_EXCEL_PATH", None)

logging.basicConfig(level=logging.WARNING)

# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Souli",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  section[data-testid="stSidebar"] { background: #0f1117; }
  .souli-tag {
      display:inline-block; padding:2px 10px; border-radius:20px;
      font-size:0.78rem; font-weight:600; margin:2px;
  }
  .phase-badge  { background:#1e3a5f; color:#7ec8e3; }
  .node-badge   { background:#1a3a2a; color:#6fcf97; }
  .turn-badge   { background:#2a1a3a; color:#bb86fc; }
  .fallback-badge { background:#3a1a1a; color:#ff6b6b; }
  .llm-badge    { background:#1a2a3a; color:#56ccf2; }
  .rag-item     { border-left:3px solid #444; padding:4px 8px; margin:4px 0;
                  font-size:0.8rem; color:#aaa; border-radius:2px; }
  .think-header { color:#888; font-size:0.72rem; text-transform:uppercase;
                  letter-spacing:1px; margin-top:12px; margin-bottom:4px; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading Souli config...")
def _get_config():
    from souli_pipeline.config_loader import load_config
    return load_config(CONFIG_PATH)


def get_engine():
    if "engine" not in st.session_state:
        from souli_pipeline.conversation.engine import ConversationEngine
        cfg = _get_config()
        st.session_state.engine = ConversationEngine.from_config(
            cfg, gold_path=GOLD_PATH, excel_path=EXCEL_PATH
        )
    return st.session_state.engine


@st.cache_resource(show_spinner="Loading STT...")
def get_stt():
    from souli_pipeline.voice.stt import WhisperSTT
    return WhisperSTT(model_name="base")


@st.cache_resource(show_spinner="Loading TTS...")
def get_tts():
    from souli_pipeline.voice.tts import EdgeTTS
    return EdgeTTS(voice="en-IN-NeerjaNeural")


def init_session():
    engine = get_engine()
    if "messages" not in st.session_state:
        greeting = engine.greeting()
        st.session_state.messages = [{"role": "assistant", "content": greeting}]
        st.session_state.debug_log = []   # list of per-turn debug dicts


_PHASE_LABELS = {
    "greeting":    ("Greeting",     "phase-badge"),
    "intake":      ("Intake",       "phase-badge"),
    "deepening":   ("Deepening",    "phase-badge"),
    "intent_check":("Intent Check", "phase-badge"),
    "venting":     ("Venting",      "phase-badge"),
    "solution":    ("Solution",     "phase-badge"),
}

_NODE_LABELS = {
    "blocked_energy":     ("Blocked Energy",      "#e74c3c"),
    "depleted_energy":    ("Depleted Energy",      "#e67e22"),
    "scattered_energy":   ("Scattered Energy",     "#f1c40f"),
    "outofcontrol_energy":("Out-of-Control Energy","#9b59b6"),
    "normal_energy":      ("Normal / Growth",      "#27ae60"),
}


def render_sidebar():
    """Render the LLM thinking / debug sidebar."""
    engine = get_engine()
    diag   = engine.diagnosis_summary
    debug  = st.session_state.get("debug_log", [])

    with st.sidebar:
        st.markdown("### 🧠 Souli's Thinking")
        st.divider()

        # ── Current state ──────────────────────────────────────────────
        phase = diag.get("phase", "greeting")
        label, css = _PHASE_LABELS.get(phase, (phase, "phase-badge"))
        st.markdown(
            f'<div class="think-header">Conversation Phase</div>'
            f'<span class="souli-tag {css}">{label}</span>',
            unsafe_allow_html=True,
        )

        node = diag.get("energy_node")
        if node:
            node_label, color = _NODE_LABELS.get(node, (node, "#888"))
            st.markdown(
                f'<div class="think-header">Detected Energy Node</div>'
                f'<span class="souli-tag node-badge" style="border-left:3px solid {color};">'
                f'{node_label}</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="think-header">Detected Energy Node</div>'
                '<span style="color:#555;font-size:0.8rem;">Still listening...</span>',
                unsafe_allow_html=True,
            )

        confidence = diag.get("confidence", "")
        intent     = diag.get("intent", "")
        turn       = diag.get("turn_count", 0)

        cols = st.columns(2)
        with cols[0]:
            st.markdown(
                f'<div class="think-header">Turn</div>'
                f'<span class="souli-tag turn-badge">#{turn}</span>',
                unsafe_allow_html=True,
            )
        with cols[1]:
            if intent:
                st.markdown(
                    f'<div class="think-header">Intent</div>'
                    f'<span class="souli-tag phase-badge">{intent}</span>',
                    unsafe_allow_html=True,
                )

        if confidence and confidence != "unknown":
            st.markdown(
                f'<div class="think-header">Diagnosis Method</div>'
                f'<span style="color:#777;font-size:0.78rem;">{confidence}</span>',
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Per-turn debug log ─────────────────────────────────────────
        if debug:
            st.markdown("### Last Turn")
            last = debug[-1]

            source = last.get("source", "llm")
            if source == "llm":
                st.markdown(
                    '<span class="souli-tag llm-badge">LLM (llama3.1)</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<span class="souli-tag fallback-badge">Fallback (Ollama offline)</span>',
                    unsafe_allow_html=True,
                )

            rag = last.get("rag_chunks", [])
            if rag:
                st.markdown(
                    f'<div class="think-header">RAG — {len(rag)} chunks retrieved</div>',
                    unsafe_allow_html=True,
                )
                for c in rag[:3]:
                    text  = (c.get("text") or "")[:120]
                    enode = c.get("energy_node", "")
                    st.markdown(
                        f'<div class="rag-item">'
                        f'<span style="color:#56ccf2;font-size:0.7rem;">[{enode}]</span> {text}…'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    '<div class="think-header">RAG</div>'
                    '<span style="color:#555;font-size:0.78rem;">No chunks retrieved</span>',
                    unsafe_allow_html=True,
                )

            if user_name := last.get("user_name"):
                st.markdown(
                    f'<div class="think-header">Name Detected</div>'
                    f'<span style="color:#6fcf97;font-size:0.85rem;">{user_name}</span>',
                    unsafe_allow_html=True,
                )

        st.divider()

        # ── Reset button ───────────────────────────────────────────────
        if st.button("Start New Session", use_container_width=True):
            for key in ["messages", "voice_messages", "engine", "debug_log"]:
                st.session_state.pop(key, None)
            st.rerun()


def _run_turn(user_input: str) -> tuple[str, dict]:
    """Run engine turn and return (response, debug_info)."""
    engine = get_engine()

    # Capture RAG chunks by monkey-patching temporarily
    rag_captured = []
    _orig_rag = engine._rag_retrieve

    def _capturing_rag(query, energy_node):
        chunks = _orig_rag(query, energy_node)
        rag_captured.extend(chunks)
        return chunks

    engine._rag_retrieve = _capturing_rag

    source = "llm"
    try:
        full_response = ""
        for chunk in engine.turn_stream(user_input):
            full_response += chunk
    except Exception:
        full_response = engine.turn(user_input)
        source = "fallback"
    finally:
        engine._rag_retrieve = _orig_rag

    # Detect if fallback was used (short, no streaming variation)
    diag = engine.diagnosis_summary
    debug = {
        "source":     source,
        "rag_chunks": rag_captured,
        "user_name":  engine.state.user_name,
        "phase":      diag.get("phase"),
        "node":       diag.get("energy_node"),
    }
    return full_response, debug


# ── Layout ────────────────────────────────────────────────────────────────────

init_session()
render_sidebar()

st.markdown("## 🌿 Souli")
st.caption("Your inner wellness companion")

tab_text, tab_voice = st.tabs(["💬 Text Chat", "🎤 Voice Chat"])

# ═══════════════════════════════════════════════════════════════════════════════
# TEXT CHAT TAB
# ═══════════════════════════════════════════════════════════════════════════════

with tab_text:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if user_input := st.chat_input("Share what's on your mind..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.write("_Souli is with you..._")

            full_response, debug = _run_turn(user_input)
            placeholder.write(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.session_state.debug_log.append(debug)
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# VOICE CHAT TAB
# ═══════════════════════════════════════════════════════════════════════════════

with tab_voice:
    if "voice_messages" not in st.session_state:
        greeting = st.session_state.messages[0]["content"]
        st.session_state.voice_messages = [{"role": "assistant", "content": greeting}]

    for msg in st.session_state.voice_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and "audio" in msg:
                st.audio(msg["audio"], format="audio/mp3")

    audio_input = st.audio_input("Press to record", key="voice_input")

    if audio_input is not None:
        with st.spinner("Transcribing your voice..."):
            stt = get_stt()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_input.read())
                tmp_path = tmp.name
            try:
                transcript = stt.transcribe_file(tmp_path)
            finally:
                os.unlink(tmp_path)

        if transcript.strip():
            st.session_state.voice_messages.append({"role": "user", "content": transcript})

            with st.spinner("Souli is thinking..."):
                response, debug = _run_turn(transcript)
            st.session_state.debug_log.append(debug)

            with st.spinner("Generating voice response..."):
                tts = get_tts()
                audio_bytes = tts.synthesize(response)

            st.session_state.voice_messages.append({
                "role": "assistant",
                "content": response,
                "audio": audio_bytes,
            })
            st.rerun()
        else:
            st.warning("Could not transcribe. Please try again.")

    st.markdown("---")
    if voice_text := st.chat_input("Or type here...", key="voice_text_input"):
        st.session_state.voice_messages.append({"role": "user", "content": voice_text})
        with st.spinner("Souli is thinking..."):
            response, debug = _run_turn(voice_text)
        st.session_state.debug_log.append(debug)
        with st.spinner("Generating voice response..."):
            tts = get_tts()
            audio_bytes = tts.synthesize(response)
        st.session_state.voice_messages.append({
            "role": "assistant", "content": response, "audio": audio_bytes,
        })
        st.rerun()
