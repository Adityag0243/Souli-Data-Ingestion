"""
Souli Streamlit App — Text chat + Voice chat
Run: streamlit run souli_pipeline/streamlit_app.py
"""
from __future__ import annotations

import os
import tempfile
import asyncio
import logging
from pathlib import Path

import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────

CONFIG_PATH = os.environ.get(
    "SOULI_CONFIG_PATH",
    str(Path(__file__).parent.parent / "configs" / "pipeline.gcp.yaml"),
)
GOLD_PATH = os.environ.get("SOULI_GOLD_PATH", None)
EXCEL_PATH = os.environ.get("SOULI_EXCEL_PATH", None)

logging.basicConfig(level=logging.WARNING)

# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Souli — Wellness Companion",
    page_icon="🌿",
    layout="centered",
)

st.markdown("""
<style>
    .stChatMessage { border-radius: 12px; }
    .block-container { max-width: 780px; padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

st.title("🌿 Souli")
st.caption("Your inner wellness companion")

# ── Session helpers ───────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading Souli...")
def get_engine():
    from souli_pipeline.config_loader import load_config
    from souli_pipeline.conversation.engine import ConversationEngine
    cfg = load_config(CONFIG_PATH)
    return ConversationEngine.from_config(cfg, gold_path=GOLD_PATH, excel_path=EXCEL_PATH)


@st.cache_resource(show_spinner="Loading voice models...")
def get_stt():
    from souli_pipeline.voice.stt import WhisperSTT
    return WhisperSTT(model_name="base")


@st.cache_resource(show_spinner="Loading TTS...")
def get_tts():
    from souli_pipeline.voice.tts import EdgeTTS
    return EdgeTTS(voice="en-IN-NeerjaNeural")


def init_chat():
    engine = get_engine()
    if "messages" not in st.session_state:
        greeting = engine.greeting()
        st.session_state.messages = [{"role": "assistant", "content": greeting}]
    if "engine_ready" not in st.session_state:
        st.session_state.engine_ready = True


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_text, tab_voice = st.tabs(["Text Chat", "Voice Chat"])

# ═══════════════════════════════════════════════════════════════════════════════
# TEXT CHAT TAB
# ═══════════════════════════════════════════════════════════════════════════════

with tab_text:
    init_chat()
    engine = get_engine()

    # Show history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Input
    if user_input := st.chat_input("Share what's on your mind..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            try:
                for chunk in engine.turn_stream(user_input):
                    full_response += chunk
                    placeholder.write(full_response + "▌")
                placeholder.write(full_response)
            except Exception:
                full_response = engine.turn(user_input)
                placeholder.write(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

        # Show energy node diagnosis if available
        diag = engine.diagnosis_summary
        if diag.get("energy_node"):
            st.caption(
                f"Energy node: `{diag['energy_node']}` | Phase: `{diag['phase']}` | Turn: {diag['turn_count']}"
            )

# ═══════════════════════════════════════════════════════════════════════════════
# VOICE CHAT TAB
# ═══════════════════════════════════════════════════════════════════════════════

with tab_voice:
    init_chat()
    engine = get_engine()

    st.markdown("**Record your voice — Souli will listen and respond.**")

    # Show voice conversation history
    if "voice_messages" not in st.session_state:
        greeting = st.session_state.messages[0]["content"]
        st.session_state.voice_messages = [{"role": "assistant", "content": greeting}]

    for msg in st.session_state.voice_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and "audio" in msg:
                st.audio(msg["audio"], format="audio/mp3")

    # Audio input (Streamlit >= 1.31)
    audio_input = st.audio_input("Press to record", key="voice_input")

    if audio_input is not None:
        with st.spinner("Transcribing..."):
            stt = get_stt()
            # Save uploaded audio to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_input.read())
                tmp_path = tmp.name
            try:
                transcript = stt.transcribe_file(tmp_path)
            finally:
                os.unlink(tmp_path)

        if transcript.strip():
            # Show user message
            st.session_state.voice_messages.append({"role": "user", "content": transcript})
            with st.chat_message("user"):
                st.write(transcript)

            # Get Souli response
            with st.spinner("Souli is thinking..."):
                response = engine.turn(transcript)

            # Generate TTS
            with st.spinner("Generating voice response..."):
                tts = get_tts()
                audio_bytes = tts.synthesize(response)

            # Store and show
            st.session_state.voice_messages.append({
                "role": "assistant",
                "content": response,
                "audio": audio_bytes,
            })
            with st.chat_message("assistant"):
                st.write(response)
                st.audio(audio_bytes, format="audio/mp3", autoplay=True)

            st.rerun()
        else:
            st.warning("Could not transcribe audio. Please try again.")

    # Text input fallback in voice tab
    with st.expander("Or type instead"):
        if voice_text := st.chat_input("Type your message...", key="voice_text_input"):
            st.session_state.voice_messages.append({"role": "user", "content": voice_text})

            with st.spinner("Souli is thinking..."):
                response = engine.turn(voice_text)

            with st.spinner("Generating voice response..."):
                tts = get_tts()
                audio_bytes = tts.synthesize(response)

            st.session_state.voice_messages.append({
                "role": "assistant",
                "content": response,
                "audio": audio_bytes,
            })
            st.rerun()
