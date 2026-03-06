#!/usr/bin/env python3
"""
Standalone LiveKit worker entry point for Souli.

This script is launched as a subprocess by `souli voice` so that
livekit-agents can properly spawn/manage worker processes without
conflicting with the souli CLI entrypoint.

Usage (invoked automatically by cli.py voice_cmd):
    python -m souli_pipeline.voice.run_worker start
"""
from __future__ import annotations
import asyncio
import logging
import os
import sys

logger = logging.getLogger(__name__)


def _load_agent():
    config_path = os.environ.get("SOULI_CONFIG_PATH") or os.environ.get(
        "SOULI_CONFIG", "configs/pipeline.gcp.yaml"
    )
    gold_path = os.environ.get("SOULI_GOLD_PATH")
    excel_path = os.environ.get("SOULI_EXCEL_PATH")

    from souli_pipeline.config_loader import load_config
    from souli_pipeline.voice.livekit_agent import SouliVoiceAgent

    cfg = load_config(config_path)
    return SouliVoiceAgent(cfg, gold_path=gold_path, excel_path=excel_path)


async def entrypoint(ctx):
    from livekit.agents import AutoSubscribe
    from livekit import rtc
    from souli_pipeline.voice.livekit_agent import _handle_audio_track, _publish_audio

    agent = _load_agent()
    engine = agent._get_engine()
    stt = agent._get_stt()
    tts = agent._get_tts()

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    greeting = engine.greeting()
    logger.info("Greeting: %s", greeting)
    audio_bytes = tts.synthesize(greeting)
    await _publish_audio(ctx.room, audio_bytes)

    @ctx.room.on("track_subscribed")
    def on_track(track, publication, participant):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            asyncio.ensure_future(
                _handle_audio_track(track, engine, stt, tts, ctx.room)
            )


if __name__ == "__main__":
    from livekit.agents import WorkerOptions, cli
    from souli_pipeline.config_loader import load_config

    config_path = os.environ.get("SOULI_CONFIG_PATH") or os.environ.get(
        "SOULI_CONFIG", "configs/pipeline.gcp.yaml"
    )
    cfg = load_config(config_path)
    v = cfg.voice

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            api_key=v.livekit_api_key or os.environ.get("LIVEKIT_API_KEY", ""),
            api_secret=v.livekit_api_secret or os.environ.get("LIVEKIT_API_SECRET", ""),
            ws_url=v.livekit_url or os.environ.get("LIVEKIT_URL", ""),
        )
    )
