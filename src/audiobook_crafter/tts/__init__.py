"""TTS engines behind one interface. Heavy model packages are imported only when an engine loads."""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass, field
from typing import Any

from ..config import TTSConfig
from .base import Line, TTSEngine, TTSError, Voice


@dataclass(frozen=True)
class EngineInfo:
    module: str
    cls: str
    package: str  # import name that must be present
    defaults: dict[str, Any] = field(default_factory=dict)
    summary: str = ""


ENGINES: dict[str, EngineInfo] = {
    "qwen3-tts": EngineInfo(
        "audiobook_crafter.tts.qwen3", "Qwen3Engine", "qwen_tts", {},
        "Qwen3-TTS: clones designed or supplied voices; 9 presets with instructions; 10 languages",
    ),
    "qwen3-voice-design": EngineInfo(
        "audiobook_crafter.tts.qwen3", "Qwen3Engine", "qwen_tts", {},
        "Qwen3-TTS VoiceDesign: invents a voice from a description (used to make reference clips)",
    ),
    "chatterbox-multilingual": EngineInfo(
        "audiobook_crafter.tts.chatterbox", "ChatterboxEngine", "chatterbox", {"variant": "multilingual"},
        "Chatterbox Multilingual: cloning + emotion intensity, 23 languages incl. Hindi",
    ),
    "chatterbox": EngineInfo(
        "audiobook_crafter.tts.chatterbox", "ChatterboxEngine", "chatterbox", {"variant": "english"},
        "Chatterbox (English): cloning + exaggeration / cfg controls",
    ),
    "chatterbox-turbo": EngineInfo(
        "audiobook_crafter.tts.chatterbox", "ChatterboxEngine", "chatterbox", {"variant": "turbo"},
        "Chatterbox Turbo (English): fast, [laugh] [sigh] [gasp] ... tags",
    ),
    "voxcpm2": EngineInfo(
        "audiobook_crafter.tts.voxcpm",
        "VoxCPMEngine",
        "voxcpm",
        {},
        "VoxCPM2: cloning with per-line style prompts, voice design, 30 languages incl. Hindi, 48 kHz",
    ),
    "mock": EngineInfo(
        "audiobook_crafter.tts.mock", "MockEngine", "numpy", {},
        "Stand-in tones for dry runs and tests (no model, no GPU)",
    ),
}  # fmt: skip


def create_local_engine(name: str, options: dict[str, Any] | None = None, device: str = "auto") -> TTSEngine:
    info = ENGINES.get(name)
    if info is None:
        raise TTSError(f"Unknown TTS engine '{name}'. Available: {', '.join(ENGINES)}")
    cls = getattr(importlib.import_module(info.module), info.cls)
    return cls(name, {**info.defaults, **(options or {})}, device)


def create_engine(name: str, config: TTSConfig) -> TTSEngine:
    options = dict(config.engines.get(name, {}))
    if options.get("python"):
        from .worker import WorkerEngine

        if name not in ENGINES:
            raise TTSError(f"Unknown TTS engine '{name}'. Available: {', '.join(ENGINES)}")
        return WorkerEngine(name, {**ENGINES[name].defaults, **options}, config.device)
    return create_local_engine(name, options, config.device)


def engine_available(name: str, config: TTSConfig | None = None) -> bool:
    if config and config.engines.get(name, {}).get("python"):
        return True  # lives in another environment; checked when it starts
    info = ENGINES.get(name)
    return bool(info) and importlib.util.find_spec(info.package) is not None


class EnginePool:
    """Engines are created and loaded on first use, and all unloaded together."""

    def __init__(self, config: TTSConfig):
        self.config = config
        self._engines: dict[str, TTSEngine] = {}

    def get(self, name: str) -> TTSEngine:
        if name not in self._engines:
            engine = create_engine(name, self.config)
            engine.load()
            self._engines[name] = engine
        return self._engines[name]

    def release(self, name: str) -> None:
        engine = self._engines.pop(name, None)
        if engine is not None:
            engine.unload()

    def close(self) -> None:
        for name in list(self._engines):
            self.release(name)


__all__ = [
    "ENGINES",
    "EnginePool",
    "Line",
    "TTSEngine",
    "TTSError",
    "Voice",
    "create_engine",
    "engine_available",
]
