"""The engine interface every TTS model plugs into."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


class TTSError(RuntimeError):
    pass


@dataclass(frozen=True)
class Voice:
    """A cast member's voice, resolved and ready for an engine."""

    id: str
    reference: Path | None = None  # prepared mono WAV
    reference_text: str | None = None
    preset: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    seed: int = 1234
    fingerprint: str = ""  # content hash of everything above; part of every clip's cache key


@dataclass
class Line:
    text: str
    language: str = "en"
    emotion: str = "neutral"
    intensity: float = 0.3
    direction: str = ""
    tags: list[str] = field(default_factory=list)


class TTSEngine(ABC):
    name: str = ""
    default_max_chars: int = 300
    languages: frozenset[str] | None = None  # ISO 639-1 codes; None = anything
    can_design: bool = False

    def __init__(self, name: str, options: dict[str, Any] | None = None, device: str = "auto"):
        self.name = name
        self.options = dict(options or {})
        self.device = device

    @property
    def version(self) -> str:
        """Changes whenever the same inputs would produce different audio (model, package version)."""
        return ""

    def load(self) -> None:  # noqa: B027 - optional hook
        """Load weights. Called once before the first request."""

    def unload(self) -> None:  # noqa: B027 - optional hook
        """Release GPU memory."""

    @abstractmethod
    def synthesize(self, line: Line, voice: Voice, seed: int) -> tuple[np.ndarray, int]:
        """Mono float32 audio and its sample rate."""

    def design(self, description: str, text: str, language: str, seed: int) -> tuple[np.ndarray, int]:
        """Create a new voice from a description, speaking `text` (voice-design engines only)."""
        raise TTSError(f"{self.name} can't design voices")

    def supports(self, language: str) -> bool:
        return self.languages is None or language in self.languages


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
