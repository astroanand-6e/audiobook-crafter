"""A stand-in voice for dry runs and tests: soft syllable-like tones whose pitch depends on the voice,
so speaker changes, pacing and the whole assembly can be heard without a GPU or model download."""

from __future__ import annotations

import hashlib

import numpy as np

from .base import Line, TTSEngine, Voice
from .emotion import AROUSAL


class MockEngine(TTSEngine):
    default_max_chars = 400
    can_design = True
    sample_rate = 24000

    @property
    def version(self) -> str:
        return "mock-1"

    def _tone(self, text: str, key: str, seed: int, energy: float = 0.0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        digest = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16)
        pitch = 110 + digest % 160  # 110-270 Hz: each voice sounds different
        chars_per_second = 14.0 * (1 + 0.15 * energy)
        duration = max(0.4, len(text) / chars_per_second)
        sr = self.sample_rate
        t = np.arange(int(duration * sr)) / sr
        syllables = max(1, int(duration * 4))
        envelope = np.clip(np.sin(np.pi * syllables * t / duration) ** 2, 0, 1)
        wobble = 1 + 0.03 * np.sin(2 * np.pi * 3 * t + rng.uniform(0, np.pi))
        wave = np.sin(2 * np.pi * pitch * wobble * t) + 0.3 * np.sin(4 * np.pi * pitch * t)
        return (0.18 * (1 + 0.4 * energy) * envelope * wave).astype(np.float32)

    def synthesize(self, line: Line, voice: Voice, seed: int) -> tuple[np.ndarray, int]:
        energy = AROUSAL.get(line.emotion, 0.0) * line.intensity
        key = voice.fingerprint or voice.preset or voice.id
        return self._tone(line.text, key, seed, energy), self.sample_rate

    def design(self, description: str, text: str, language: str, seed: int) -> tuple[np.ndarray, int]:
        return self._tone(text, description, seed), self.sample_rate
