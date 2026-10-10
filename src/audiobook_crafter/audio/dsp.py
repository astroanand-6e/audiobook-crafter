"""Small, dependency-free audio helpers (numpy only)."""

from __future__ import annotations

import numpy as np

_FRAME_S = 0.01


def to_mono(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    return audio.mean(axis=1) if audio.ndim == 2 else audio.reshape(-1)


def _frame_levels(audio: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    frame = max(1, int(sr * _FRAME_S))
    count = len(audio) // frame
    if count == 0:
        return np.zeros(0), frame
    frames = audio[: count * frame].reshape(count, frame)
    return np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1)), frame


def trim_silence(audio: np.ndarray, sr: int, threshold_db: float = -45.0, pad_ms: float = 40.0) -> np.ndarray:
    """Cut leading/trailing silence (and the breathy tails TTS models leave), keeping a short pad."""
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak <= 1e-6:
        return audio[:0]
    levels, frame = _frame_levels(audio, sr)
    if levels.size == 0:
        return audio
    active = np.nonzero(20 * np.log10(levels / peak + 1e-12) > threshold_db)[0]
    if active.size == 0:
        return audio[:0]
    pad = int(sr * pad_ms / 1000)
    start = max(0, active[0] * frame - pad)
    end = min(len(audio), (active[-1] + 1) * frame + pad)
    return audio[start:end]


def fade(audio: np.ndarray, sr: int, ms: float = 8.0) -> np.ndarray:
    n = min(len(audio) // 2, int(sr * ms / 1000))
    if n <= 1:
        return audio
    audio = audio.copy()
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    audio[:n] *= ramp
    audio[-n:] *= ramp[::-1]
    return audio


def clean_clip(audio: np.ndarray, sr: int) -> np.ndarray:
    """Mono, DC removed, silence trimmed, de-clicked edges, never clipping."""
    audio = to_mono(audio)
    if audio.size:
        audio = audio - float(np.mean(audio))
    audio = fade(trim_silence(audio, sr), sr)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.98:
        audio = audio * (0.98 / peak)
    return audio.astype(np.float32)


def active_rms_db(audio: np.ndarray, sr: int) -> float:
    """Loudness of the speech itself (frames within 40 dB of the loudest), in dBFS."""
    levels, _ = _frame_levels(to_mono(audio), sr)
    if levels.size == 0 or levels.max() <= 0:
        return -120.0
    active = levels[levels > levels.max() * 0.01]
    return float(20 * np.log10(np.sqrt(np.mean(active**2)) + 1e-12))


def room_tone(samples: int, level_db: float, rng: np.random.Generator) -> np.ndarray:
    """Faint pink noise so pauses don't drop to digital silence."""
    if samples <= 0:
        return np.zeros(0, dtype=np.float32)
    spectrum = np.fft.rfft(rng.standard_normal(samples))
    spectrum /= np.sqrt(np.maximum(np.arange(spectrum.size), 1))
    pink = np.fft.irfft(spectrum, n=samples)
    pink /= np.sqrt(np.mean(pink**2)) + 1e-12
    return (pink * 10 ** (level_db / 20)).astype(np.float32)
