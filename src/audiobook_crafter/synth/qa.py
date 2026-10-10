"""Automatic checks on every generated clip, so a bad take is regenerated instead of shipped.

- Duration: TTS models sometimes babble on, loop, or stop half way; length per character catches most of it.
- ASR (optional, faster-whisper): transcribe the clip and compare it with the text (word error rate).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np

from ..config import QAConfig
from ..text.normalize import normalize_for_tts

log = logging.getLogger(__name__)

# Rough speaking rates in characters per second.
_RATES = {"zh": 5.0, "ja": 7.0, "ko": 7.0}


def _words(text: str) -> list[str]:
    return re.sub(r"[^\w\s']", " ", text.lower()).split()


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref, hyp = _words(reference), _words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        current = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, start=1):
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h))
        previous = current
    return previous[-1] / len(ref)


def _to_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000:
        return audio.astype(np.float32)
    positions = np.arange(0, len(audio), sr / 16000)
    return np.interp(positions, np.arange(len(audio)), audio).astype(np.float32)


@dataclass
class Verdict:
    issue: str  # "" when the clip passed
    score: float  # lower is better; used to pick the best of several takes
    wer: float | None = None


class ClipChecker:
    def __init__(self, config: QAConfig, language: str, device: str = "auto"):
        self.config = config
        self.language = language
        self.device = device
        self._asr = None

    def expected_seconds(self, text: str) -> float:
        rate = _RATES.get(self.language, 14.5)
        return max(0.6, len(re.sub(r"\s+", "", text)) / rate * 1.08)

    def duration_issue(self, text: str, seconds: float) -> tuple[str, float]:
        expected = self.expected_seconds(text)
        ratio = seconds / expected
        if seconds <= 0.05:
            return "silent", 10.0
        if seconds > max(2.2 * expected, expected + 3.5):
            return f"too long ({seconds:.1f}s vs ~{expected:.1f}s): runaway or repeated speech", ratio
        if len(text) > 20 and seconds < 0.4 * expected:
            return f"too short ({seconds:.1f}s vs ~{expected:.1f}s): probably cut off", 1 / max(ratio, 1e-3)
        return "", abs(np.log(max(ratio, 1e-3)))

    def _transcribe(self, audio: np.ndarray, sr: int) -> str | None:
        if self._asr is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError:
                log.warning(
                    "qa.asr is on but faster-whisper isn't installed (`uv sync --extra qa`); skipping ASR"
                )
                self.config.asr = False
                return None
            device = self.config.asr_device if self.config.asr_device != "auto" else self.device
            if device == "auto":
                try:
                    import torch

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device = "cpu"
            compute = "float16" if device == "cuda" else "int8"
            self._asr = WhisperModel(self.config.asr_model, device=device, compute_type=compute)
        segments, _ = self._asr.transcribe(_to_16k(audio, sr), language=self.language, beam_size=1)
        return " ".join(s.text for s in segments)

    def check(self, text: str, audio: np.ndarray, sr: int) -> Verdict:
        seconds = len(audio) / sr if sr else 0.0
        issue, score = self.duration_issue(text, seconds) if self.config.duration_check else ("", 0.0)
        if issue == "silent":
            return Verdict(issue, score)
        wer = None
        if self.config.asr and len(text.split()) >= 3:
            heard = self._transcribe(audio, sr)
            if heard is not None:
                wer = word_error_rate(text, normalize_for_tts(heard, self.language))
                score = score + wer * 5
                if wer > self.config.max_wer and not issue:
                    issue = f"ASR mismatch (WER {wer:.0%})"
        return Verdict(issue, score, wer)

    def close(self) -> None:
        self._asr = None
