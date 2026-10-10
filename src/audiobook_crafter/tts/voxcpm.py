"""VoxCPM2 (OpenBMB, Apache-2.0): 30 languages incl. Hindi, 48 kHz, voice design, and style control
*on cloned voices*: a "(very angry, through clenched teeth)" prefix steers one line while the reference
clip keeps the timbre. Install with `audiobook-crafter[voxcpm]` (Python < 3.13)."""

from __future__ import annotations

import importlib.util
import sys
from importlib import metadata
from typing import Any

import numpy as np

from .base import Line, TTSEngine, TTSError, Voice, resolve_device, seed_everything
from .emotion import style_phrase

LANGUAGES = frozenset(
    "ar my zh da nl en fi fr de el he hi id it ja km ko lo ms no pl pt ru es sw sv tl th tr vi".split()
)


class VoxCPMEngine(TTSEngine):
    default_max_chars = 300
    languages = LANGUAGES
    can_design = True

    def __init__(self, name: str, options: dict[str, Any] | None = None, device: str = "auto"):
        super().__init__(name, options, device)
        self.model: Any = None

    @property
    def version(self) -> str:
        try:
            package = metadata.version("voxcpm")
        except metadata.PackageNotFoundError:
            package = "?"
        return f"voxcpm={package};{self.options.get('model', 'openbmb/VoxCPM2')}"

    def load(self) -> None:
        if self.model is not None:
            return
        if importlib.util.find_spec("voxcpm") is None:
            raise TTSError("VoxCPM isn't installed: `uv sync --extra voxcpm` (or pip install voxcpm)")
        from voxcpm import VoxCPM

        self.model = VoxCPM.from_pretrained(
            self.options.get("model", "openbmb/VoxCPM2"),
            load_denoiser=False,
            # torch.compile needs Triton, which Windows lacks.
            optimize=bool(self.options.get("optimize", sys.platform != "win32")),
            device=resolve_device(self.device),
        )

    def _generate(self, text: str, voice: Voice | None) -> tuple[np.ndarray, int]:
        params = voice.params if voice else {}
        controls = {
            "cfg_value": float(params.get("cfg_value", 2.0)),
            "inference_timesteps": int(params.get("inference_timesteps", 10)),
        }
        if voice is not None and voice.reference is not None:
            controls["reference_wav_path"] = str(voice.reference)
            if voice.reference_text and params.get("continuation"):
                # "Ultimate cloning": continue from the clip itself. Closest copy, but style prefixes matter less.
                controls["prompt_wav_path"] = str(voice.reference)
                controls["prompt_text"] = voice.reference_text
        wav = self.model.generate(text=text, **controls)
        return np.asarray(wav, dtype=np.float32), int(self.model.tts_model.sample_rate)

    def synthesize(self, line: Line, voice: Voice, seed: int) -> tuple[np.ndarray, int]:
        self.load()
        if voice.reference is None:
            raise TTSError(
                f"VoxCPM needs a reference clip for voice '{voice.id}' (run `abcraft voices design`)."
            )
        seed_everything(seed)
        style = style_phrase(line.emotion, line.intensity, line.direction)
        return self._generate(f"({style}){line.text}" if style else line.text, voice)

    def design(self, description: str, text: str, language: str, seed: int) -> tuple[np.ndarray, int]:
        self.load()
        seed_everything(seed)
        return self._generate(f"({description}){text}", None)

    def unload(self) -> None:
        self.model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
