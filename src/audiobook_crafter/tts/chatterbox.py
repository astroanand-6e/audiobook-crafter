"""Resemble AI Chatterbox (MIT): Multilingual (23 languages incl. Hindi), the original English model,
and Turbo (English, inline [laugh]/[sigh] tags). Install with `audiobook-crafter[chatterbox]`.

Emotion maps onto `exaggeration` / `cfg_weight`; every voice's conditioning is computed once and
reused for all of its lines."""

from __future__ import annotations

import importlib.util
from importlib import metadata
from typing import Any

import numpy as np

from .base import Line, TTSEngine, TTSError, Voice, resolve_device, seed_everything
from .emotion import chatterbox_controls, turbo_prefix

MULTILINGUAL = frozenset("ar da de el en es fi fr he hi it ja ko ms nl no pl pt ru sv sw tr zh".split())


class ChatterboxEngine(TTSEngine):
    def __init__(self, name: str, options: dict[str, Any] | None = None, device: str = "auto"):
        super().__init__(name, options, device)
        self.variant = str(self.options.get("variant", "multilingual"))
        self.languages = MULTILINGUAL if self.variant == "multilingual" else frozenset({"en"})
        self.default_max_chars = 280 if self.variant == "multilingual" else 300
        self.model: Any = None
        self._default_conds: Any = None
        self._conds: dict[str, Any] = {}

    @property
    def version(self) -> str:
        try:
            package = metadata.version("chatterbox-tts")
        except metadata.PackageNotFoundError:
            package = "?"
        return f"chatterbox-tts={package};{self.variant};{self.options.get('t3_model', '')}"

    def load(self) -> None:
        if self.model is not None:
            return
        if importlib.util.find_spec("chatterbox") is None:
            raise TTSError(
                "Chatterbox isn't installed: `uv sync --extra chatterbox` (or pip install chatterbox-tts)"
            )
        device = resolve_device(self.device)
        if self.variant == "multilingual":
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS

            extra = {"t3_model": self.options["t3_model"]} if self.options.get("t3_model") else {}
            self.model = ChatterboxMultilingualTTS.from_pretrained(device=device, **extra)
        elif self.variant == "turbo":
            from chatterbox.tts_turbo import ChatterboxTurboTTS

            self.model = ChatterboxTurboTTS.from_pretrained(device=device)
        else:
            from chatterbox.tts import ChatterboxTTS

            self.model = ChatterboxTTS.from_pretrained(device=device)
        self._default_conds = self.model.conds

    def _use_voice(self, voice: Voice) -> None:
        if voice.reference is None:
            if self._default_conds is None:
                raise TTSError(f"{self.name} needs a reference clip for voice '{voice.id}' (5 s or longer).")
            self.model.conds = self._default_conds
            return
        conds = self._conds.get(voice.fingerprint)
        if conds is None:
            try:
                if self.variant == "turbo":
                    self.model.prepare_conditionals(
                        str(voice.reference), exaggeration=0.0, norm_loudness=True
                    )
                else:
                    self.model.prepare_conditionals(str(voice.reference), exaggeration=0.5)
            except AssertionError as err:  # Turbo: "Audio prompt must be longer than 5 seconds!"
                raise TTSError(f"Voice '{voice.id}': {err} ({voice.reference})") from err
            conds = self._conds[voice.fingerprint] = self.model.conds
        self.model.conds = conds

    def synthesize(self, line: Line, voice: Voice, seed: int) -> tuple[np.ndarray, int]:
        self.load()
        self._use_voice(voice)
        seed_everything(seed)
        params = voice.params
        sampling = {
            k: params[k] for k in ("temperature", "top_p", "min_p", "repetition_penalty") if k in params
        }
        if self.variant == "turbo":
            sampling.pop("min_p", None)  # Turbo ignores it and warns
            if "top_k" in params:
                sampling["top_k"] = params["top_k"]
            wav = self.model.generate(turbo_prefix(line.tags) + line.text, **sampling)
        else:
            exaggeration, cfg_weight = chatterbox_controls(
                line.emotion,
                line.intensity,
                float(params.get("exaggeration", 0.5)),
                float(params.get("cfg_weight", 0.5)),
            )
            if params.get("cross_language"):
                cfg_weight = 0.0  # reference clip in another language: stops its accent bleeding through
            controls = {"exaggeration": exaggeration, "cfg_weight": cfg_weight, **sampling}
            if self.variant == "multilingual":
                wav = self.model.generate(line.text, language_id=line.language, **controls)
            else:
                wav = self.model.generate(line.text, **controls)
        audio = wav.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return audio, int(self.model.sr)

    def unload(self) -> None:
        self.model = None
        self._default_conds = None
        self._conds.clear()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
