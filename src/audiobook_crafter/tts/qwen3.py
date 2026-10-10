"""Qwen3-TTS (Apache-2.0): cloning (Base), preset speakers that follow instructions (CustomVoice), and
voice design from a text description (VoiceDesign). Install with `audiobook-crafter[qwen]`.

Cloning a designed reference keeps a character's timbre identical for the whole book. The Base model
takes no emotion instruction, so cloned voices act from the text itself; preset voices (CustomVoice)
get the line's emotion as an instruction."""

from __future__ import annotations

import gc
import importlib.util
import logging
from importlib import metadata
from typing import Any

import numpy as np

from .base import Line, TTSEngine, TTSError, Voice, resolve_device, seed_everything
from .emotion import instruction

log = logging.getLogger(__name__)

LANGUAGE_NAMES = {
    "zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean", "de": "German",
    "fr": "French", "ru": "Russian", "pt": "Portuguese", "es": "Spanish", "it": "Italian",
}  # fmt: skip
PRESETS = ("Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee")
_GENERATION_KEYS = (
    "do_sample", "top_k", "top_p", "temperature", "repetition_penalty", "subtalker_dosample",
    "subtalker_top_k", "subtalker_top_p", "subtalker_temperature", "max_new_tokens",
)  # fmt: skip


class Qwen3Engine(TTSEngine):
    default_max_chars = 400
    languages = frozenset(LANGUAGE_NAMES)
    can_design = True

    def __init__(self, name: str, options: dict[str, Any] | None = None, device: str = "auto"):
        super().__init__(name, options, device)
        self._models: dict[str, Any] = {}
        self._prompts: dict[str, Any] = {}  # voice fingerprint -> reusable clone prompt

    def _repo(self, kind: str) -> str:
        if self.options.get(f"{kind.lower()}_model"):
            return str(self.options[f"{kind.lower()}_model"])
        size = "1.7B" if kind == "VoiceDesign" else str(self.options.get("model_size", "1.7B"))
        return f"Qwen/Qwen3-TTS-12Hz-{size}-{kind}"

    @property
    def version(self) -> str:
        try:
            package = metadata.version("qwen-tts")
        except metadata.PackageNotFoundError:
            package = "?"
        return f"qwen-tts={package};{self._repo('Base')};{self._repo('CustomVoice')}"

    def load(self) -> None:
        if importlib.util.find_spec("qwen_tts") is None:
            raise TTSError("Qwen3-TTS isn't installed: `uv sync --extra qwen` (or pip install qwen-tts)")

    def _model(self, kind: str) -> Any:
        if kind in self._models:
            return self._models[kind]
        import torch
        from qwen_tts import Qwen3TTSModel

        # Designing and speaking are separate phases; keep only one family member on the GPU at a time.
        if kind == "VoiceDesign":
            self.unload()
        elif "VoiceDesign" in self._models:
            self._drop("VoiceDesign")
        device = resolve_device(self.device)
        on_gpu = device.startswith("cuda")  # "cuda", "cuda:0", "cuda:1", ...
        attn = self.options.get("attn_implementation")
        if not attn:
            attn = "flash_attention_2" if on_gpu and importlib.util.find_spec("flash_attn") else "sdpa"
        log.info("Loading %s on %s", self._repo(kind), device)
        self._models[kind] = Qwen3TTSModel.from_pretrained(
            self._repo(kind),
            device_map="cuda:0" if device == "cuda" else device,
            dtype=torch.bfloat16 if on_gpu else torch.float32,
            attn_implementation=attn,
        )
        return self._models[kind]

    def _generation(self, voice: Voice | None) -> dict[str, Any]:
        params = {**self.options.get("generation", {}), **(voice.params if voice else {})}
        return {k: params[k] for k in _GENERATION_KEYS if k in params}

    @staticmethod
    def _language(code: str) -> str:
        return LANGUAGE_NAMES.get(code, "Auto")

    def synthesize(self, line: Line, voice: Voice, seed: int) -> tuple[np.ndarray, int]:
        seed_everything(seed)
        language = self._language(line.language)
        if voice.preset:
            if voice.preset not in PRESETS:
                raise TTSError(f"Unknown Qwen3 preset '{voice.preset}'. Choose one of: {', '.join(PRESETS)}")
            style = voice.params.get("style", "")
            spoken = instruction(line.emotion, line.intensity, line.direction)
            wavs, sr = self._model("CustomVoice").generate_custom_voice(
                text=line.text,
                speaker=voice.preset,
                language=language,
                instruct=" ".join(p for p in (style, spoken) if p) or None,
                **self._generation(voice),
            )
        elif voice.reference is not None:
            model = self._model("Base")
            prompt = self._prompts.get(voice.fingerprint)
            if prompt is None:
                # With a transcript, ICL mode copies the voice closely; without one, use the speaker embedding.
                in_context = bool(voice.reference_text) and voice.params.get("icl", True)
                prompt = model.create_voice_clone_prompt(
                    ref_audio=str(voice.reference),
                    ref_text=voice.reference_text if in_context else None,
                    x_vector_only_mode=not in_context,
                )
                self._prompts[voice.fingerprint] = prompt
            wavs, sr = model.generate_voice_clone(
                text=line.text, language=language, voice_clone_prompt=prompt, **self._generation(voice)
            )
        else:
            raise TTSError(
                f"Voice '{voice.id}' has no reference clip or preset. Run `abcraft voices design`, put a clip at "
                f"voices/{voice.id}.wav, or set `preset:` in cast.yaml ({', '.join(PRESETS)})."
            )
        return np.asarray(wavs[0], dtype=np.float32), int(sr)

    def design(self, description: str, text: str, language: str, seed: int) -> tuple[np.ndarray, int]:
        seed_everything(seed)
        wavs, sr = self._model("VoiceDesign").generate_voice_design(
            text=text, instruct=description, language=self._language(language), **self._generation(None)
        )
        return np.asarray(wavs[0], dtype=np.float32), int(sr)

    def _drop(self, kind: str) -> None:
        self._models.pop(kind, None)
        if kind == "Base":
            self._prompts.clear()
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def unload(self) -> None:
        for kind in list(self._models):
            self._drop(kind)
