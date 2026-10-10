"""Engine adapters against stand-ins that copy the real libraries' signatures (qwen-tts 0.1.1,
chatterbox-tts 0.1.7), so a renamed or misspelled argument fails here instead of on a GPU box."""

from __future__ import annotations

import importlib.machinery
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from audiobook_crafter.tts import create_local_engine
from audiobook_crafter.tts.base import Line, Voice


def _module(name: str, **attrs: Any) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, None)
    module.__dict__.update(attrs)
    return module


def fake_torch() -> types.ModuleType:
    cuda = types.SimpleNamespace(
        is_available=lambda: False, manual_seed_all=lambda s: None, empty_cache=lambda: None
    )
    backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
    return _module(
        "torch", bfloat16="bf16", float32="f32", cuda=cuda, backends=backends, manual_seed=lambda s: None
    )


class FakeQwenModel:
    instances: list[FakeQwenModel] = []

    def __init__(self, repo: str, kwargs: dict[str, Any]):
        self.repo, self.load_kwargs, self.calls = repo, kwargs, []
        FakeQwenModel.instances.append(self)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs: Any) -> FakeQwenModel:
        return cls(pretrained_model_name_or_path, kwargs)

    def create_voice_clone_prompt(self, ref_audio, ref_text=None, x_vector_only_mode=False):
        self.calls.append(("prompt", ref_audio, ref_text, x_vector_only_mode))
        return ["prompt-item"]

    def generate_voice_clone(self, text, language=None, ref_audio=None, ref_text=None, x_vector_only_mode=False,
                             voice_clone_prompt=None, non_streaming_mode=False, **kwargs):  # fmt: skip
        self.calls.append(("clone", text, language, voice_clone_prompt, kwargs))
        return [np.zeros(2400, dtype=np.float32)], 24000

    def generate_voice_design(self, text, instruct, language=None, non_streaming_mode=True, **kwargs):
        self.calls.append(("design", text, instruct, language))
        return [np.zeros(2400, dtype=np.float32)], 24000

    def generate_custom_voice(
        self, text, speaker, language=None, instruct=None, non_streaming_mode=True, **kwargs
    ):
        self.calls.append(("custom", text, speaker, language, instruct))
        return [np.zeros(2400, dtype=np.float32)], 24000


@pytest.fixture
def qwen(monkeypatch: pytest.MonkeyPatch):
    FakeQwenModel.instances = []
    monkeypatch.setitem(sys.modules, "torch", fake_torch())
    monkeypatch.setitem(sys.modules, "qwen_tts", _module("qwen_tts", Qwen3TTSModel=FakeQwenModel))
    engine = create_local_engine("qwen3-tts", {}, "cpu")
    engine.load()
    return engine


def test_qwen_clone_reuses_prompt(qwen, tmp_path: Path) -> None:
    voice = Voice(id="clara", reference=tmp_path / "clara.wav", reference_text="Hello there.", fingerprint="f1",
                  params={"temperature": 0.7})  # fmt: skip
    for text in ("One.", "Two."):
        audio, sr = qwen.synthesize(Line(text=text, language="en", emotion="angry", intensity=0.8), voice, 1)
        assert sr == 24000 and audio.dtype == np.float32
    base = FakeQwenModel.instances[0]
    assert base.repo == "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    assert base.load_kwargs == {"device_map": "cpu", "dtype": "f32", "attn_implementation": "sdpa"}
    prompts = [c for c in base.calls if c[0] == "prompt"]
    assert prompts == [("prompt", str(tmp_path / "clara.wav"), "Hello there.", False)]  # built once
    clone = [c for c in base.calls if c[0] == "clone"][0]
    assert clone[1:4] == ("One.", "English", ["prompt-item"]) and clone[4] == {"temperature": 0.7}


def test_qwen_clip_without_transcript_uses_speaker_embedding(qwen, tmp_path: Path) -> None:
    qwen.synthesize(Line(text="Hi."), Voice(id="x", reference=tmp_path / "x.wav", fingerprint="f2"), 1)
    assert FakeQwenModel.instances[0].calls[0] == ("prompt", str(tmp_path / "x.wav"), None, True)


def test_qwen_preset_gets_emotion_instruction(qwen) -> None:
    line = Line(
        text="Get out!", language="fr", emotion="angry", intensity=0.9, direction="through clenched teeth"
    )
    qwen.synthesize(line, Voice(id="n", preset="Ryan", params={"style": "Deep and slow."}), 1)
    model = FakeQwenModel.instances[0]
    assert model.repo.endswith("-CustomVoice")
    assert model.calls == [("custom", "Get out!", "Ryan", "French",
                            "Deep and slow. Speak extremely angry, through clenched teeth.")]  # fmt: skip


def test_qwen_design_then_speak_swaps_models(qwen) -> None:
    qwen.design("An elderly man. Gravelly.", "Sample text.", "en", 3)
    design = FakeQwenModel.instances[0]
    assert design.repo == "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    assert design.calls == [("design", "Sample text.", "An elderly man. Gravelly.", "English")]
    qwen.synthesize(Line(text="Hi."), Voice(id="n", preset="Aiden"), 1)
    assert "VoiceDesign" not in qwen._models  # unloaded before the speaking model came in


def test_qwen_unknown_language_falls_back_to_auto(qwen) -> None:
    assert not qwen.supports("hi")
    qwen.synthesize(Line(text="नमस्ते", language="hi"), Voice(id="n", preset="Aiden"), 1)
    assert FakeQwenModel.instances[0].calls[0][3] == "Auto"


# ---------------------------------------------------------------- chatterbox


class FakeChatterbox:
    sr = 24000

    def __init__(self, kind: str):
        self.kind, self.conds, self.calls = kind, "builtin", []

    def prepare_conditionals(self, wav_fpath, exaggeration=0.5, norm_loudness=True):
        self.calls.append(("prepare", wav_fpath, exaggeration))
        self.conds = f"conds:{wav_fpath}"

    def generate(self, text, language_id=None, audio_prompt_path=None, exaggeration=0.5, cfg_weight=0.5,
                 temperature=0.8, repetition_penalty=2.0, min_p=0.05, top_p=1.0, top_k=1000):  # fmt: skip
        self.calls.append(("generate", text, language_id, exaggeration, cfg_weight, self.conds))

        class Wave:
            def squeeze(self, dim):
                return self

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return np.zeros(2400, dtype=np.float32)

        return Wave()


@pytest.fixture
def chatterbox_modules(monkeypatch: pytest.MonkeyPatch):
    made: dict[str, FakeChatterbox] = {}

    def loader(kind: str):
        def from_pretrained(device):
            made[kind] = FakeChatterbox(kind)
            return made[kind]

        return types.SimpleNamespace(from_pretrained=from_pretrained)

    monkeypatch.setitem(sys.modules, "torch", fake_torch())
    monkeypatch.setitem(sys.modules, "chatterbox", _module("chatterbox"))
    monkeypatch.setitem(
        sys.modules,
        "chatterbox.mtl_tts",
        _module("chatterbox.mtl_tts", ChatterboxMultilingualTTS=loader("mtl")),
    )
    monkeypatch.setitem(
        sys.modules,
        "chatterbox.tts_turbo",
        _module("chatterbox.tts_turbo", ChatterboxTurboTTS=loader("turbo")),
    )
    monkeypatch.setitem(sys.modules, "chatterbox.tts", _module("chatterbox.tts", ChatterboxTTS=loader("en")))
    return made


def test_chatterbox_multilingual(chatterbox_modules, tmp_path: Path) -> None:
    engine = create_local_engine("chatterbox-multilingual", {}, "cpu")
    engine.load()
    voice = Voice(id="ravi", reference=tmp_path / "ravi.wav", fingerprint="r")
    engine.synthesize(Line(text="नमस्ते", language="hi", emotion="angry", intensity=1.0), voice, 1)
    engine.synthesize(Line(text="फिर मिलेंगे", language="hi"), voice, 1)
    engine.synthesize(Line(text="Narration."), Voice(id="narrator"), 1)  # no clip: built-in voice
    model = chatterbox_modules["mtl"]
    assert [c for c in model.calls if c[0] == "prepare"] == [("prepare", str(tmp_path / "ravi.wav"), 0.5)]
    generated = [c for c in model.calls if c[0] == "generate"]
    assert generated[0][2] == "hi" and generated[0][3] > 0.85 and generated[0][4] < 0.35
    assert generated[1][3:5] == (0.5, 0.5)
    assert generated[2][5] == "builtin"


def test_chatterbox_turbo_tags(chatterbox_modules, tmp_path: Path) -> None:
    engine = create_local_engine("chatterbox-turbo", {}, "cpu")
    engine.load()
    engine.synthesize(
        Line(text="Fine.", tags=["sigh"]), Voice(id="a", reference=tmp_path / "a.wav", fingerprint="a"), 1
    )
    assert chatterbox_modules["turbo"].calls[-1][1] == "[sigh] Fine."


# ---------------------------------------------------------------- worker process


def test_worker_engine_round_trip() -> None:
    from audiobook_crafter.config import TTSConfig
    from audiobook_crafter.tts import create_engine
    from audiobook_crafter.tts.mock import MockEngine

    config = TTSConfig(engine="mock", engines={"mock": {"python": sys.executable}})
    engine = create_engine("mock", config)
    try:
        engine.load()
        assert engine.version == "mock-1" and engine.can_design
        voice = Voice(id="v", fingerprint="abc")
        line = Line(text="Across the process boundary.", emotion="happy", intensity=0.5)
        remote, sr = engine.synthesize(line, voice, 5)
        local, _ = MockEngine("mock").synthesize(line, voice, 5)
        assert sr == 24000 and np.allclose(remote, local, atol=1e-4)  # PCM round trip
        designed, _ = engine.design("A calm voice.", "Hello.", "en", 2)
        assert designed.size > 0
    finally:
        engine.unload()
    assert engine.proc is None


# ---------------------------------------------------------------- voxcpm


class FakeVoxCPM:
    made: list[FakeVoxCPM] = []

    def __init__(self, kwargs: dict[str, Any]):
        self.kwargs, self.calls = kwargs, []
        self.tts_model = types.SimpleNamespace(sample_rate=48000)
        FakeVoxCPM.made.append(self)

    @classmethod
    def from_pretrained(cls, hf_model_id="openbmb/VoxCPM2", load_denoiser=True, zipenhancer_model_id="x",
                        cache_dir=None, local_files_only=False, optimize=True, device=None, lora_config=None,
                        lora_weights_path=None, **kwargs):  # fmt: skip
        return cls({"id": hf_model_id, "denoiser": load_denoiser, "optimize": optimize, "device": device})

    def generate(self, text, prompt_wav_path=None, prompt_text=None, reference_wav_path=None, cfg_value=2.0,
                 inference_timesteps=10, min_len=2, max_len=4096, normalize=False, denoise=False,
                 retry_badcase=True, retry_badcase_max_times=3, retry_badcase_ratio_threshold=6.0):  # fmt: skip
        self.calls.append((text, reference_wav_path, prompt_wav_path, prompt_text, cfg_value))
        return np.zeros(4800, dtype=np.float32)


def test_voxcpm_style_prefix_on_cloned_voice(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    FakeVoxCPM.made = []
    monkeypatch.setitem(sys.modules, "torch", fake_torch())
    monkeypatch.setitem(sys.modules, "voxcpm", _module("voxcpm", VoxCPM=FakeVoxCPM))
    engine = create_local_engine("voxcpm2", {}, "cpu")
    engine.load()
    model = FakeVoxCPM.made[0]
    assert model.kwargs["id"] == "openbmb/VoxCPM2" and model.kwargs["denoiser"] is False
    voice = Voice(id="ravi", reference=tmp_path / "ravi.wav", reference_text="Namaste.", fingerprint="r")
    audio, sr = engine.synthesize(
        Line(text="Get out!", language="hi", emotion="angry", intensity=0.9), voice, 1
    )
    assert sr == 48000 and audio.dtype == np.float32
    assert model.calls[-1] == ("(extremely angry)Get out!", str(tmp_path / "ravi.wav"), None, None, 2.0)
    engine.synthesize(Line(text="Calm words."), voice, 1)
    assert model.calls[-1][0] == "Calm words."  # neutral: no prefix
    engine.design("A warm elderly woman.", "Hello there.", "en", 2)
    assert model.calls[-1][:2] == ("(A warm elderly woman.)Hello there.", None)
    assert engine.supports("hi")


def test_chatterbox_english_never_zeroes_cfg_and_old_versions_ignore_t3_model(
    chatterbox_modules, tmp_path: Path
) -> None:
    english = create_local_engine("chatterbox", {}, "cpu")
    english.load()
    voice = Voice(
        id="a",
        reference=tmp_path / "a.wav",
        fingerprint="a",
        params={"cfg_weight": 0.0, "cross_language": True},
    )
    english.synthesize(Line(text="Hello."), voice, 1)
    assert chatterbox_modules["en"].calls[-1][4] > 0  # 0 would break the English model's CFG batch

    multilingual = create_local_engine("chatterbox-multilingual", {"t3_model": "v3"}, "cpu")
    multilingual.load()  # PyPI 0.1.7's from_pretrained(device) has no t3_model: warn, don't crash
    multilingual.synthesize(Line(text="Bonjour.", language="fr"), voice, 1)
    assert chatterbox_modules["mtl"].calls[-1][4] == 0.0  # cross-language clip: cfg off for multilingual
