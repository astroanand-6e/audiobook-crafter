"""Regression tests for issues found in review: edit/re-run workflows, QA on cached clips, levelling."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audiobook_crafter import pipeline
from audiobook_crafter.casting.cast import auto_cast
from audiobook_crafter.models import Character, CharacterMap, VoiceProfile
from audiobook_crafter.project import Project, primary_language
from audiobook_crafter.synth.qa import ClipChecker
from audiobook_crafter.text.normalize import normalize_for_tts
from audiobook_crafter.tts.base import TTSError

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def mock_project(tmp_path: Path) -> Project:
    project = Project(tmp_path / "book")
    project.config.tts.engine = "mock"
    project.config.llm.model = "none"
    project.save_config()
    return project


def test_reassigned_lines_get_their_own_voice(tmp_path: Path, book_pdf: Path) -> None:
    project = mock_project(tmp_path)
    pipeline.run(project, book_pdf, [1], preview=4)
    script = project.load_script(1)
    assert script is not None
    for segment in script.segments:
        if segment.kind == "dialogue":
            segment.speaker = "the_stranger"  # a hand edit introducing a new character id
    project.save_script(script)
    pipeline.cast(project)
    characters = project.load_characters()
    assert characters.get("the_stranger") and characters.get("the_stranger").lines > 0
    assert project.load_cast().voices["the_stranger"].design  # a voice of its own, not the narrator's


def test_dropped_clip_replaces_an_automatic_voice(tmp_path: Path) -> None:
    project = mock_project(tmp_path)
    characters = CharacterMap(characters=[Character(id="bo", name="Bo", gender="male", lines=5)])
    cast = project.load_cast()
    auto_cast(project, characters, cast)
    assert cast.voices["bo"].design and not cast.voices["bo"].reference
    cast.voices["bo"].params["temperature"] = 0.7
    project.voices_dir.mkdir(parents=True, exist_ok=True)
    sf.write(str(project.voices_dir / "bo.wav"), np.zeros(24000, dtype=np.float32), 24000)
    auto_cast(project, characters, cast)
    assert cast.voices["bo"].reference == "voices/bo.wav"
    assert cast.voices["bo"].params == {"temperature": 0.7}  # your tweaks survive the swap
    cast.voices["bo"] = VoiceProfile(preset="Ryan")  # chosen by hand: a clip on disk doesn't override it
    auto_cast(project, characters, cast)
    assert cast.voices["bo"].preset == "Ryan"


@needs_ffmpeg
def test_asr_checks_cached_clips_and_retake_redoes_failures(
    tmp_path: Path, book_pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = mock_project(tmp_path)
    pipeline.run(project, book_pdf, [1], preview=6)

    def fake_transcribe(self, audio, sr):  # pretend Whisper mishears one line
        return "completely different words here" if len(audio) % 7 == 0 else None

    monkeypatch.setattr(ClipChecker, "_transcribe", fake_transcribe)
    project.config.qa.asr = True
    project.save_config()
    pipeline.run(project, book_pdf, [1], preview=6)
    metas = [json.loads(p.read_text()) for p in (project.cache_dir / "tts").rglob("*.json")]
    checked = [m for m in metas if len(m["text"].split()) >= 3]
    assert checked and all("wer" in m for m in checked)

    failed_before = {m["text"]: m["takes"] for m in metas if m.get("issue")}
    project.config.qa.retake_failed = True
    result = pipeline.run(project, book_pdf, [1], preview=6)
    after = {
        m["text"]: m for m in (json.loads(p.read_text()) for p in (project.cache_dir / "tts").rglob("*.json"))
    }
    for text, takes in failed_before.items():
        assert after[text]["takes"] > takes  # new takes, not the same seeds again
    assert "retake_failed" not in (project.root / "project.yaml").read_text()  # never persisted
    assert result.audio


def test_levelling_never_pushes_a_voice_past_full_scale(tmp_path: Path) -> None:
    from audiobook_crafter.assemble import speaker_gains
    from audiobook_crafter.synth.runner import Clip

    project = mock_project(tmp_path)
    sr = 24000
    quiet = np.full(sr, 0.01, dtype=np.float32)
    quiet[100] = 0.9  # one shouted moment
    loud = (0.3 * np.sin(np.linspace(0, 400, sr))).astype(np.float32)
    clips = []
    for name, audio in (("quiet", quiet), ("loud", loud)):
        path = project.root / f"{name}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), audio, sr, subtype="FLOAT")
        clips.append(Clip(segment=name, piece=0, kind="dialogue", block=0, voice=name, path=f"{name}.wav"))
    gains = speaker_gains(project, clips)
    assert gains["quiet"] * 0.9 <= 0.98 + 1e-6


def test_numbers_in_other_languages() -> None:
    assert normalize_for_tts("Er zahlte 1.500 Euro und 3,5 Prozent.", "de") == (
        "Er zahlte eintausendfünfhundert Euro und 3,5 Prozent."
    )
    assert normalize_for_tts("Il y avait 5 000 personnes.", "fr") == "Il y avait cinq mille personnes."
    assert normalize_for_tts("Tenía 12.000 pesos.", "es") == "Tenía doce mil pesos."


def test_regional_language_codes() -> None:
    assert primary_language("en-GB") == "en"
    assert primary_language("pt_BR") == "pt"
    project = Project(Path("unused"))
    project.config.language = "en-GB"
    assert project.language == "en"


def test_worker_that_dies_raises_a_clean_error() -> None:
    from audiobook_crafter.tts.worker import WorkerEngine

    engine = WorkerEngine("no-such-engine", {"python": sys.executable})
    with pytest.raises(TTSError):
        engine.load()
    engine.unload()
