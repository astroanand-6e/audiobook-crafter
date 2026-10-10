from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fakes import FakeLLM, green_seal_handler

from audiobook_crafter import pipeline
from audiobook_crafter.assemble import gap_after, timeline
from audiobook_crafter.config import PacingConfig
from audiobook_crafter.models import NARRATOR, Cast, VoiceProfile
from audiobook_crafter.project import Project
from audiobook_crafter.synth.qa import ClipChecker, word_error_rate
from audiobook_crafter.synth.runner import Clip
from audiobook_crafter.tts.emotion import chatterbox_controls, instruction, turbo_prefix

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def mock_project(tmp_path: Path) -> Project:
    project = Project(tmp_path / "book")
    project.config.tts.engine = "mock"
    project.config.llm.model = "none"
    project.save_config()
    return project


@needs_ffmpeg
def test_one_chapter_end_to_end(tmp_path: Path, book_pdf: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = mock_project(tmp_path)
    monkeypatch.setattr(pipeline, "create_llm", lambda config: FakeLLM(green_seal_handler))

    result = pipeline.run(project, book_pdf, [1])

    # character map + script
    characters = {c.id: c for c in project.load_characters().characters}
    assert {"clara_hartwell", "mr_hartwell"} <= set(characters)
    script = project.load_script(1)
    assert script and script.segments[0].kind == "heading"

    # casting: narrator + every speaking character got a voice, and it was designed into a clip
    cast = project.load_cast()
    for voice_id in (NARRATOR, "clara_hartwell", "mr_hartwell"):
        _, profile = cast.resolve(voice_id)
        assert profile.reference and project.resolve(profile.reference).exists()
    assert set(result.designed) >= {NARRATOR, "clara_hartwell"}

    # audio: a mastered MP3 and an M4B with one chapter marker
    audio = result.audio[0]
    assert audio.mp3.exists() and audio.seconds > 10
    assert result.m4b and result.m4b.exists()
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_chapters", "-show_format", "-of", "json", str(result.m4b)],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    info = json.loads(probe.stdout)
    assert len(info["chapters"]) == 1
    assert info["chapters"][0]["tags"]["title"] == "Chapter One: The Letter"
    assert float(info["format"]["duration"]) == pytest.approx(audio.seconds, abs=1.0)

    # a second run is fully cached: no TTS calls
    calls = []
    from audiobook_crafter.tts.mock import MockEngine

    original = MockEngine.synthesize
    monkeypatch.setattr(MockEngine, "synthesize", lambda self, *a: calls.append(1) or original(self, *a))
    pipeline.run(project, book_pdf, [1])
    assert calls == []

    # editing one voice re-renders only that voice's lines
    cast.voices["mr_hartwell"].params["exaggeration"] = 0.7
    project.save_cast(cast)
    pipeline.run(project, book_pdf, [1])
    spoken = {(s.text, s.emotion, s.intensity) for s in script.segments if s.speaker == "mr_hartwell"}
    assert len(calls) == len(spoken)  # identical lines share one cached take


@needs_ffmpeg
def test_preview_and_loudness(tmp_path: Path, book_pdf: Path) -> None:
    project = mock_project(tmp_path)
    result = pipeline.run(project, book_pdf, [2], preview=5)
    master = result.audio[0].master
    assert sf.info(str(master)).samplerate == 44100
    stats = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(master), "-af", "ebur128", "-f", "null", "-"],
        capture_output=True, text=True,
    ).stderr  # fmt: skip
    integrated = float(stats.split("Integrated loudness:")[1].split("I:")[1].split("LUFS")[0])
    assert integrated == pytest.approx(project.config.mastering.loudness_lufs, abs=1.5)


def test_cast_is_stable_and_respects_user_clips(tmp_path: Path) -> None:
    from audiobook_crafter.casting.cast import auto_cast
    from audiobook_crafter.models import Character, CharacterMap

    project = mock_project(tmp_path)
    project.voices_dir.mkdir(parents=True)
    sf.write(str(project.voices_dir / "ana.wav"), np.zeros(24000, dtype=np.float32), 24000)
    characters = CharacterMap(
        characters=[
            Character(id="ana", name="Ana", gender="female", lines=9),
            Character(id="bo", name="Bo", gender="male", age="elderly", voice="Gravelly, slow.", lines=4),
            Character(id="cy", name="Cy", gender="male", lines=1),
            Character(id="di", name="Di", lines=0),
        ]
    )
    cast = Cast()
    auto_cast(project, characters, cast)
    assert cast.voices["ana"].reference == "voices/ana.wav"
    assert cast.voices["bo"].design and cast.voices["bo"].design.startswith("An elderly man. Gravelly, slow.")
    assert cast.voices["cy"].same_as and cast.voices["cy"].same_as.startswith("extra_male")
    assert "di" not in cast.voices  # never spoke: not cast yet
    assert cast.voices[cast.voices["cy"].same_as].design

    before = cast.model_dump()
    auto_cast(project, characters, cast)
    assert cast.model_dump() == before  # re-casting never changes an existing voice

    characters.characters[2].lines = 5  # Cy became important: promoted to his own voice
    auto_cast(project, characters, cast)
    assert cast.voices["cy"].design and not cast.voices["cy"].same_as


def test_cast_resolve_follows_links() -> None:
    cast = Cast(voices={"a": VoiceProfile(same_as="b"), "b": VoiceProfile(preset="Ryan")})
    assert cast.resolve("a") == ("b", VoiceProfile(preset="Ryan"))
    assert cast.resolve("nobody")[0] == NARRATOR
    with pytest.raises(ValueError):
        Cast(voices={"a": VoiceProfile(same_as="b"), "b": VoiceProfile(same_as="a")}).resolve("a")


def test_pacing() -> None:
    pacing = PacingConfig()

    def clip(segment: str, kind: str, block: int, voice: str, text: str, pause: int | None = None) -> Clip:
        return Clip(
            segment=segment,
            piece=0,
            kind=kind,
            block=block,
            voice=voice,
            text=text,
            path="x.wav",
            pause_ms=pause,
        )

    a = clip("s1", "dialogue", 1, "ana", "“Who's there?”")
    b = clip("s2", "dialogue", 2, "bo", "“Only me.”")
    assert gap_after(a, b, pacing) == pacing.dialogue_turn_ms + 60
    tag = clip("s3", "narration", 2, NARRATOR, "he said.")
    assert gap_after(b, tag, pacing) == pacing.same_paragraph_ms
    cut = clip("s4", "dialogue", 3, "ana", "“But I—”")
    assert gap_after(cut, b, pacing) == 120
    pause = Clip(segment="p", piece=0, kind="pause", block=4)
    items = timeline([a, pause, b], pacing)
    assert [gap for _, gap in items] == [pacing.scene_break_ms, 0]


def test_emotion_controls() -> None:
    assert chatterbox_controls("neutral", 0.3) == (0.5, 0.5)
    exaggeration, cfg = chatterbox_controls("angry", 1.0)
    assert exaggeration > 0.85 and cfg < 0.35
    assert chatterbox_controls("whispering", 1.0)[0] < 0.5
    assert (
        instruction("angry", 0.9, "through clenched teeth")
        == "Speak extremely angry, through clenched teeth."
    )
    assert instruction("neutral", 0.2) == ""
    assert turbo_prefix(["sigh", "clear_throat"]) == "[sigh] [clear throat] "


def test_qa_checks() -> None:
    from audiobook_crafter.config import QAConfig

    checker = ClipChecker(QAConfig(), "en")
    text = "This sentence should take roughly three seconds to say."
    assert checker.check(text, np.zeros(int(3.6 * 24000), dtype=np.float32) + 0.1, 24000).issue == ""
    assert "too long" in checker.check(text, np.ones(24000 * 20, dtype=np.float32), 24000).issue
    assert "too short" in checker.check(text, np.ones(6000, dtype=np.float32), 24000).issue
    assert word_error_rate("the cat sat on the mat", "the cat sat on a mat") == pytest.approx(1 / 6)


@needs_ffmpeg
def test_preview_is_kept_out_of_the_audiobook(tmp_path: Path, book_pdf: Path) -> None:
    project = mock_project(tmp_path)
    result = pipeline.run(project, book_pdf, [1], preview=4)
    assert result.m4b is None and "(preview)" in result.audio[0].mp3.name
    assert not list(project.render_dir.glob("*.master.wav"))


@needs_ffmpeg
def test_pages_override_one_chapter_and_stale_work_is_redone(tmp_path: Path, book_pdf: Path) -> None:
    from audiobook_crafter.assemble import build_audiobook

    project = mock_project(tmp_path)
    pipeline.run(project, book_pdf, [2], preview=3)
    first = project.load_script(2)
    full = pipeline.run(project, book_pdf, [2])
    assert full.m4b is not None

    # Detection "got chapter 2 wrong": point it at pages 3-4 (really chapter one's text).
    book = pipeline.ingest(project, book_pdf, pages=(3, 4), chapter=2)
    assert book.chapter(1).title == "Chapter One: The Letter"  # other chapters untouched
    assert book.chapter(2).blocks[0].text == "CHAPTER ONE"
    assert project.config.chapter_pages == {2: [3, 4]}
    assert build_audiobook(project, book) is None  # the old chapter-2 master no longer matches its text

    pipeline.run(project, book_pdf, [2], preview=3)
    second = project.load_script(2)
    assert first and second and first.source != second.source  # re-analysed, not reused
    with pytest.raises(ValueError, match="exactly one"):
        pipeline.run(project, book_pdf, [1, 2], pages=(3, 4))


@needs_ffmpeg
def test_preview_never_replaces_the_full_chapter(tmp_path: Path, book_pdf: Path) -> None:
    from audiobook_crafter.assemble import assemble_chapter
    from audiobook_crafter.synth.runner import load_manifest

    project = mock_project(tmp_path)
    full = pipeline.run(project, book_pdf, [1]).audio[0]
    pipeline.run(project, book_pdf, [1], preview=3)
    assert len(load_manifest(project, 1)) > len(load_manifest(project, 1, preview=True))
    again = assemble_chapter(project, project.load_book(), 1)
    assert again.seconds == pytest.approx(full.seconds, abs=0.05)
