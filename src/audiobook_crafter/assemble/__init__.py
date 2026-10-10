"""Stage 5: lay clips on a timeline with natural pauses, even out voices, master, export MP3 + M4B."""

from __future__ import annotations

import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from ..audio import ffmpeg
from ..audio.dsp import active_rms_db, room_tone
from ..config import PacingConfig
from ..ingest import extract_cover
from ..models import Book
from ..project import Project
from ..synth.runner import Clip, load_manifest

_SENTENCE_END = (".", "!", "?", "…", "”", '"', "’", "।", "。", "！", "？")


@dataclass
class ChapterAudio:
    index: int
    title: str
    master: Path
    mp3: Path
    seconds: float


def gap_after(prev: Clip, nxt: Clip, pacing: PacingConfig) -> int:
    """Silence between two clips, in milliseconds."""
    if prev.pause_ms is not None:
        return prev.pause_ms
    if prev.segment == nxt.segment:
        gap = pacing.sentence_ms if prev.text.rstrip().endswith(_SENTENCE_END) else pacing.clause_ms
    elif prev.kind == "heading":
        gap = pacing.after_heading_ms
    elif nxt.kind == "heading":
        gap = pacing.after_heading_ms
    elif prev.block == nxt.block:
        gap = pacing.same_paragraph_ms
    elif prev.kind == "dialogue" and nxt.kind == "dialogue" and prev.voice != nxt.voice:
        gap = pacing.dialogue_turn_ms
    else:
        gap = pacing.paragraph_ms
    text = prev.text.rstrip("”\"’' ")
    if text.endswith(("—", "–", "-")):
        return min(gap, 120)  # an interruption: the next voice cuts straight in
    if text.endswith(("?", "!")):
        gap += 60
    return gap


def timeline(clips: list[Clip], pacing: PacingConfig) -> list[tuple[Clip, int]]:
    """[(clip, silence after it in ms)], with scene breaks folded into the gap."""
    items: list[tuple[Clip, int]] = []
    pending_break: int | None = None
    for clip in clips:
        if clip.kind == "pause":
            pending_break = clip.pause_ms or pacing.scene_break_ms
            continue
        if clip.path is None:
            continue
        if items:
            prev, _ = items[-1]
            items[-1] = (prev, pending_break if pending_break is not None else gap_after(prev, clip, pacing))
        items.append((clip, 0))
        pending_break = None
    return items


def _read(project: Project, clip: Clip, sample_rate: int) -> np.ndarray:
    path = project.resolve(clip.path or "")
    info = sf.info(str(path))
    if info.samplerate != sample_rate:
        converted = project.cache_dir / "resampled" / f"{path.stem}-{sample_rate}.wav"
        if not converted.exists():
            converted.parent.mkdir(parents=True, exist_ok=True)
            ffmpeg.resample(path, converted, sample_rate)
        path = converted
    audio, _ = sf.read(str(path), dtype="float32", always_2d=False)
    return audio if audio.ndim == 1 else audio.mean(axis=1)


def speaker_gains(project: Project, clips: list[Clip]) -> dict[str, float]:
    """Bring every voice to the same average level, keeping each voice's own whispers and shouts."""
    levels: dict[str, list[float]] = defaultdict(list)
    for clip in clips:
        if clip.path:
            audio, sr = sf.read(str(project.resolve(clip.path)), dtype="float32")
            levels[clip.voice].append(active_rms_db(audio, sr))
    medians = {voice: statistics.median(values) for voice, values in levels.items() if values}
    if not medians:
        return {}
    target = statistics.median(medians.values())
    return {voice: 10 ** (max(-12.0, min(12.0, target - level)) / 20) for voice, level in medians.items()}


def render_voice_track(project: Project, clips: list[Clip], out: Path) -> float:
    config = project.config
    rates = [sf.info(str(project.resolve(c.path))).samplerate for c in clips if c.path]
    if not rates:
        raise ValueError("Nothing to render: the chapter produced no audio clips")
    sample_rate = statistics.mode(rates)
    gains = speaker_gains(project, clips) if config.mastering.level_speakers else {}
    rng = np.random.default_rng(config.tts.seed)
    tone = config.mastering.room_tone_db

    def silence(ms: int) -> np.ndarray:
        samples = int(sample_rate * ms / 1000)
        return room_tone(samples, tone, rng) if tone is not None else np.zeros(samples, dtype=np.float32)

    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with sf.SoundFile(str(out), "w", samplerate=sample_rate, channels=1, subtype="PCM_24") as track:

        def write(audio: np.ndarray) -> None:
            nonlocal written
            track.write(audio)
            written += len(audio)

        write(silence(config.pacing.lead_in_ms))
        for clip, gap in timeline(clips, config.pacing):
            write(np.clip(_read(project, clip, sample_rate) * gains.get(clip.voice, 1.0), -1.0, 1.0))
            write(silence(gap))
        write(silence(config.pacing.tail_ms))
    return written / sample_rate


def _safe(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "", name).strip().rstrip(".")
    return re.sub(r"\s+", " ", name)[:80] or "untitled"


def chapter_title(book: Book, index: int) -> str:
    return book.chapter(index).title or f"Chapter {index}"


def assemble_chapter(project: Project, book: Book, index: int) -> ChapterAudio:
    clips = load_manifest(project, index)
    voice = project.render_dir / f"ch{index:03d}.voice.wav"
    seconds = render_voice_track(project, clips, voice)
    master = ffmpeg.master(voice, project.render_dir / f"ch{index:03d}.master.wav", project.config.mastering)
    title = chapter_title(book, index)
    project.output_dir.mkdir(parents=True, exist_ok=True)
    mp3 = project.output_dir / f"{_safe(book.title)} - {index:02d} - {_safe(title)}.mp3"
    tags = {"title": title, "album": book.title, "artist": book.author, "album_artist": book.author,
            "track": str(index), "genre": "Audiobook"}  # fmt: skip
    ffmpeg.encode_mp3(master, mp3, project.config.mastering, tags)
    return ChapterAudio(index=index, title=title, master=master, mp3=mp3, seconds=seconds)


def build_audiobook(project: Project, book: Book) -> Path | None:
    """One M4B with chapter markers from every chapter mastered so far."""
    masters = sorted(project.render_dir.glob("ch*.master.wav"))
    if not masters:
        return None
    chapters = []
    for path in masters:
        index = int(re.match(r"ch(\d+)", path.name).group(1))  # type: ignore[union-attr]
        chapters.append((path, chapter_title(book, index)))
    cover = project.render_dir / "cover.png"
    if not cover.exists() and book.source:
        extract_cover(Path(book.source), cover)
    tags = {"title": book.title, "album": book.title, "artist": book.author, "album_artist": book.author,
            "genre": "Audiobook"}  # fmt: skip
    out = project.output_dir / f"{_safe(book.title)}.m4b"
    return ffmpeg.build_m4b(chapters, out, project.config.mastering, tags, cover if cover.exists() else None)
