"""Voice design: turn each character's description into a short reference clip, once.

Every later line is cloned from that clip, which is what keeps a character sounding the same from
the first chapter to the last."""

from __future__ import annotations

import logging
import zlib
from collections.abc import Callable

import soundfile as sf

from ..audio.dsp import clean_clip
from ..models import NARRATOR, Book, Cast, ChapterScript
from ..project import Project
from ..text.normalize import normalize_for_tts
from ..text.sentences import split_sentences
from ..tts.base import TTSEngine

log = logging.getLogger(__name__)

GENERIC_SAMPLE = {
    "en": "I have been meaning to tell you something for a while now. It is nothing terrible, I promise, "
    "but it matters to me, and I would rather you heard it from me than from anyone else.",
}
_CALM = {"neutral", "calm", "serious", "tender", "happy", "amused"}


def _fit(sentences: list[str], low: int = 140, high: int = 260) -> str:
    """Whole sentences adding up to roughly 10-15 seconds of speech."""
    text = ""
    for sentence in sentences:
        candidate = f"{text} {sentence}".strip()
        if len(candidate) > high and len(text) >= low:
            break
        text = candidate
        if len(text) >= high:
            break
    limit = high + 80
    if len(text) > limit:  # one very long sentence: end it at a word boundary
        text = text[:limit].rsplit(" ", 1)[0].rstrip(",;:—–- ") + "."
    return text


def _scripts(project: Project) -> list[ChapterScript]:
    return [
        ChapterScript.model_validate_json(p.read_text(encoding="utf-8"))
        for p in sorted((project.root / "scripts").glob("ch*.json"))
    ]


def sample_text(project: Project, book: Book, voice_id: str) -> str:
    """What the new voice says in its reference clip: its own calm lines if it has some, else book text."""
    language = project.language
    if voice_id != NARRATOR:
        lines = [
            normalize_for_tts(s.text, language)
            for script in _scripts(project)
            for s in script.segments
            if s.kind == "dialogue" and s.speaker == voice_id and s.emotion in _CALM and s.intensity <= 0.5
        ]
        text = _fit([sentence for line in lines for sentence in split_sentences(line)])
        if len(text) >= 80:
            return text
        if language in GENERIC_SAMPLE:
            return GENERIC_SAMPLE[language]
    narration = [
        normalize_for_tts(b.text, language)
        for c in book.chapters
        if c.kind == "chapter"
        for b in c.blocks
        if b.kind == "paragraph" and "“" not in b.text and '"' not in b.text and len(b.text) > 120
    ]
    text = _fit([sentence for paragraph in narration[:6] for sentence in split_sentences(paragraph)])
    return text or GENERIC_SAMPLE["en"]


def design_voices(
    project: Project,
    book: Book,
    cast: Cast,
    engine: TTSEngine,
    *,
    only: set[str] | None = None,
    force: bool = False,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> list[str]:
    """Create reference clips for every voice that has a `design` brief but no clip yet."""
    todo = [
        (voice_id, profile)
        for voice_id, profile in cast.voices.items()
        if profile.design
        and not profile.same_as
        and (force or not profile.reference)
        and (only is None or voice_id in only)
    ]
    out_dir = project.voices_dir / "designed"
    out_dir.mkdir(parents=True, exist_ok=True)
    language = project.language
    for n, (voice_id, profile) in enumerate(todo, start=1):
        if on_progress:
            on_progress(n, len(todo), voice_id)
        text = sample_text(project, book, voice_id)
        seed = project.config.tts.seed + zlib.crc32(voice_id.encode()) % 10_000
        audio, sr = engine.design(profile.design or "", text, language, seed)
        audio = clean_clip(audio, sr)
        if len(audio) < 5 * sr:  # some engines need 5 s+ of reference (Chatterbox Turbo)
            text = f"{text} {GENERIC_SAMPLE.get(language, text)}"
            audio, sr = engine.design(profile.design or "", text, language, seed)
            audio = clean_clip(audio, sr)
        path = out_dir / f"{voice_id}.wav"
        sf.write(str(path), audio, sr)
        path.with_suffix(".txt").write_text(text + "\n", encoding="utf-8")
        profile.reference = project.relative(path)
        profile.reference_text = text
        project.save_cast(cast)  # keep finished voices even if a later one fails
        log.info("Designed %s (%.1fs)", voice_id, len(audio) / sr)
    return [voice_id for voice_id, _ in todo]
