"""Stage 2, the LLM brain: build the character map, then decide who says every line and how."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable

from ..llm import LLM
from ..models import Book, ChapterScript, Character, CharacterMap, Segment
from ..project import Project
from ..text.quotes import QuoteStyle, detect_quote_style, is_speech_tag
from .attribution import Attribution, Performance, attribute_with_llm
from .characters import chunk_passages, update_character_map
from .heuristics import HeuristicAttributor, detect_narration
from .segment import segment_chapter

Progress = Callable[[str, int, int], None]


def book_quote_style(book: Book) -> QuoteStyle:
    paragraphs = [b.text for c in book.chapters for b in c.blocks if b.kind == "paragraph"]
    return detect_quote_style(paragraphs[:400])


def scan_characters(
    project: Project, book: Book, llm: LLM, chapters: list[int], progress: Progress | None = None
) -> CharacterMap:
    """Read whole chapters (or the whole book) just to build the character map."""
    characters = project.load_characters()
    paragraphs = [b.text for i in chapters for b in book.chapter(i).blocks if b.kind == "paragraph"]
    passages = chunk_passages(paragraphs, project.config.llm.chunk_chars)
    update_character_map(
        llm,
        characters,
        passages,
        title=book.title,
        author=book.author,
        language=book.language,
        on_progress=(lambda n, total: progress("characters", n, total)) if progress else None,
    )
    project.save_characters(characters)
    return characters


def drop_speech_tags(segments: list[Segment]) -> list[Segment]:
    """Drama mode: "she said." between lines is redundant once every character has their own voice."""
    kept: list[Segment] = []
    for i, segment in enumerate(segments):
        neighbours = [segments[j] for j in (i - 1, i + 1) if 0 <= j < len(segments)]
        next_to_dialogue = any(n.kind == "dialogue" and n.block == segment.block for n in neighbours)
        if segment.kind == "narration" and next_to_dialogue and is_speech_tag(segment.text):
            continue
        kept.append(segment)
    return kept


def speaker_counts(project: Project) -> Counter[str]:
    """Lines of dialogue per speaker id across every analysed chapter."""
    counts: Counter[str] = Counter()
    for path in sorted((project.root / "scripts").glob("ch*.json")):
        script = ChapterScript.model_validate_json(path.read_text(encoding="utf-8"))
        counts.update(script.speaker_counts())
    return counts


def recount_lines(project: Project, characters: CharacterMap) -> None:
    counts = speaker_counts(project)
    for character in characters.characters:
        character.lines = counts.get(character.id, 0)


def analyze_chapter(
    project: Project,
    book: Book,
    index: int,
    llm: LLM | None,
    *,
    scan: bool = True,
    progress: Progress | None = None,
) -> ChapterScript:
    config = project.config
    chapter = book.chapter(index)
    segmented = segment_chapter(chapter, book_quote_style(book), config.style.announce_chapter_titles)
    characters = project.load_characters()

    given: dict[str, str] = {}
    for segment_id, raw_name in segmented.given_speakers.items():
        name = raw_name.title() if raw_name.isupper() else raw_name
        character = characters.find(name) or characters.add(Character(id="", name=name))
        given[segment_id] = character.id

    paragraphs = [b.text for b in chapter.blocks if b.kind == "paragraph"]
    if llm is not None and scan:
        update_character_map(
            llm,
            characters,
            chunk_passages(paragraphs, config.llm.chunk_chars),
            title=book.title,
            author=book.author,
            language=book.language,
            on_progress=(lambda n, total: progress("characters", n, total)) if progress else None,
        )
    if characters.narration == "unknown":
        characters.narration = detect_narration(paragraphs)  # type: ignore[assignment]

    fallback = HeuristicAttributor(characters, discover=llm is None).attribute(segmented)
    fallback.update(given)
    if llm is not None:
        attribution = attribute_with_llm(
            llm,
            segmented,
            characters,
            max_chars=config.llm.chunk_chars,
            fallback=fallback,
            given=given,
            on_progress=(lambda n, total: progress("lines", n, total)) if progress else None,
        )
    else:
        attribution = Attribution(
            lines={sid: Performance(speaker=spk) for sid, spk in fallback.items()}, moods={}
        )

    for segment in segmented.segments:
        if segment.kind == "dialogue" and segment.id in attribution.lines:
            line = attribution.lines[segment.id]
            segment.speaker = line.speaker
            segment.emotion = line.emotion  # coerced to the known emotions on assignment
            segment.intensity = line.intensity
            segment.direction = segment.direction or line.direction
            segment.tags = line.tags
        elif (
            segment.kind == "narration" and config.style.narrator_mood and segment.block in attribution.moods
        ):
            emotion, intensity = attribution.moods[segment.block]
            segment.emotion = emotion
            segment.intensity = min(0.5, 0.6 * intensity)  # the narrator colours a scene, never acts it out

    segments = segmented.segments
    if config.style.mode == "drama":
        segments = drop_speech_tags(segments)
    script = ChapterScript(chapter=index, title=chapter.title, segments=segments)
    project.save_script(script)
    recount_lines(project, characters)
    project.save_characters(characters)
    return script
