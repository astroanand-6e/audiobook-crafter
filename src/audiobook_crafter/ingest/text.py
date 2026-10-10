"""Plain text and Markdown books, plus speaker-labelled scripts and transcripts (TXT/MD/SRT/VTT).

A "script" is any text where most lines look like `NAME: line`, `NAME (whispering): line` or
`[NAME] line`. Speakers then come from the source and the LLM only has to describe the characters.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..models import NARRATOR, Block, Book, Chapter
from .common import (
    classify_chapters,
    clean_text,
    detect_language,
    is_scene_break,
    join_lines,
    looks_like_chapter_heading,
    title_from_headings,
)

_GUTENBERG_START = re.compile(
    r"^\*{3}\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*$", re.MULTILINE | re.IGNORECASE
)
_GUTENBERG_END = re.compile(
    r"^\*{3}\s*END OF (?:THE|THIS) PROJECT GUTENBERG.*$", re.MULTILINE | re.IGNORECASE
)
_MD_HEADING = re.compile(r"^#{1,3}\s+(.+?)\s*#*\s*$")
_SCRIPT_LINE = re.compile(
    r"^(?P<name>[^\W\d_][\w .'’-]{0,38}?)\s*(?:\((?P<dir>[^)]{1,60})\))?\s*:\s+(?P<text>\S.*)$"
)
_BRACKET_LINE = re.compile(r"^\[(?P<name>[^\]]{1,40})\]\s*(?:\((?P<dir>[^)]{1,60})\))?\s*(?P<text>\S.*)$")
_TIMESTAMP = re.compile(r"^\s*(?:\d+:)?\d{1,2}:\d{2}[.,]\d{1,3}\s*-->\s*(?:\d+:)?\d{1,2}:\d{2}[.,]\d{1,3}")
_VTT_VOICE = re.compile(r"<v(?:\.[\w.-]+)?\s+([^>]+)>(.*?)(?:</v>|$)", re.DOTALL)
_TAGS = re.compile(r"<[^>]+>")
_NARRATOR_NAMES = {"narrator", "narration", "narr", "voiceover", "voice over", "vo"}


def _speaker(name: str) -> str:
    name = clean_text(name).strip(" -")
    return NARRATOR if name.lower() in _NARRATOR_NAMES else name


def _parse_script_line(line: str) -> tuple[str, str | None, str] | None:
    match = _BRACKET_LINE.match(line) or _SCRIPT_LINE.match(line)
    if not match:
        return None
    return _speaker(match["name"]), match["dir"], match["text"].strip()


def _is_script(lines: list[str]) -> bool:
    content = [line for line in lines if line.strip() and not _MD_HEADING.match(line)]
    if len(content) < 4:
        return False
    parsed = [_parse_script_line(line.strip()) for line in content]
    hits = [p for p in parsed if p]
    names = {p[0] for p in hits}
    return len(hits) >= 0.6 * len(content) and len(names) <= max(12, len(hits) // 4)


def _split_chapters(items: list[Block]) -> list[Chapter]:
    """Cut a flat list of blocks into chapters at heading blocks."""
    chapters: list[Chapter] = []
    current: list[Block] = []
    for block in items:
        starts_chapter = block.kind == "heading" and not (current and current[-1].kind == "heading")
        if starts_chapter and any(b.kind == "paragraph" for b in current):
            chapters.append(_chapter(current, len(chapters) + 1))
            current = []
        current.append(block)
    if current:
        chapters.append(_chapter(current, len(chapters) + 1))
    if len(chapters) > 1 and not any(b.kind == "heading" for b in chapters[0].blocks):
        chapters[0].kind = "front_matter"  # text before the first heading: title page, preface, licence
        chapters[0].title = "Front matter"
    return chapters


def _chapter(blocks: list[Block], index: int) -> Chapter:
    headings = []
    for block in blocks:
        if block.kind != "heading":
            break
        headings.append(block.text)
    return Chapter(index=index, title=title_from_headings(headings) or f"Part {index}", blocks=blocks)


def _prose_blocks(text: str) -> list[Block]:
    chunks = re.split(r"\n[ \t]*\n", text)
    if len(chunks) <= 2 and text.count("\n") > 20:  # one paragraph per line, no blank lines
        chunks = text.split("\n")
    blocks: list[Block] = []
    for chunk in chunks:
        lines = [line for line in chunk.split("\n") if line.strip()]
        if not lines:
            continue
        md = _MD_HEADING.match(lines[0].strip())
        if md:
            blocks.append(Block(kind="heading", text=clean_text(md.group(1))))
            lines = lines[1:]
            if not lines:
                continue
        paragraph = clean_text(join_lines(lines))
        if is_scene_break(paragraph):
            if blocks and blocks[-1].kind != "scene_break":
                blocks.append(Block(kind="scene_break"))
        elif len(lines) == 1 and looks_like_chapter_heading(paragraph):
            blocks.append(Block(kind="heading", text=paragraph))
        elif paragraph:
            blocks.append(Block(kind="paragraph", text=paragraph))
    return blocks


def _script_blocks(lines: list[str]) -> list[Block]:
    blocks: list[Block] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        md = _MD_HEADING.match(line)
        if md:
            blocks.append(Block(kind="heading", text=clean_text(md.group(1))))
            continue
        parsed = _parse_script_line(line)
        if parsed:
            speaker, direction, text = parsed
            blocks.append(
                Block(kind="paragraph", text=clean_text(text), speaker=speaker, direction=direction)
            )
        elif blocks and blocks[-1].kind == "paragraph":
            blocks[-1].text = f"{blocks[-1].text} {clean_text(line)}"  # wrapped continuation line
        else:
            blocks.append(Block(kind="paragraph", text=clean_text(line), speaker=NARRATOR))
    return blocks


def _subtitle_blocks(text: str, vtt: bool) -> list[Block]:
    """SRT/VTT cues -> blocks, merging consecutive cues from the same speaker."""
    blocks: list[Block] = []
    for cue in re.split(r"\n[ \t]*\n", text.replace("\r\n", "\n")):
        lines = [line.strip() for line in cue.split("\n") if line.strip()]
        while lines and not _TIMESTAMP.match(lines[0]):
            lines = lines[1:]  # cue number / id / WEBVTT header
        if not lines:
            continue
        body = " ".join(lines[1:])
        speaker: str | None = None
        voice = _VTT_VOICE.search(body) if vtt else None
        if voice:
            speaker, body = _speaker(voice.group(1)), voice.group(2)
        body = clean_text(_TAGS.sub("", body)).lstrip("- ")
        parsed = _parse_script_line(body) if speaker is None else None
        if parsed:
            speaker, _, body = parsed
        if not body:
            continue
        speaker = speaker or NARRATOR
        if blocks and blocks[-1].speaker == speaker:
            blocks[-1].text = f"{blocks[-1].text} {body}"
        else:
            blocks.append(Block(kind="paragraph", text=body, speaker=speaker))
    return blocks


def load_text(path: Path) -> Book:
    raw = Path(path).read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
    start, end = _GUTENBERG_START.search(raw), _GUTENBERG_END.search(raw)
    if start:
        raw = raw[start.end() : end.start() if end else None]

    suffix = Path(path).suffix.lower()
    if suffix in (".srt", ".vtt"):
        blocks = _subtitle_blocks(raw, vtt=suffix == ".vtt")
        chapters = [Chapter(index=1, title=Path(path).stem, blocks=blocks)]
    else:
        lines = raw.split("\n")
        blocks = _script_blocks(lines) if _is_script(lines) else _prose_blocks(raw)
        chapters = _split_chapters(blocks)
        if any(b.speaker for b in blocks):
            for n, chapter in enumerate(chapters, start=1):
                chapter.index = n  # scripts: keep every scene, no front-matter heuristics
        else:
            chapters = classify_chapters(chapters)

    sample = " ".join(b.text for b in blocks[:200])
    return Book(
        title=Path(path).stem.replace("_", " "),
        language=detect_language(sample),
        source=str(path),
        chapters=chapters,
    )
