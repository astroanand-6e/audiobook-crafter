"""LLM speaker attribution and performance tagging.

The chapter is labelled in windows of whole paragraphs. Each quote carries a numeric id; the model
returns labels per id and never rewrites the book's text. Anything it skips falls back to the
rule-based attributor."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..llm import LLM
from ..models import NARRATOR, RESERVED_SPEAKERS, UNKNOWN, CharacterMap, Segment
from .prompts import attribution_schema, attribution_system
from .segment import SegmentedChapter

log = logging.getLogger(__name__)
CONTEXT_PARAGRAPHS = 3


@dataclass
class Performance:
    speaker: str
    emotion: str = "neutral"
    intensity: float = 0.3
    direction: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class Attribution:
    lines: dict[str, Performance]  # dialogue segment id -> performance
    moods: dict[int, tuple[str, float]]  # block index -> narration mood


def _render(segments: list[Segment], number: dict[str, int], known: dict[str, str]) -> str:
    parts: list[str] = []
    for segment in segments:
        if segment.kind == "dialogue":
            attrs = ""
            if segment.id in number:
                attrs += f' id="{number[segment.id]}"'
            if segment.id in known:
                attrs += f' speaker="{known[segment.id]}"'
            parts.append(f"<q{attrs}>{segment.text}</q>")
        elif segment.kind == "narration":
            parts.append(segment.text)
        elif segment.kind == "heading":
            parts.append(f"# {segment.text}")
        elif segment.kind == "pause":
            parts.append("* * *")
    return " ".join(parts)


def _windows(blocks: list[int], sizes: dict[int, int], max_chars: int) -> list[list[int]]:
    windows: list[list[int]] = []
    current: list[int] = []
    size = 0
    for block in blocks:
        if current and size + sizes[block] > max_chars:
            windows.append(current)
            current, size = [], 0
        current.append(block)
        size += sizes[block]
    if current:
        windows.append(current)
    return windows


def _speaker_list(characters: CharacterMap) -> str:
    rows = [f"- {NARRATOR}: not a character (signs, letters, read-out text)"]
    rows += [f"- {sid}" for sid in RESERVED_SPEAKERS if sid != NARRATOR]
    for c in sorted(characters.characters, key=lambda c: -c.lines):
        names = ", ".join(c.names())
        rows.append(f"- {c.id}: {names} ({c.gender}, {c.age})")
    return "\n".join(rows)


def _valid_speaker(value: Any, characters: CharacterMap) -> str:
    speaker = str(value or "").strip()
    if speaker in RESERVED_SPEAKERS or characters.get(speaker):
        return speaker
    found = characters.find(speaker)
    return found.id if found else UNKNOWN


def attribute_with_llm(
    llm: LLM,
    chapter: SegmentedChapter,
    characters: CharacterMap,
    *,
    max_chars: int,
    fallback: dict[str, str],
    given: dict[str, str],
    on_progress: Callable[[int, int], None] | None = None,
) -> Attribution:
    segments = chapter.segments
    dialogue = [s for s in segments if s.kind == "dialogue"]
    number = {s.id: n for n, s in enumerate(dialogue, start=1)}
    by_number = {number[s.id]: s for s in dialogue}
    by_block: dict[int, list[Segment]] = {}
    for segment in segments:
        by_block.setdefault(segment.block, []).append(segment)
    sizes = {b: sum(len(s.text) + 12 for s in items) for b, items in by_block.items()}
    block_order = list(by_block)

    result = Attribution(lines={}, moods={})
    known: dict[str, str] = dict(given)  # resolved speakers, shown in later context
    allowed = [*RESERVED_SPEAKERS, *sorted(characters.ids())]
    schema = attribution_schema(allowed)
    system = attribution_system()
    windows = _windows(block_order, sizes, max_chars)

    for w, window in enumerate(windows, start=1):
        wanted = [number[s.id] for b in window for s in by_block[b] if s.kind == "dialogue"]
        if not wanted:
            if on_progress:
                on_progress(w, len(windows))
            continue
        first = block_order.index(window[0])
        context_blocks = block_order[max(0, first - CONTEXT_PARAGRAPHS) : first]
        context = "\n".join(f"[P{b}] {_render(by_block[b], {}, known)}" for b in context_blocks)
        passage = "\n".join(f"[P{b}] {_render(by_block[b], number, given)}" for b in window)

        pending = list(wanted)
        for attempt in range(2):
            ids = ", ".join(map(str, pending))
            prompt = (
                f"Allowed speakers:\n{_speaker_list(characters)}\n\n"
                f"Narration: {characters.narration}"
                + (f" (the narrator is {characters.pov_character})" if characters.pov_character else "")
                + "\n\n"
                + (
                    f"Earlier in the chapter (already labelled, context only):\n{context}\n\n"
                    if context
                    else ""
                )
                + f"Passage:\n{passage}\n\nLabel these ids: {ids}"
            )
            try:
                data = llm.complete_json(system, prompt, schema)
            except Exception as err:  # noqa: BLE001 - one bad window must not sink the chapter
                log.warning("Attribution window %d failed (%s); using rule-based speakers", w, err)
                break
            for item in data.get("lines") or []:
                if not isinstance(item, dict):
                    continue
                try:
                    n = int(item.get("id"))
                except (TypeError, ValueError):
                    continue
                if n not in pending or n not in by_number:
                    continue
                segment = by_number[n]
                speaker = given.get(segment.id) or _valid_speaker(item.get("speaker"), characters)
                if speaker == UNKNOWN and fallback.get(segment.id, UNKNOWN) != UNKNOWN:
                    speaker = fallback[segment.id]
                result.lines[segment.id] = Performance(
                    speaker=speaker,
                    emotion=str(item.get("emotion") or "neutral"),
                    intensity=item.get("intensity", 0.3),
                    direction=str(item.get("direction") or "")[:80],
                    tags=list(item.get("tags") or []),
                )
                known[segment.id] = speaker
            if attempt == 0:
                window_blocks = set(window)
                for mood in data.get("narration") or []:
                    try:
                        block = int(mood.get("p"))
                        if block in window_blocks:
                            result.moods[block] = (
                                str(mood.get("emotion") or "neutral"),
                                float(mood.get("intensity") or 0),
                            )
                    except (TypeError, ValueError, AttributeError):
                        continue
            pending = [n for n in pending if by_number[n].id not in result.lines]
            if not pending:
                break
        if pending:
            log.info("Window %d: %d lines left unlabelled; using rule-based speakers", w, len(pending))
        if on_progress:
            on_progress(w, len(windows))

    for segment in dialogue:
        if segment.id not in result.lines:
            speaker = given.get(segment.id) or fallback.get(segment.id, UNKNOWN)
            result.lines[segment.id] = Performance(speaker=speaker)
    # A paragraph-spanning quote keeps its speaker unless the model was sure it changed.
    for later, earlier in chapter.continues.items():
        if result.lines[later].speaker == UNKNOWN and earlier in result.lines:
            result.lines[later].speaker = result.lines[earlier].speaker
    return result
