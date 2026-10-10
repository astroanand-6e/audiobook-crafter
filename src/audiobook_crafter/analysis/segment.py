"""Deterministic first pass: chapter blocks -> ordered narration / dialogue / heading / pause segments."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models import NARRATOR, Chapter, Segment
from ..text.quotes import QuoteStyle, split_dialogue


@dataclass
class SegmentedChapter:
    segments: list[Segment]
    # dialogue segment id -> id of the open quote it continues from the previous paragraph
    continues: dict[str, str] = field(default_factory=dict)
    # segment id -> speaker name given by the source (scripts / transcripts)
    given_speakers: dict[str, str] = field(default_factory=dict)


def segment_chapter(chapter: Chapter, style: QuoteStyle, announce_title: bool = True) -> SegmentedChapter:
    result = SegmentedChapter(segments=[])
    segments = result.segments

    def add(**fields) -> Segment:
        segment = Segment(id=f"c{chapter.index:03d}-s{len(segments):04d}", **fields)
        segments.append(segment)
        return segment

    if announce_title and chapter.title and not any(b.kind == "heading" for b in chapter.blocks[:3]):
        add(block=-1, kind="heading", text=chapter.title)

    open_quote: str | None = None  # dialogue left unclosed at the end of the previous paragraph
    for index, block in enumerate(chapter.blocks):
        if block.kind == "heading":
            add(block=index, kind="heading", text=block.text)
            open_quote = None
        elif block.kind == "scene_break":
            add(block=index, kind="pause")
            open_quote = None
        elif block.speaker:
            kind = "narration" if block.speaker == NARRATOR else "dialogue"
            segment = add(block=index, kind=kind, text=block.text, direction=block.direction or "")
            if kind == "dialogue":
                result.given_speakers[segment.id] = block.speaker
        else:
            spans = split_dialogue(block.text, style)
            previous_open, open_quote = open_quote, None
            for position, span in enumerate(spans):
                segment = add(block=index, kind=span.kind, text=span.text)
                if span.kind == "dialogue" and position == 0 and previous_open:
                    result.continues[segment.id] = previous_open
                if span.kind == "dialogue" and span.open_end:
                    open_quote = segment.id
    return result
