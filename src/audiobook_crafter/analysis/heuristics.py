"""Rule-based speaker attribution: the no-LLM fallback and the baseline for the benchmark.

Uses explicit speech tags ("said Clara", "she asked"), action beats ("Clara ran down the stairs."),
one-speaker-per-paragraph, multi-paragraph quotes and A/B alternation in conversations.
"""

from __future__ import annotations

import re

from ..models import (
    NARRATOR,
    UNKNOWN,
    UNKNOWN_FEMALE,
    UNKNOWN_MALE,
    Character,
    CharacterMap,
    Segment,
    normalize_name,
)
from ..text.quotes import SPEECH_VERB_ALTERNATION as _VERBS
from .segment import SegmentedChapter

_NAME = r"(?:[A-Z][\w'’]*\.?)(?:\s+[A-Z][\w'’-]*){0,2}"
_SUBJECT = rf"(?P<subject>{_NAME}|[Hh]e|[Ss]he|I|[Tt]hey|[Ww]e)"
_TAG_AFTER = re.compile(
    rf"^[\s,;:—–-]*(?:(?i:{_VERBS})\s+(?:the\s+)?(?P<name>{_NAME}|[Hh]e|[Ss]he)\b"
    rf"|{_SUBJECT}\s+(?:\w+ly\s+)?(?i:{_VERBS}))"
)
_TAG_BEFORE = re.compile(rf"{_SUBJECT}\s+(?:\w+ly\s+)?(?i:{_VERBS})(?:\s+\w+ly)?\s*[,:—–-]?\s*$")
_BEAT = re.compile(rf"^(?:The\s+)?(?P<subject>{_NAME})\s+[a-z]")

_MALE_TITLES = {"mr", "mister", "sir", "lord", "king", "prince", "father", "uncle", "brother", "duke", "count",
                "baron", "master", "monsieur", "herr", "señor", "don", "captain", "colonel", "general"}  # fmt: skip
_FEMALE_TITLES = {"mrs", "miss", "ms", "lady", "queen", "princess", "mother", "aunt", "sister", "madam", "madame",
                  "duchess", "countess", "mademoiselle", "frau", "señora", "doña", "dame", "mistress"}  # fmt: skip
_NOT_NAMES = {
    "the", "a", "an", "and", "but", "then", "so", "yes", "no", "oh", "well", "now", "here", "there", "what",
    "why", "how", "when", "where", "who", "it", "this", "that", "these", "those", "his", "her", "their", "my",
    "our", "your", "if", "as", "at", "in", "on", "after", "before", "still", "even", "just", "only", "once",
    "outside", "inside", "suddenly", "finally", "later", "everyone", "someone", "nobody", "something",
    "nothing", "all", "both", "each", "some", "one", "two", "for", "with", "without", "god", "chapter",
}  # fmt: skip


def gender_from_name(name: str) -> str:
    first = normalize_name(name).split(" ")[0].rstrip(".") if name else ""
    if first in _MALE_TITLES:
        return "male"
    if first in _FEMALE_TITLES:
        return "female"
    return "unknown"


def _plausible_name(name: str) -> bool:
    words = name.split()
    return bool(words) and words[0].lower().rstrip(".") not in _NOT_NAMES and len(name) <= 40


class HeuristicAttributor:
    def __init__(self, characters: CharacterMap, discover: bool = True):
        self.characters = characters
        self.discover = discover  # add newly seen names to the character map

    def resolve(self, subject: str, recent: list[str]) -> str | None:
        lowered = subject.lower()
        if lowered in ("he", "she"):
            gender = "male" if lowered == "he" else "female"
            for speaker in reversed(recent[:-1] or recent):
                character = self.characters.get(speaker)
                if character and character.gender == gender:
                    return speaker
            return UNKNOWN_MALE if gender == "male" else UNKNOWN_FEMALE
        if lowered == "i":
            return self.characters.pov_character or NARRATOR
        if lowered in ("they", "we"):
            return None
        words = subject.split()
        while words and words[0].lower().rstrip(".") in _NOT_NAMES:
            words.pop(0)  # "Then Clara" -> "Clara"
        subject = " ".join(words)
        if not _plausible_name(subject):
            return None
        character = self.characters.find(subject)
        if character is None and self.discover:
            character = self.characters.add(Character(id="", name=subject, gender=gender_from_name(subject)))
        return character.id if character else None

    def attribute(self, chapter: SegmentedChapter) -> dict[str, str]:
        """segment id -> speaker id for every dialogue segment."""
        speakers: dict[str, str] = {}
        recent: list[str] = []  # speakers of previous dialogue paragraphs, oldest first
        segments = chapter.segments
        by_block: dict[int, list[Segment]] = {}
        for segment in segments:
            by_block.setdefault(segment.block, []).append(segment)

        previous_block_had_dialogue = False
        for items in by_block.values():
            dialogue = [s for s in items if s.kind == "dialogue"]
            if not dialogue:
                previous_block_had_dialogue = False
                continue
            found: str | None = None
            for i, segment in enumerate(items):
                if segment.kind != "dialogue" or found:
                    continue
                after = items[i + 1].text if i + 1 < len(items) and items[i + 1].kind == "narration" else ""
                before = items[i - 1].text if i > 0 and items[i - 1].kind == "narration" else ""
                match = _TAG_AFTER.match(after) if after else None
                subject = (match.group("name") or match.group("subject")) if match else None
                if not subject and before:
                    tag = _TAG_BEFORE.search(before)
                    beat = _BEAT.match(before)
                    subject = tag.group("subject") if tag else (beat.group("subject") if beat else None)
                if subject:
                    found = self.resolve(subject, recent)
            first = dialogue[0]
            if not found and first.id in chapter.continues:
                found = speakers.get(chapter.continues[first.id])
            if not found and previous_block_had_dialogue and len(recent) >= 2 and recent[-1] != recent[-2]:
                found = recent[-2]  # A, B, A, B ...
            speaker = found or UNKNOWN
            for segment in dialogue:
                speakers[segment.id] = speaker
            recent.append(speaker)
            previous_block_had_dialogue = True
        return speakers


def detect_narration(paragraphs: list[str]) -> str:
    """first_person if "I" shows up a lot outside quotes."""
    outside = " ".join(re.sub(r"[“\"«„].*?[”\"»“]", " ", p) for p in paragraphs[:300])
    words = outside.split()
    if not words:
        return "unknown"
    first = sum(1 for w in words if w in ("I", "I’m", "I'm", "I’d", "I'd", "me", "my", "My"))
    return "first_person" if first / len(words) > 0.012 else "third_person"
