"""Rule-based speaker attribution: the no-LLM fallback and the baseline for the benchmark.

Uses explicit speech tags ("said Clara", "she asked", "replied his wife"), action beats ("Clara ran
down the stairs."), one speaker per paragraph, paragraph-spanning quotes, and who is "on stage" to
answer an untagged reply.
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
_ROLES = {
    "wife": "female", "lady": "female", "mother": "female", "daughter": "female", "sister": "female",
    "aunt": "female", "mistress": "female", "girl": "female", "woman": "female",
    "husband": "male", "lord": "male", "father": "male", "son": "male", "brother": "male",
    "uncle": "male", "master": "male", "boy": "male", "man": "male",
}  # fmt: skip
_ROLE = rf"(?:his|her|their|the)\s+(?:{'|'.join(_ROLES)})"
_PERSON = rf"{_ROLE}|{_NAME}|[Hh]e|[Ss]he|I|[Tt]hey|[Ww]e"
_TAG_AFTER = re.compile(
    rf"^[\s,;:—–-]*(?:(?i:{_VERBS})\s+(?P<name>{_ROLE}|(?:the\s+)?{_NAME}|[Hh]e|[Ss]he)\b"
    rf"|(?P<subject>{_PERSON})\s+(?:\w+ly\s+)?(?i:{_VERBS}))"
)
_TAG_BEFORE = re.compile(rf"(?P<subject>{_PERSON})\s+(?:\w+ly\s+)?(?i:{_VERBS})(?:\s+\w+ly)?\s*[,:—–-]?\s*$")
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


def _strip_leading_words(name: str) -> str:
    words = name.split()
    while words and words[0].lower().rstrip(".") in _NOT_NAMES:
        words.pop(0)  # "Then Clara" -> "Clara"
    return " ".join(words)


class HeuristicAttributor:
    """Tracks who is "on stage" (recent speakers and people named in the narration) to resolve
    pronouns, role nouns ("his wife") and untagged replies."""

    def __init__(self, characters: CharacterMap, discover: bool = True):
        self.characters = characters
        self.discover = discover  # add newly seen names to the character map
        self.participants: list[str] = []  # most recent last

    def _gender_of(self, speaker: str) -> str:
        if speaker == UNKNOWN_MALE:
            return "male"
        if speaker == UNKNOWN_FEMALE:
            return "female"
        character = self.characters.get(speaker)
        return character.gender if character else "unknown"

    def _by_gender(self, gender: str, last: str | None) -> str:
        candidates = [p for p in reversed(self.participants) if self._gender_of(p) == gender]
        preferred = [p for p in candidates if p != last]
        if preferred or candidates:
            return (preferred or candidates)[0]
        return UNKNOWN_MALE if gender == "male" else UNKNOWN_FEMALE

    def _note(self, speaker: str) -> None:
        if speaker in self.participants:
            self.participants.remove(speaker)
        self.participants.append(speaker)

    def mentions(self, text: str) -> list[str]:
        """Known characters, or titled names ("Mr. Bennet"), named in a stretch of narration."""
        found = []
        for match in re.finditer(_NAME, text):
            name = _strip_leading_words(match.group(0))
            if not name:
                continue
            character = self.characters.find(name)
            if character is None and self.discover and gender_from_name(name) != "unknown" and " " in name:
                character = self.characters.add(Character(id="", name=name, gender=gender_from_name(name)))
            if character:
                found.append(character.id)
        return found

    def resolve(self, subject: str, last: str | None) -> str | None:
        lowered = subject.lower()
        if lowered in ("he", "she"):
            return self._by_gender("male" if lowered == "he" else "female", last)
        role = re.fullmatch(_ROLE, lowered)
        if role:
            return self._by_gender(_ROLES[lowered.split()[-1]], last)
        if lowered == "i":
            return self.characters.pov_character or NARRATOR
        if lowered in ("they", "we"):
            return None
        subject = _strip_leading_words(re.sub(r"^the\s+", "", subject, flags=re.IGNORECASE))
        if not subject or len(subject) > 40:
            return None
        character = self.characters.find(subject)
        if character is None and self.discover:
            character = self.characters.add(Character(id="", name=subject, gender=gender_from_name(subject)))
        return character.id if character else None

    def attribute(self, chapter: SegmentedChapter) -> dict[str, str]:
        """segment id -> speaker id for every dialogue segment."""
        speakers: dict[str, str] = {}
        by_block: dict[int, list[Segment]] = {}
        for segment in chapter.segments:
            by_block.setdefault(segment.block, []).append(segment)

        last: str | None = None  # speaker of the most recent dialogue paragraph
        since_dialogue = 99  # paragraphs since the last line of dialogue
        for items in by_block.values():
            if any(s.kind in ("heading", "pause") for s in items):
                last, since_dialogue = None, 99  # new scene: nobody is mid-conversation
            dialogue = [s for s in items if s.kind == "dialogue"]
            if not dialogue:
                for segment in items:
                    if segment.kind == "narration":
                        for mention in self.mentions(segment.text):
                            self._note(mention)
                since_dialogue += 1
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
                    found = self.resolve(subject, last)
            first = dialogue[0]
            if not found and first.id in chapter.continues:
                found = speakers.get(chapter.continues[first.id])
            if not found and last is not None and since_dialogue <= 2:
                # An untagged reply: whoever else is on stage answers.
                found = next((p for p in reversed(self.participants) if p != last), None)
            speaker = found or UNKNOWN
            for segment in dialogue:
                speakers[segment.id] = speaker
            for segment in items:
                if segment.kind == "narration":
                    for mention in self.mentions(segment.text):
                        if mention != speaker:
                            self._note(mention)
            if speaker != UNKNOWN:
                self._note(speaker)
            last, since_dialogue = speaker, 0
        return speakers


def detect_narration(paragraphs: list[str]) -> str:
    """first_person if "I" shows up a lot outside quotes."""
    outside = " ".join(re.sub(r"[“\"«„].*?[”\"»“]", " ", p) for p in paragraphs[:300])
    words = outside.split()
    if not words:
        return "unknown"
    first = sum(1 for w in words if w in ("I", "I’m", "I'm", "I’d", "I'd", "me", "my", "My"))
    return "first_person" if first / len(words) > 0.012 else "third_person"
