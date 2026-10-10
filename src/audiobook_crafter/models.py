"""Core data model shared by every pipeline stage.

Book -> Chapter -> Block        what ingest produces from a PDF / EPUB / text / script
CharacterMap -> Character       what the LLM brain builds (who speaks, how they should sound)
ChapterScript -> Segment        the "screenplay": every line, its speaker and its delivery
Cast -> VoiceProfile            which voice performs each character, for the whole book
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

NARRATOR = "narrator"
UNKNOWN = "unknown"
UNKNOWN_MALE = "unknown_male"
UNKNOWN_FEMALE = "unknown_female"
RESERVED_SPEAKERS = (NARRATOR, UNKNOWN, UNKNOWN_MALE, UNKNOWN_FEMALE)

EMOTIONS = (
    "neutral",
    "calm",
    "happy",
    "excited",
    "amused",
    "tender",
    "sad",
    "angry",
    "fearful",
    "anxious",
    "surprised",
    "disgusted",
    "sarcastic",
    "serious",
    "whispering",
    "shouting",
)
NONVERBAL_TAGS = ("laugh", "chuckle", "sigh", "gasp", "cough", "groan", "sniff", "clear_throat", "shush")

Gender = Literal["male", "female", "neutral", "unknown"]
Age = Literal["child", "teen", "young_adult", "adult", "middle_aged", "elderly", "unknown"]
GENDERS: tuple[str, ...] = ("male", "female", "neutral", "unknown")
AGES: tuple[str, ...] = ("child", "teen", "young_adult", "adult", "middle_aged", "elderly", "unknown")


def slugify(name: str) -> str:
    """Stable, filesystem-safe id for a character name ("Mrs. Bennet" -> "mrs_bennet")."""
    ascii_text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", ascii_text).strip("_").lower()
    if slug:
        return slug
    # Non-Latin names (e.g. Devanagari) keep their letters and combining marks.
    kept = "".join(ch if unicodedata.category(ch)[0] in "LMN" else "_" for ch in name)
    return re.sub(r"_+", "_", kept).strip("_").lower() or "character"


def normalize_name(name: str) -> str:
    """Comparison key for character names: case, punctuation and a leading "the" don't matter."""
    text = unicodedata.normalize("NFKC", name).lower().replace("’", "'")
    text = re.sub(r"[^\w\s'-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"^the ", "", text)


def coerce_gender(value: Any) -> str:
    value = str(value or "unknown").strip().lower()
    aliases = {"m": "male", "f": "female", "man": "male", "woman": "female", "boy": "male", "girl": "female"}
    value = aliases.get(value, value)
    return value if value in GENDERS else "unknown"


def coerce_age(value: Any) -> str:
    value = str(value or "unknown").strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {"young": "young_adult", "old": "elderly", "middle": "middle_aged", "kid": "child"}
    value = aliases.get(value, value)
    return value if value in AGES else "unknown"


# --------------------------------------------------------------------------- book


class Block(BaseModel):
    kind: Literal["paragraph", "heading", "scene_break"] = "paragraph"
    text: str = ""
    # Set when the source already says who speaks (scripts / transcripts); skips LLM attribution.
    speaker: str | None = None
    direction: str | None = None


class Chapter(BaseModel):
    index: int
    title: str
    kind: Literal["chapter", "front_matter", "back_matter"] = "chapter"
    pages: tuple[int, int] | None = None
    blocks: list[Block] = Field(default_factory=list)

    @property
    def word_count(self) -> int:
        return sum(len(b.text.split()) for b in self.blocks if b.kind == "paragraph")

    def preview(self, words: int = 14) -> str:
        for block in self.blocks:
            if block.kind == "paragraph" and block.text.strip():
                tokens = block.text.split()
                return " ".join(tokens[:words]) + (" …" if len(tokens) > words else "")
        return ""


class Book(BaseModel):
    title: str
    author: str = ""
    language: str = "en"
    source: str = ""
    chapters: list[Chapter] = Field(default_factory=list)

    def chapter(self, index: int) -> Chapter:
        for chapter in self.chapters:
            if chapter.index == index:
                return chapter
        known = ", ".join(str(c.index) for c in self.chapters) or "none"
        raise KeyError(f"No chapter {index} in this book (chapters: {known})")


# --------------------------------------------------------------------------- characters


class Character(BaseModel):
    id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    gender: Gender = "unknown"
    age: Age = "unknown"
    description: str = ""
    voice: str = ""  # how they should sound; drives casting and voice design
    lines: int = 0  # attributed lines of dialogue so far

    @field_validator("gender", mode="before")
    @classmethod
    def _gender(cls, value: Any) -> str:
        return coerce_gender(value)

    @field_validator("age", mode="before")
    @classmethod
    def _age(cls, value: Any) -> str:
        return coerce_age(value)

    def names(self) -> list[str]:
        return [self.name, *self.aliases]


class CharacterMap(BaseModel):
    narration: Literal["first_person", "third_person", "unknown"] = "unknown"
    pov_character: str | None = None  # the "I" of a first-person book
    characters: list[Character] = Field(default_factory=list)

    def get(self, character_id: str) -> Character | None:
        return next((c for c in self.characters if c.id == character_id), None)

    def find(self, name: str) -> Character | None:
        """Exact match on id, name or alias (after normalisation). Never fuzzy: "Mr. Bennet" != "Mrs. Bennet"."""
        if not name:
            return None
        direct = self.get(name)
        if direct:
            return direct
        key = normalize_name(name)
        for character in self.characters:
            if key == character.id or any(normalize_name(n) == key for n in character.names()):
                return character
        return None

    def add(self, character: Character) -> Character:
        taken = {c.id for c in self.characters} | set(RESERVED_SPEAKERS)
        base = character.id or slugify(character.name)
        new_id, n = base, 2
        while new_id in taken:
            new_id, n = f"{base}_{n}", n + 1
        character.id = new_id
        self.characters.append(character)
        return character

    def ids(self) -> set[str]:
        return {c.id for c in self.characters}


# --------------------------------------------------------------------------- script


SegmentKind = Literal["narration", "dialogue", "heading", "pause"]


class Segment(BaseModel):
    model_config = ConfigDict(validate_assignment=True)  # edits made by later stages get coerced too

    id: str
    block: int  # paragraph index inside the chapter; drives pacing
    kind: SegmentKind
    text: str = ""
    speaker: str = NARRATOR
    emotion: str = "neutral"
    intensity: float = 0.3
    direction: str = ""  # free-text delivery note, e.g. "through clenched teeth"
    tags: list[str] = Field(default_factory=list)  # nonverbal sounds for engines that support them
    pause_ms: int | None = None  # silence after this segment (or the length of a "pause" segment)

    @field_validator("emotion", mode="before")
    @classmethod
    def _emotion(cls, value: Any) -> str:
        value = str(value or "neutral").strip().lower()
        return value if value in EMOTIONS else "neutral"

    @field_validator("intensity", mode="before")
    @classmethod
    def _intensity(cls, value: Any) -> float:
        try:
            return min(1.0, max(0.0, float(value)))
        except (TypeError, ValueError):
            return 0.3

    @field_validator("tags", mode="before")
    @classmethod
    def _tags(cls, value: Any) -> list[str]:
        if not value:
            return []
        if isinstance(value, str):
            value = [value]
        cleaned = [str(v).strip().lower().replace(" ", "_").strip("[]") for v in value]
        return [t for t in cleaned if t in NONVERBAL_TAGS]


class ChapterScript(BaseModel):
    chapter: int
    title: str
    segments: list[Segment] = Field(default_factory=list)

    def speaker_counts(self) -> Counter[str]:
        return Counter(s.speaker for s in self.segments if s.kind == "dialogue")


# --------------------------------------------------------------------------- casting


class VoiceProfile(BaseModel):
    """How one voice is produced. Usually exactly one of reference / preset / same_as is set."""

    engine: str | None = None  # None -> the project's default engine
    reference: str | None = None  # reference clip to clone (path relative to the project folder)
    reference_text: str | None = None  # transcript of the reference clip (improves Qwen3 cloning)
    preset: str | None = None  # an engine's built-in speaker, e.g. Qwen3 CustomVoice "Ryan"
    design: str | None = None  # description for a voice-design model to create the reference clip
    same_as: str | None = None  # share another voice (minor characters, crowds)
    params: dict[str, Any] = Field(default_factory=dict)  # engine knobs: exaggeration, temperature, ...
    seed: int | None = None


class Cast(BaseModel):
    voices: dict[str, VoiceProfile] = Field(default_factory=dict)

    def resolve(self, speaker: str) -> tuple[str, VoiceProfile]:
        """Follow `same_as` links; unknown speakers fall back to the narrator voice."""
        seen: list[str] = []
        current = speaker if speaker in self.voices else NARRATOR
        while True:
            if current in seen:
                raise ValueError(f"Voice loop in cast: {' -> '.join([*seen, current])}")
            seen.append(current)
            profile = self.voices.get(current)
            if profile is None:
                return current, VoiceProfile()
            if profile.same_as:
                current = profile.same_as
                continue
            return current, profile
