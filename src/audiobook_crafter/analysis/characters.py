"""Build and grow the character map passage by passage, so any context size works (a 9B local model
or a 1M-token API model) and ids never change once assigned."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from typing import Any

from ..llm import LLM
from ..models import (
    RESERVED_SPEAKERS,
    Character,
    CharacterMap,
    coerce_age,
    coerce_gender,
    normalize_name,
    slugify,
)
from .prompts import CHARACTERS_SYSTEM, characters_schema

log = logging.getLogger(__name__)


def chunk_passages(paragraphs: Iterable[str], max_chars: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for paragraph in paragraphs:
        if current and size + len(paragraph) > max_chars:
            chunks.append("\n\n".join(current))
            current, size = [], 0
        current.append(paragraph)
        size += len(paragraph) + 2
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _relevant(characters: CharacterMap, passage: str, limit: int = 120) -> list[Character]:
    """Characters named in the passage first, then the busiest ones, so big casts don't flood the prompt."""
    text = normalize_name(passage)
    named = [
        c for c in characters.characters if any(normalize_name(n) in text for n in c.names() if len(n) > 2)
    ]
    rest = sorted((c for c in characters.characters if c not in named), key=lambda c: -c.lines)
    return (named + rest)[:limit]


def roster(characters: Iterable[Character]) -> str:
    rows = [
        f"- {c.id} | {c.name} | aliases: {', '.join(c.aliases) or '-'} | {c.gender}, {c.age} | {c.description[:120]}"
        for c in characters
    ]
    return "\n".join(rows) or "(none yet)"


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def merge_character(characters: CharacterMap, item: dict[str, Any]) -> Character | None:
    name = _clean(item.get("name"))
    if not name or normalize_name(name) in RESERVED_SPEAKERS or slugify(name) in RESERVED_SPEAKERS:
        return None
    aliases = [_clean(a) for a in item.get("aliases") or [] if _clean(a)]
    existing = characters.get(_clean(item.get("id"))) or characters.find(name)
    if existing is None:
        # An alias may only match another character's *name*: "Bennet" must not merge Mr. into Mrs. Bennet.
        keys = {normalize_name(a) for a in aliases}
        existing = next((c for c in characters.characters if normalize_name(c.name) in keys), None)

    if existing is None:
        character = Character(
            id=slugify(name),
            name=name,
            aliases=aliases,
            gender=item.get("gender") or "unknown",
            age=item.get("age") or "unknown",
            description=_clean(item.get("description")),
            voice=_clean(item.get("voice")),
        )
        return characters.add(character)

    known = {normalize_name(n) for n in existing.names()}
    other_names = {normalize_name(c.name) for c in characters.characters if c is not existing}
    for alias in [name, *aliases]:
        key = normalize_name(alias)
        if key not in known and key not in other_names:
            existing.aliases.append(alias)
            known.add(key)
    if existing.gender == "unknown":
        existing.gender = coerce_gender(item.get("gender"))
    if existing.age == "unknown":
        existing.age = coerce_age(item.get("age"))
    existing.description = existing.description or _clean(item.get("description"))
    existing.voice = existing.voice or _clean(item.get("voice"))
    return existing


def apply_character_update(characters: CharacterMap, data: dict[str, Any]) -> None:
    narration = data.get("narration")
    if characters.narration == "unknown" and narration in ("first_person", "third_person"):
        characters.narration = narration
    for item in data.get("characters") or []:
        if isinstance(item, dict):
            merge_character(characters, item)
    narrator = _clean(data.get("narrator"))
    if characters.narration == "first_person" and narrator and not characters.pov_character:
        found = characters.get(narrator) or characters.find(narrator)
        if found:
            characters.pov_character = found.id


def update_character_map(
    llm: LLM,
    characters: CharacterMap,
    passages: list[str],
    *,
    title: str,
    author: str,
    language: str,
    on_progress: Callable[[int, int], None] | None = None,
) -> CharacterMap:
    for n, passage in enumerate(passages, start=1):
        prompt = (
            f'Book: "{title}"{f" by {author}" if author else ""} (language: {language}).\n\n'
            "Known characters (id | name | aliases | gender, age | description):\n"
            f"{roster(_relevant(characters, passage))}\n\n"
            f"Passage:\n<<<\n{passage}\n>>>"
        )
        data = llm.complete_json(CHARACTERS_SYSTEM, prompt, characters_schema())
        apply_character_update(characters, data)
        if on_progress:
            on_progress(n, len(passages))
    return characters
