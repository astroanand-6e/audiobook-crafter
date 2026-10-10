"""Prompts and JSON schemas for the LLM brain. Every field is required and objects are closed
(additionalProperties: false) so the same schema works for Ollama, OpenAI-style strict mode and Claude."""

from __future__ import annotations

from typing import Any

from ..models import AGES, EMOTIONS, GENDERS, NONVERBAL_TAGS

CHARACTERS_SYSTEM = """\
You are the casting director for a full-cast audiobook. You read a book one passage at a time and \
keep the list of characters who SPEAK, so each one gets a consistent voice for the whole book.

Rules:
- Include every character who speaks at least one line of quoted dialogue in this passage. Also include \
a known character if this passage reveals something new about them. Skip people who are only mentioned.
- A known character keeps their exact id. Never create a second entry for someone already known under \
another name, nickname, title or role ("Lizzy" = "Elizabeth Bennet"). For a new character set id to "".
- name: the fullest name the text gives ("Elizabeth Bennet"), or a role if unnamed ("the innkeeper").
- aliases: other names, nicknames and titles the text uses for them.
- gender: male | female | neutral | unknown, and age: child | teen | young_adult | adult | middle_aged | \
elderly | unknown. Use pronouns, titles and descriptions; answer unknown when there is no evidence.
- description: one sentence on who they are (role, relationships).
- voice: one sentence on how they should sound in an audiobook: pitch, timbre, pace, accent, attitude. \
Ground it in the text (age, class, region, temperament). Never compare to real people.
- narration: first_person if the narrator is a character saying "I", otherwise third_person. narrator: \
that character's name or id when first_person, else "".
Reply with JSON only."""

ATTRIBUTION_SYSTEM = """\
You are the dialogue editor for a full-cast audiobook. In the passage, every quoted line is wrapped in \
<q id="N">…</q>. For EVERY id, decide who speaks it and how it should be performed.

Finding the speaker:
- Explicit tags first ("said Jane", "he asked"); resolve pronouns from context.
- Without tags, a two-person exchange alternates speakers, and a new paragraph usually means a new speaker.
- An action beat in the same paragraph usually belongs to the speaker ("Tom slammed the door. <q>…</q>").
- A line that addresses someone by name ("Jane, come here") is usually NOT spoken by that person.
- A quote that continues a paragraph-spanning speech keeps the same speaker.
- Quoted text that nobody says aloud (a sign, a letter, a title, a word being discussed) -> "narrator".
- If the book is narrated in first person, the narrator's own spoken lines belong to their character id.
- A line that already has speaker="…" keeps that speaker; only add the performance.
- Use only the allowed speaker ids. If you cannot tell, use unknown_male, unknown_female or unknown.

Performance:
- emotion: one of {emotions}.
- intensity: 0.0-1.0. Most lines sit between 0.2 and 0.5; keep 0.8+ for screaming, sobbing or terror.
- direction: a few words of delivery ("through clenched teeth", "teasing") or "".
- tags: nonverbal sounds the text explicitly describes for that line, from {tags}; usually [].
- narration: list only the [P#] paragraphs of narration whose mood clearly departs from neutral \
(a tense chase, a grieving scene), with an emotion and intensity. Usually few or none.
Reply with JSON only."""


def characters_schema() -> dict[str, Any]:
    character = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "name": {"type": "string"},
            "aliases": {"type": "array", "items": {"type": "string"}},
            "gender": {"type": "string", "enum": list(GENDERS)},
            "age": {"type": "string", "enum": list(AGES)},
            "description": {"type": "string"},
            "voice": {"type": "string"},
        },
        "required": ["id", "name", "aliases", "gender", "age", "description", "voice"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "narration": {"type": "string", "enum": ["first_person", "third_person", "unknown"]},
            "narrator": {"type": "string"},
            "characters": {"type": "array", "items": character},
        },
        "required": ["narration", "narrator", "characters"],
        "additionalProperties": False,
    }


def attribution_schema(speaker_ids: list[str]) -> dict[str, Any]:
    line = {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "speaker": {"type": "string", "enum": speaker_ids},
            "emotion": {"type": "string", "enum": list(EMOTIONS)},
            "intensity": {"type": "number"},
            "direction": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string", "enum": list(NONVERBAL_TAGS)}},
        },
        "required": ["id", "speaker", "emotion", "intensity", "direction", "tags"],
        "additionalProperties": False,
    }
    mood = {
        "type": "object",
        "properties": {
            "p": {"type": "integer"},
            "emotion": {"type": "string", "enum": list(EMOTIONS)},
            "intensity": {"type": "number"},
        },
        "required": ["p", "emotion", "intensity"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "lines": {"type": "array", "items": line},
            "narration": {"type": "array", "items": mood},
        },
        "required": ["lines", "narration"],
        "additionalProperties": False,
    }


def attribution_system() -> str:
    return ATTRIBUTION_SYSTEM.format(emotions=", ".join(EMOTIONS), tags=", ".join(NONVERBAL_TAGS))
