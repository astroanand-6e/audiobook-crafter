"""Casting: give every speaking character a voice once, and keep it for the whole book.

Order of preference for a new character: your clip in voices/<id>.wav, an unused clip from
voices/pool/ of the same gender, then a voice designed from the character map's description.
Characters with only a line or two share a small pool of "extra" voices until they matter more."""

from __future__ import annotations

import zlib
from collections.abc import Mapping

from ..models import (
    NARRATOR,
    UNKNOWN,
    UNKNOWN_FEMALE,
    UNKNOWN_MALE,
    Cast,
    Character,
    CharacterMap,
    VoiceProfile,
)
from ..project import Project
from .voices import pool_clips, transcript_for, user_clips

EXTRAS: dict[str, list[tuple[str, str]]] = {
    "male": [
        ("extra_male_1", "An ordinary adult man with a mid-pitched, plain, natural voice."),
        ("extra_male_2", "An older man with a slightly rough, lower voice and a slower pace."),
        ("extra_male_3", "A young man with a light, clear voice."),
    ],
    "female": [
        ("extra_female_1", "An ordinary adult woman with a mid-pitched, plain, natural voice."),
        ("extra_female_2", "An older woman with a warm, slightly husky voice."),
        ("extra_female_3", "A young woman with a bright, light voice."),
    ],
    "unknown": [("extra_neutral", "An adult with a neutral, mid-range voice and plain delivery.")],
}
_STUDIO = " Neutral, conversational delivery; close-mic studio recording with no background noise."


def describe_voice(character: Character) -> str:
    """A voice-design brief from the character map: who they are plus how they should sound."""
    nouns = {"male": "man", "female": "woman"}
    if character.age == "child":
        who = {"male": "A young boy", "female": "A young girl"}.get(character.gender, "A young child")
    elif character.age == "teen":
        who = {"male": "A teenage boy", "female": "A teenage girl"}.get(character.gender, "A teenager")
    else:
        adjective = {
            "young_adult": "young",
            "adult": "adult",
            "middle_aged": "middle-aged",
            "elderly": "elderly",
        }.get(character.age, "")
        noun = nouns.get(character.gender, "person")
        phrase = f"{adjective} {noun}".strip()
        who = f"{'An' if phrase[0] in 'aeiou' else 'A'} {phrase}"
    detail = character.voice or character.description
    return f"{who}. {detail}".strip().rstrip(".") + "." + _STUDIO


def _extra_for(character_id: str, gender: str) -> str:
    pool = EXTRAS.get(gender, EXTRAS["unknown"])
    return pool[zlib.crc32(character_id.encode()) % len(pool)][0]


def auto_cast(
    project: Project, characters: CharacterMap, cast: Cast, speaking: Mapping[str, int] | None = None
) -> list[str]:
    """Fill in voices for anyone who doesn't have one yet. Returns human-readable changes.

    `speaking`: lines per speaker id in the scripts, so unattributed speakers only get a voice if used."""
    config = project.config.casting
    voices = cast.voices
    clips = user_clips(project)
    changes: list[str] = []

    def clip_profile(path) -> VoiceProfile:
        return VoiceProfile(reference=project.relative(path), reference_text=transcript_for(path))

    def adopt_clip(voice_id: str) -> bool:
        """A clip you dropped at voices/<id>.wav replaces a voice that was chosen automatically."""
        current = voices.get(voice_id)
        clip = clips.get(voice_id)
        if clip is None or (current and current.reference == project.relative(clip)):
            return False
        if current and not (current.design or (current.same_as or "").startswith("extra_")):
            return False  # you set a preset, a clip or a link by hand: keep it
        keep = {"engine": current.engine, "params": current.params, "seed": current.seed} if current else {}
        voices[voice_id] = clip_profile(clip).model_copy(update=keep)
        changes.append(f"{voice_id}: your clip {project.relative(clip)}")
        return True

    if not adopt_clip(NARRATOR) and NARRATOR not in voices:
        voices[NARRATOR] = VoiceProfile(design=config.narrator)
        changes.append(f"{NARRATOR}: designed narrator voice")

    taken = {p.reference for p in voices.values() if p.reference}
    pool = [(g, p) for g, p in pool_clips(project) if project.relative(p) not in taken]

    for character in sorted(characters.characters, key=lambda c: (-c.lines, c.id)):
        if character.lines and adopt_clip(character.id):
            continue
        current = voices.get(character.id)
        shares_extra = bool(current and current.same_as and current.same_as.startswith("extra_"))
        promote = shares_extra and character.lines >= config.min_lines_for_own_voice
        if (current and not promote) or character.lines == 0:
            continue
        if characters.pov_character == character.id and config.pov_uses_narrator_voice:
            voices[character.id] = VoiceProfile(same_as=NARRATOR)
            changes.append(f"{character.id}: the narrator's voice (first-person narrator)")
        elif character.lines < config.min_lines_for_own_voice:
            extra = _extra_for(character.id, character.gender)
            voices[character.id] = VoiceProfile(same_as=extra)
            changes.append(f"{character.id}: shares {extra} ({character.lines} line)")
        elif (index := next((i for i, (g, _) in enumerate(pool) if g == character.gender), None)) is not None:
            _, path = pool.pop(index)
            voices[character.id] = clip_profile(path)
            changes.append(f"{character.id}: pool clip {project.relative(path)}")
        else:
            voices[character.id] = VoiceProfile(design=describe_voice(character))
            changes.append(f"{character.id}: {'promoted to own ' if promote else ''}designed voice")

    for speaker, gender in ((UNKNOWN_MALE, "male"), (UNKNOWN_FEMALE, "female"), (UNKNOWN, "unknown")):
        if speaker not in voices and (speaking is None or speaking.get(speaker)):
            voices[speaker] = VoiceProfile(same_as=EXTRAS[gender][0][0])

    wanted = {p.same_as for p in voices.values() if p.same_as}
    for pool_voices in EXTRAS.values():
        for voice_id, brief in pool_voices:
            if voice_id in wanted and voice_id not in voices:
                voices[voice_id] = VoiceProfile(design=brief + _STUDIO)
    return changes
