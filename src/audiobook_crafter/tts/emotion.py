"""Translate a line's emotion into each engine's controls: an intensity knob (Chatterbox), a natural
language instruction (Qwen3 CustomVoice / VoiceDesign), or inline sound tags (Chatterbox Turbo)."""

from __future__ import annotations

# How much an emotion pushes energy up (+) or down (-).
AROUSAL = {
    "neutral": 0.0, "calm": -0.5, "happy": 0.4, "excited": 0.9, "amused": 0.4, "tender": -0.4,
    "sad": -0.5, "angry": 0.9, "fearful": 0.7, "anxious": 0.5, "surprised": 0.7, "disgusted": 0.5,
    "sarcastic": 0.2, "serious": 0.0, "whispering": -0.8, "shouting": 1.0,
}  # fmt: skip

_PHRASES = {
    "neutral": "",
    "calm": "calm and even",
    "happy": "happy, warm",
    "excited": "excited and energetic",
    "amused": "amused, with a hint of a smile",
    "tender": "tender and gentle",
    "sad": "sad, subdued",
    "angry": "angry",
    "fearful": "frightened",
    "anxious": "anxious and tense",
    "surprised": "surprised",
    "disgusted": "disgusted",
    "sarcastic": "dry and sarcastic",
    "serious": "serious and measured",
    "whispering": "whispering, hushed",
    "shouting": "shouting, loud",
}

TURBO_TAGS = {"clear_throat": "clear throat"}  # Chatterbox Turbo spells this one with a space


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def chatterbox_controls(
    emotion: str,
    intensity: float,
    base_exaggeration: float = 0.5,
    base_cfg: float = 0.5,
    strength: float = 1.0,
) -> tuple[float, float]:
    """(exaggeration, cfg_weight). Per Resemble's tips, more exaggeration speeds speech up, so a
    lower cfg_weight compensates with slower, more deliberate pacing."""
    push = AROUSAL.get(emotion, 0.0) * intensity * strength
    if push >= 0:
        exaggeration = base_exaggeration + 0.45 * push
        cfg = base_cfg - 0.22 * push
    else:
        exaggeration = base_exaggeration + 0.2 * push
        cfg = base_cfg
    return round(_clamp(exaggeration, 0.25, 1.2), 3), round(_clamp(cfg, 0.2, 1.0), 3)


def style_phrase(emotion: str, intensity: float, direction: str = "", strength: float = 1.0) -> str:
    """ "very angry, through clenched teeth" or "" for a plain read."""
    phrase = _PHRASES.get(emotion, "")
    level = intensity * strength
    parts = []
    if phrase:
        degree = (
            "slightly " if level < 0.3 else "" if level < 0.6 else "very " if level < 0.85 else "extremely "
        )
        parts.append(f"{degree}{phrase}")
    if direction:
        parts.append(direction)
    return ", ".join(parts)


def instruction(emotion: str, intensity: float, direction: str = "", strength: float = 1.0) -> str:
    """ "Speak very angry, through clenched teeth." or "" for a plain read."""
    phrase = style_phrase(emotion, intensity, direction, strength)
    return f"Speak {phrase}." if phrase else ""


def turbo_prefix(tags: list[str]) -> str:
    return "".join(f"[{TURBO_TAGS.get(t, t)}] " for t in tags)
