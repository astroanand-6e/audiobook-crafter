from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from audiobook_crafter.analysis.prompts import CHARACTERS_SYSTEM
from audiobook_crafter.config import LLMConfig
from audiobook_crafter.llm.base import LLM

Handler = Callable[[str, str, dict[str, Any] | None], Any]


class FakeLLM(LLM):
    """Answers from a Python function instead of a model; records every call."""

    provider = "fake"

    def __init__(self, handler: Handler):
        super().__init__("fake", LLMConfig())
        self.handler = handler
        self.calls: list[tuple[str, str]] = []

    def chat(self, system: str, messages: list[dict[str, str]], schema: dict[str, Any] | None) -> str:
        prompt = messages[-1]["content"]
        self.calls.append((system, prompt))
        reply = self.handler(system, prompt, schema)
        return reply if isinstance(reply, str) else json.dumps(reply)


GREEN_SEAL_CAST = [
    {"id": "", "name": "Clara Hartwell", "aliases": ["Clara", "Miss Hartwell"], "gender": "female",
     "age": "young_adult", "description": "The curious daughter.", "voice": "Bright, quick, warm."},
    {"id": "", "name": "Mr. Hartwell", "aliases": ["Father"], "gender": "male", "age": "middle_aged",
     "description": "Clara's father.", "voice": "Low, dry, amused."},
    {"id": "", "name": "Mrs. Pell", "aliases": [], "gender": "female", "age": "adult",
     "description": "A stern woman in black.", "voice": "Clipped and cool."},
]  # fmt: skip

LINE_SPEAKERS = {
    "Is it from London?": "clara_hartwell",
    "It is,": "mr_hartwell",
    "And it is addressed to you.": "mr_hartwell",
    "Then give it to me": "clara_hartwell",
    "Miss Hartwell?": "mrs_pell",
    "I am Mrs. Pell": "mrs_pell",
    "Where are we going?": "clara_hartwell",
    "Somewhere dry,": "mrs_pell",
}


def green_seal_handler(system: str, prompt: str, schema: dict[str, Any] | None) -> dict[str, Any]:
    if system == CHARACTERS_SYSTEM:
        return {"narration": "third_person", "narrator": "", "characters": GREEN_SEAL_CAST}
    lines = []
    for number, text in re.findall(r'<q id="(\d+)">(.*?)</q>', prompt):
        speaker = next((s for key, s in LINE_SPEAKERS.items() if key in text), "unknown")
        emotion = "excited" if "!" in text else "neutral"
        lines.append({"id": int(number), "speaker": speaker, "emotion": emotion, "intensity": 0.6,
                      "direction": "", "tags": []})  # fmt: skip
    return {"lines": lines, "narration": []}
