"""Pull one JSON object out of whatever an LLM sends back: fences, <think> blocks, trailing commas,
or an object cut short by the token limit."""

from __future__ import annotations

import json
import re
from typing import Any

_THINK = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")


class JSONParseError(ValueError):
    pass


def _scan(text: str, start: int) -> tuple[int | None, list[str]]:
    """Index of the brace closing text[start], or None plus the still-open brackets."""
    stack: list[str] = []
    in_string = escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not stack or stack[-1] != ch:
                return None, stack
            stack.pop()
            if not stack:
                return i, []
    if in_string:
        stack.append('"')
    return None, stack


def _loads(chunk: str) -> Any:
    try:
        return json.loads(chunk)
    except json.JSONDecodeError:
        return json.loads(_TRAILING_COMMA.sub(r"\1", chunk))


def _first_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    while start != -1:
        end, still_open = _scan(text, start)
        if end is not None:
            try:
                value = _loads(text[start : end + 1])
                if isinstance(value, dict):
                    return value
            except json.JSONDecodeError:
                pass
        elif still_open:
            value = _close_truncated(text[start:])
            if value is not None:
                return value
        start = text.find("{", start + 1)
    return None


def _close_truncated(text: str) -> dict[str, Any] | None:
    """Output cut off by the token limit: keep every complete element, drop the dangling one.

    Tries cut points after the last complete `}` / `]` first, then before the last commas."""
    after_close = [i + 1 for i, ch in enumerate(text) if ch in "}]"]
    before_comma = [i for i, ch in enumerate(text) if ch == ","]
    for cuts in (after_close, before_comma):
        for cut in reversed(cuts):
            prefix = text[:cut]
            _, still_open = _scan(prefix, 0)
            if not still_open or still_open[-1] == '"':
                continue  # complete already (handled elsewhere) or cut inside a string
            try:
                value = _loads(prefix + "".join(reversed(still_open)))
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    return None


def parse_json_object(text: str) -> dict[str, Any]:
    if not text or not text.strip():
        raise JSONParseError("empty response")
    cleaned = _THINK.sub("", text).strip()
    if re.match(r"<(think|thinking|reasoning)>", cleaned, re.IGNORECASE):  # unterminated reasoning block
        cleaned = re.split(r"</(?:think|thinking|reasoning)>", cleaned, flags=re.IGNORECASE)[-1]
    candidates = [m.group(1) for m in _FENCE.finditer(cleaned)] + [cleaned]
    for candidate in candidates:
        value = _first_object(candidate)
        if value is not None:
            return value
    snippet = cleaned[:200].replace("\n", " ")
    raise JSONParseError(f"no JSON object found in: {snippet!r}")
