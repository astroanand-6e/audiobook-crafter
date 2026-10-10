from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, TypeVar

from ..config import LLMConfig
from .jsonutil import JSONParseError, parse_json_object

log = logging.getLogger(__name__)
T = TypeVar("T")


class LLMError(RuntimeError):
    pass


class TransientLLMError(LLMError):
    """Rate limits, timeouts, 5xx: worth retrying."""


def with_retries(call: Callable[[], T], attempts: int = 4, base_delay: float = 2.0) -> T:
    for attempt in range(attempts):
        try:
            return call()
        except TransientLLMError as err:
            if attempt == attempts - 1:
                raise
            delay = base_delay * 2**attempt
            log.warning("LLM call failed (%s); retrying in %.0fs", err, delay)
            time.sleep(delay)
    raise AssertionError("unreachable")


class LLM(ABC):
    """A chat model that answers with JSON. Providers implement `chat`."""

    provider: str = ""

    def __init__(self, model: str, config: LLMConfig):
        self.model = model
        self.config = config

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"

    @abstractmethod
    def chat(self, system: str, messages: list[dict[str, str]], schema: dict[str, Any] | None) -> str:
        """Return the assistant's text for one request."""

    def complete_json(self, system: str, prompt: str, schema: dict[str, Any] | None = None) -> dict[str, Any]:
        messages = [{"role": "user", "content": prompt}]
        last_error: Exception | None = None
        for _ in range(2):
            text = with_retries(lambda msgs=messages: self.chat(system, msgs, schema))
            try:
                return parse_json_object(text)
            except JSONParseError as err:
                last_error = err
                log.warning("%s returned invalid JSON (%s); asking again", self.name, err)
                messages = [
                    *messages,
                    {"role": "assistant", "content": text[:6000]},
                    {"role": "user", "content": "That was not valid JSON. Reply with only the JSON object."},
                ]
        raise LLMError(f"{self.name} did not return valid JSON: {last_error}")

    def release(self) -> None:  # noqa: B027 - optional hook
        """Free resources (e.g. unload a local model so the GPU is free for TTS)."""
