"""Ollama through its native /api/chat endpoint.

The OpenAI-compatible /v1 endpoint can't set num_ctx or keep_alive, and Ollama's default context
(4K on GPUs under 24 GB) silently truncates long prompts, so the native API is used here.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from ..config import LLMConfig
from .base import LLM, LLMError, TransientLLMError

log = logging.getLogger(__name__)


class OllamaLLM(LLM):
    provider = "ollama"

    def __init__(self, model: str, config: LLMConfig, base_url: str):
        super().__init__(model, config)
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=httpx.Timeout(config.timeout_s, connect=10))

    def chat(self, system: str, messages: list[dict[str, str]], schema: dict[str, Any] | None) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "stream": False,
            "keep_alive": "15m",
            "options": {
                "temperature": self.config.temperature,
                "num_ctx": self.config.context_tokens,
                "num_predict": self.config.max_output_tokens,
            },
        }
        if schema is not None:
            payload["format"] = schema
        if self.config.think is not None:
            payload["think"] = self.config.think
        try:
            response = self.client.post(f"{self.base_url}/api/chat", json=payload)
        except httpx.ConnectError as err:
            raise LLMError(
                f"Can't reach Ollama at {self.base_url}. Start it with `ollama serve` "
                f"and pull the model with `ollama pull {self.model}`."
            ) from err
        except httpx.TimeoutException as err:
            raise TransientLLMError(f"Ollama timed out after {self.config.timeout_s:.0f}s") from err
        if response.status_code == 404:
            raise LLMError(f"Ollama has no model '{self.model}'. Run `ollama pull {self.model}`.")
        if response.status_code >= 500:
            raise TransientLLMError(f"Ollama error {response.status_code}: {response.text[:300]}")
        if response.status_code >= 400:
            raise LLMError(f"Ollama error {response.status_code}: {response.text[:300]}")
        data = response.json()
        prompt_tokens = data.get("prompt_eval_count") or 0
        if prompt_tokens >= 0.97 * self.config.context_tokens:
            log.warning(
                "Prompt used %d of %d context tokens and may have been truncated; raise llm.context_tokens "
                "or lower llm.chunk_chars.",
                prompt_tokens,
                self.config.context_tokens,
            )
        return (data.get("message") or {}).get("content", "")

    def release(self) -> None:
        """Unload the model so the GPU is free for TTS."""
        try:
            self.client.post(f"{self.base_url}/api/generate", json={"model": self.model, "keep_alive": 0})
        except httpx.HTTPError:
            pass
        self.client.close()
