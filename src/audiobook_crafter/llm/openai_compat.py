"""Any OpenAI-compatible /chat/completions endpoint: OpenAI, Gemini, OpenRouter, Groq, DeepSeek,
LM Studio, llama.cpp server, vLLM. Parameters a server rejects are dropped and remembered."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from ..config import LLMConfig
from .base import LLM, LLMError, TransientLLMError

log = logging.getLogger(__name__)


class OpenAICompatLLM(LLM):
    def __init__(
        self,
        provider: str,
        model: str,
        config: LLMConfig,
        base_url: str,
        api_key: str | None,
        json_mode: str = "schema",
    ):
        super().__init__(model, config)
        self.provider = provider
        self.base_url = base_url.rstrip("/")
        self.json_mode = json_mode  # "schema" -> "object" -> "none" as servers refuse them
        self.use_temperature = True
        self.max_tokens_key = "max_tokens"
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx.Client(headers=headers, timeout=httpx.Timeout(config.timeout_s, connect=10))

    def _payload(self, system: str, messages: list[dict[str, str]], schema: dict[str, Any] | None) -> dict:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            self.max_tokens_key: self.config.max_output_tokens,
        }
        if self.use_temperature:
            payload["temperature"] = self.config.temperature
        if schema is not None and self.json_mode == "schema":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "result", "schema": schema, "strict": False},
            }
            if self.provider == "openrouter":
                payload["provider"] = {"require_parameters": True}
        elif schema is not None and self.json_mode == "object":
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _adapt(self, error_text: str) -> bool:
        """Drop whatever the server just complained about. Returns False if nothing is left to drop."""
        text = error_text.lower()
        if "response_format" in text or "json_schema" in text or "json mode" in text:
            if self.json_mode != "none":
                self.json_mode = {"schema": "object", "object": "none"}[self.json_mode]
                return True
        if "temperature" in text and self.use_temperature:
            self.use_temperature = False
            return True
        if "max_tokens" in text and self.max_tokens_key == "max_tokens":
            self.max_tokens_key = "max_completion_tokens"
            return True
        return False

    def chat(self, system: str, messages: list[dict[str, str]], schema: dict[str, Any] | None) -> str:
        for _ in range(6):  # up to four downgrades, then the request that works
            try:
                response = self.client.post(
                    f"{self.base_url}/chat/completions", json=self._payload(system, messages, schema)
                )
            except httpx.ConnectError as err:
                raise LLMError(f"Can't reach {self.provider} at {self.base_url}: {err}") from err
            except httpx.TimeoutException as err:
                raise TransientLLMError(f"{self.provider} timed out") from err
            if response.status_code == 400 and self._adapt(response.text):
                log.info("%s rejected a parameter; retrying without it", self.name)
                continue
            if response.status_code in (401, 403):
                raise LLMError(f"{self.provider} rejected the API key ({response.status_code}).")
            if response.status_code == 429 or response.status_code >= 500:
                raise TransientLLMError(
                    f"{self.provider} error {response.status_code}: {response.text[:200]}"
                )
            if response.status_code >= 400:
                raise LLMError(f"{self.provider} error {response.status_code}: {response.text[:500]}")
            choices = response.json().get("choices") or []
            content = (choices[0].get("message") or {}).get("content") if choices else None
            if not content:
                raise TransientLLMError(f"{self.provider} returned an empty message")
            return content
        raise LLMError(f"{self.provider} kept rejecting the request: {response.text[:500]}")

    def release(self) -> None:
        self.client.close()
