"""Anthropic models through the official Anthropic SDK: pip install 'audiobook-crafter[anthropic]'."""

from __future__ import annotations

import logging
from typing import Any

from ..config import LLMConfig
from .base import LLM, LLMError, TransientLLMError

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5-5"


class AnthropicLLM(LLM):
    provider = "anthropic"

    def __init__(self, model: str, config: LLMConfig, api_key: str | None = None):
        super().__init__(model or DEFAULT_MODEL, config)
        try:
            import anthropic
        except ImportError as err:
            raise LLMError(
                "The anthropic provider needs its SDK: pip install 'audiobook-crafter[anthropic]'"
            ) from err
        self._sdk = anthropic
        # Without an explicit key the SDK resolves ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN or an
        # `ant auth login` profile on its own.
        options: dict[str, Any] = {"timeout": config.timeout_s}
        if api_key:
            options["api_key"] = api_key
        self.client = anthropic.Anthropic(**options)

    def chat(self, system: str, messages: list[dict[str, str]], schema: dict[str, Any] | None) -> str:
        sdk = self._sdk
        output_config: dict[str, Any] = {}
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        if self.config.effort:
            output_config["effort"] = self.config.effort
        request: dict[str, Any] = {
            "model": self.model,
            # Thinking is always on for current models and counts toward max_tokens.
            "max_tokens": max(self.config.max_output_tokens, 16000),
            "system": system,
            "messages": messages,
            # If a safety classifier declines a passage, let the API retry it on its recommended model.
            "betas": ["server-side-fallback-2026-07-01"],
            "fallbacks": "default",
        }
        if output_config:
            request["output_config"] = output_config
        try:
            response = self.client.beta.messages.create(**request)
        except sdk.AuthenticationError as err:
            raise LLMError(
                "Anthropic rejected the credentials; set ANTHROPIC_API_KEY or run `ant auth login`."
            ) from err
        except sdk.NotFoundError as err:
            raise LLMError(f"Anthropic has no model '{self.model}'.") from err
        except sdk.RateLimitError as err:
            raise TransientLLMError("Anthropic rate limit") from err
        except sdk.APIStatusError as err:
            if err.status_code >= 500:
                raise TransientLLMError(f"Anthropic error {err.status_code}") from err
            raise LLMError(f"Anthropic error {err.status_code}: {err.message}") from err
        except sdk.APIConnectionError as err:
            raise TransientLLMError(f"Can't reach Anthropic: {err}") from err

        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) or "unspecified"
            raise LLMError(f"The model declined this passage (category: {category}).")
        if response.stop_reason == "max_tokens":
            log.warning("The model hit max_tokens; the JSON may be cut short")
        return "".join(block.text for block in response.content if block.type == "text")

    def release(self) -> None:
        self.client.close()
