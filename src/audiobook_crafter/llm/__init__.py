"""The LLM "brain": local models through Ollama / LM Studio / llama.cpp / vLLM, or your own API key."""

from __future__ import annotations

import os
from dataclasses import dataclass

from ..config import LLMConfig
from .base import LLM, LLMError


@dataclass(frozen=True)
class Provider:
    kind: str  # "ollama" | "openai" | "anthropic"
    base_url: str | None
    key_env: tuple[str, ...] = ()
    json_mode: str = "schema"


PROVIDERS: dict[str, Provider] = {
    "ollama": Provider("ollama", "http://localhost:11434"),
    "lmstudio": Provider("openai", "http://localhost:1234/v1"),
    "llamacpp": Provider("openai", "http://localhost:8080/v1"),
    "vllm": Provider("openai", "http://localhost:8000/v1"),
    "openai": Provider("openai", "https://api.openai.com/v1", ("OPENAI_API_KEY",)),
    "gemini": Provider(
        "openai",
        "https://generativelanguage.googleapis.com/v1beta/openai",
        ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    ),
    "openrouter": Provider("openai", "https://openrouter.ai/api/v1", ("OPENROUTER_API_KEY",)),
    "groq": Provider("openai", "https://api.groq.com/openai/v1", ("GROQ_API_KEY",)),
    "deepseek": Provider("openai", "https://api.deepseek.com", ("DEEPSEEK_API_KEY",), json_mode="object"),
    "together": Provider("openai", "https://api.together.xyz/v1", ("TOGETHER_API_KEY",)),
    "anthropic": Provider("anthropic", None, ("ANTHROPIC_API_KEY",)),
}


def parse_spec(spec: str) -> tuple[str, str]:
    """ "ollama:qwen3.5:9b" -> ("ollama", "qwen3.5:9b"); the model part may itself contain colons."""
    provider, _, model = spec.strip().partition(":")
    return provider.lower(), model


def create_llm(config: LLMConfig) -> LLM | None:
    """None means "no LLM": rule-based attribution only."""
    provider_name, model = parse_spec(config.model)
    if provider_name in ("", "none", "off", "heuristic"):
        return None
    provider = PROVIDERS.get(provider_name)
    if provider is None:
        if not config.base_url:
            known = ", ".join(sorted(PROVIDERS))
            raise LLMError(
                f"Unknown LLM provider '{provider_name}'. Use one of {known}, or set llm.base_url."
            )
        provider = Provider("openai", config.base_url)

    env_names = (config.api_key_env,) if config.api_key_env else provider.key_env
    api_key = next((os.environ[n] for n in env_names if os.environ.get(n)), None)
    base_url = config.base_url or provider.base_url

    if provider.kind == "anthropic":
        from .anthropic_api import AnthropicLLM

        return AnthropicLLM(model, config, api_key=api_key)
    if not model:
        raise LLMError(f"Name a model, e.g. '{provider_name}:<model>'")
    if provider.kind == "ollama":
        from .ollama import OllamaLLM

        return OllamaLLM(model, config, base_url=base_url or "http://localhost:11434")
    if provider.key_env and not api_key:
        raise LLMError(f"{provider_name} needs an API key in ${' or $'.join(env_names)}")
    from .openai_compat import OpenAICompatLLM

    return OpenAICompatLLM(
        provider_name, model, config, base_url=base_url, api_key=api_key, json_mode=provider.json_mode
    )


__all__ = ["LLM", "LLMError", "PROVIDERS", "create_llm", "parse_spec"]
