"""LLM adapters against fake HTTP servers (httpx.MockTransport): exact payloads and fallbacks."""

from __future__ import annotations

import json

import httpx
import pytest

from audiobook_crafter.config import LLMConfig
from audiobook_crafter.llm import LLMError, create_llm, parse_spec
from audiobook_crafter.llm.ollama import OllamaLLM
from audiobook_crafter.llm.openai_compat import OpenAICompatLLM

SCHEMA = {
    "type": "object",
    "properties": {"a": {"type": "integer"}},
    "required": ["a"],
    "additionalProperties": False,
}


def client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_parse_spec_keeps_colons_in_model() -> None:
    assert parse_spec("ollama:qwen3.5:9b") == ("ollama", "qwen3.5:9b")
    assert parse_spec("none") == ("none", "")


def test_ollama_payload_and_unload() -> None:
    seen: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append((request.url.path, body))
        if request.url.path == "/api/chat":
            return httpx.Response(200, json={"message": {"content": '{"a": 1}'}, "prompt_eval_count": 100})
        return httpx.Response(200, json={})

    llm = OllamaLLM("qwen3.5:9b", LLMConfig(context_tokens=32768, think=False), "http://localhost:11434")
    llm.client = client(handler)
    assert llm.complete_json("sys", "prompt", SCHEMA) == {"a": 1}
    path, body = seen[0]
    assert path == "/api/chat"
    assert body["options"]["num_ctx"] == 32768 and body["format"] == SCHEMA and body["think"] is False
    assert body["messages"][0] == {"role": "system", "content": "sys"} and body["stream"] is False
    llm.release()
    assert seen[-1] == ("/api/generate", {"model": "qwen3.5:9b", "keep_alive": 0})


def test_ollama_missing_model_is_a_clear_error() -> None:
    llm = OllamaLLM("nope:1b", LLMConfig(), "http://localhost:11434")
    llm.client = client(lambda request: httpx.Response(404, json={"error": "model not found"}))
    with pytest.raises(LLMError, match="ollama pull nope:1b"):
        llm.complete_json("s", "p", SCHEMA)


def test_openai_compat_drops_what_the_server_rejects() -> None:
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if "response_format" in body and body["response_format"]["type"] == "json_schema":
            return httpx.Response(
                400, json={"error": {"message": "response_format json_schema not supported"}}
            )
        if "temperature" in body:
            return httpx.Response(400, json={"error": {"message": "Unsupported value: 'temperature'"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": '```json\n{"a": 2}\n```'}}]})

    llm = OpenAICompatLLM("gemini", "some-model", LLMConfig(), "https://example.test/v1", api_key="k")
    llm.client = client(handler)
    assert llm.complete_json("s", "p", SCHEMA) == {"a": 2}
    assert bodies[1]["response_format"] == {"type": "json_object"}
    assert "temperature" not in bodies[-1]
    # The adaptations are remembered: the next call goes straight through.
    bodies.clear()
    assert llm.complete_json("s", "p", SCHEMA) == {"a": 2}
    assert len(bodies) == 1


def test_invalid_json_is_asked_again() -> None:
    replies = iter(["sorry, here you go: a=1", '{"a": 3}'])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": next(replies)}}]})

    llm = OpenAICompatLLM("lmstudio", "m", LLMConfig(), "http://localhost:1234/v1", api_key=None)
    llm.client = client(handler)
    assert llm.complete_json("s", "p", SCHEMA) == {"a": 3}


def test_create_llm_requires_keys_for_hosted_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(LLMError, match="OPENROUTER_API_KEY"):
        create_llm(LLMConfig(model="openrouter:vendor/model"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    assert create_llm(LLMConfig(model="openrouter:vendor/model")).name == "openrouter:vendor/model"
    assert create_llm(LLMConfig(model="none")) is None
    with pytest.raises(LLMError, match="Unknown LLM provider"):
        create_llm(LLMConfig(model="mystery:m"))
    custom = create_llm(LLMConfig(model="mine:m", base_url="http://box:9000/v1"))
    assert isinstance(custom, OpenAICompatLLM) and custom.base_url == "http://box:9000/v1"
