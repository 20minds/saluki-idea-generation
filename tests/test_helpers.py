from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace
from typing import cast

from langchain_openai import ChatOpenAI

from saluki.deepagents_helpers import (
    _build_chat_model,
    _create_deep_agent,
    _extract_text,
    _search_tools,
    exa_search,
    tivaly_search,
)
from saluki.state import Hypothesis, _upsert_hypotheses


def test_extract_text_uses_last_message_content() -> None:
    result = {
        "messages": [
            SimpleNamespace(content=" first "),
            SimpleNamespace(content=" final "),
        ]
    }
    assert _extract_text(result) == "final"


def test_extract_text_falls_back_to_string() -> None:
    assert _extract_text(123) == "123"


def test_build_chat_model_no_base_url(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    llm = _build_chat_model("gpt-4o-mini")
    assert isinstance(llm, ChatOpenAI)
    assert llm.model_name == "gpt-4o-mini"


def test_build_chat_model_with_base_url(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8080/v1")
    llm = _build_chat_model("local-model")
    assert isinstance(llm, ChatOpenAI)
    assert str(llm.openai_api_base) == "http://localhost:8080/v1"
    assert llm.use_responses_api is False


def test_build_chat_model_uses_chat_completions_endpoint(monkeypatch) -> None:
    paths: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            paths.append(self.path)
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            body = {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "local-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def log_message(self, format: str, *args: object) -> None:
            _ = (format, args)
            return None

    server = HTTPServer(("127.0.0.1", 0), Handler)
    host, port = cast(tuple[str, int], server.server_address)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        monkeypatch.setenv("OPENAI_BASE_URL", f"http://{host}:{port}/v1")
        llm = _build_chat_model("local-model")
        llm.invoke("ping")
    finally:
        server.shutdown()
        thread.join()
        server.server_close()

    assert paths == ["/v1/chat/completions"]


def test_tivaly_search_logs_and_returns(monkeypatch) -> None:
    """Deep-agent tool use is logged and the result is passed through."""
    import saluki.deepagents_helpers as dh
    from loguru import logger

    class FakeClient:
        def search(self, query, **kwargs):
            return {"results": [{"title": "T", "content": "C"}], "query": query}

    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(m), level="INFO")
    try:
        monkeypatch.setattr(dh, "_tavily_client", FakeClient())
        out = dh.tivaly_search("my query", max_results=3)
    finally:
        logger.remove(sink_id)

    assert out["results"][0]["title"] == "T"
    assert any("tool:tivaly_search" in m for m in messages)


def test_tivaly_search_without_client_returns_empty(monkeypatch) -> None:
    import saluki.deepagents_helpers as dh

    monkeypatch.setattr(dh, "_tavily_client", None)
    out = dh.tivaly_search("q")
    assert out == {"results": [], "query": "q"}


def test_tivaly_search_handles_client_error(monkeypatch) -> None:
    import saluki.deepagents_helpers as dh

    class BoomClient:
        def search(self, query, **kwargs):
            raise RuntimeError("api down")

    monkeypatch.setattr(dh, "_tavily_client", BoomClient())
    out = dh.tivaly_search("q")
    assert out == {"results": [], "query": "q"}


def test_exa_search_without_key_returns_empty(monkeypatch) -> None:
    import saluki.deepagents_helpers as dh

    monkeypatch.setattr(dh, "_exa_api_key", "")
    out = dh.exa_search("q")
    assert out == {"results": [], "query": "q"}


def test_exa_search_posts_and_normalizes(monkeypatch) -> None:
    import saluki.deepagents_helpers as dh

    captured: dict[str, object] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "results": [
                    {
                        "title": "Paper",
                        "url": "https://example.com/paper",
                        "text": "evidence",
                        "publishedDate": "2026-01-01",
                        "score": 0.9,
                    }
                ]
            }

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr(dh, "_exa_api_key", "exa-key")
    monkeypatch.setattr(dh.httpx, "post", fake_post)

    out = dh.exa_search("my query", max_results=2)

    assert captured["url"] == "https://api.exa.ai/search"
    assert captured["json"] == {
        "query": "my query",
        "numResults": 2,
        "type": "auto",
        "contents": {"text": True},
    }
    assert out["results"][0]["title"] == "Paper"
    assert out["results"][0]["content"] == "evidence"


def test_search_tools_only_includes_configured_providers(monkeypatch) -> None:
    import saluki.deepagents_helpers as dh

    monkeypatch.setattr(dh, "_tavily_client", None)
    monkeypatch.setattr(dh, "_exa_api_key", "")
    assert _search_tools() == []

    monkeypatch.setattr(dh, "_tavily_client", object())
    assert _search_tools() == [tivaly_search]

    monkeypatch.setattr(dh, "_exa_api_key", "exa-key")
    assert _search_tools() == [tivaly_search, exa_search]

    monkeypatch.setattr(dh, "_tavily_client", None)
    assert _search_tools() == [exa_search]


def test_create_deep_agent_uses_imported_factory(monkeypatch) -> None:
    import saluki.deepagents_helpers as dh

    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setattr(dh, "_tavily_client", object())
    monkeypatch.setattr(dh, "_exa_api_key", "exa-key")
    captured: dict[str, object] = {}

    def fake_create_deep_agent(**kwargs: object) -> str:
        captured.update(kwargs)
        return "agent"

    monkeypatch.setitem(
        sys.modules,
        "deepagents",
        SimpleNamespace(create_deep_agent=fake_create_deep_agent),
    )

    agent = _create_deep_agent(
        model="gpt-4o-mini",
        system_prompt="  hi  ",
        subagents=[{"name": "sub"}],
    )

    assert agent == "agent"
    model = captured["model"]
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "gpt-4o-mini"
    assert captured["system_prompt"] == "hi"
    assert captured["subagents"] == [{"name": "sub"}]
    tools = cast(list[object], captured["tools"])
    assert len(tools) == 2


def test_create_deep_agent_uses_local_chat_completions_endpoint(monkeypatch) -> None:
    import saluki.deepagents_helpers as dh

    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8080/v1")
    monkeypatch.setattr(dh, "_tavily_client", None)
    monkeypatch.setattr(dh, "_exa_api_key", "")
    captured: dict[str, object] = {}

    def fake_create_deep_agent(**kwargs: object) -> str:
        captured.update(kwargs)
        return "agent"

    monkeypatch.setitem(
        sys.modules,
        "deepagents",
        SimpleNamespace(create_deep_agent=fake_create_deep_agent),
    )

    agent = _create_deep_agent(
        model="local-model",
        system_prompt="review",
        subagents=[],
    )

    assert agent == "agent"
    model = captured["model"]
    assert isinstance(model, ChatOpenAI)
    assert str(model.openai_api_base) == "http://127.0.0.1:8080/v1"
    assert model.use_responses_api is False


def test_upsert_hypotheses_merges_by_id() -> None:
    existing = cast(list[Hypothesis], [{"id": "a", "elo": 1}, {"id": "b", "elo": 2}])
    update = cast(list[Hypothesis], [{"id": "b", "elo": 9}, {"id": "c", "elo": 3}])
    merged = _upsert_hypotheses(existing, update)

    assert merged == [
        {"id": "a", "elo": 1},
        {"id": "b", "elo": 9},
        {"id": "c", "elo": 3},
    ]
