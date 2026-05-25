from __future__ import annotations

import os
from typing import Any, Literal, Sequence, cast
from textwrap import dedent

import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from loguru import logger
from tavily import TavilyClient


load_dotenv()

_tavily_api_key = os.environ.get("TAVILY_API_KEY", "")
_tavily_client: TavilyClient | None = (
    TavilyClient(api_key=_tavily_api_key) if _tavily_api_key else None
)
_exa_api_key = os.environ.get("EXA_API_KEY", "")


def tivaly_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a Tivaly web search.

    Max query length is 400 characters.
    """
    logger.info(
        "  tool:tivaly_search | query={!r} max_results={} topic={} raw={}",
        query,
        max_results,
        topic,
        include_raw_content,
    )
    if _tavily_client is None:
        logger.warning(
            "  tool:tivaly_search | Tavily unavailable (no TAVILY_API_KEY) — returning no results"
        )
        return {"results": [], "query": query}
    try:
        result = _tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
        )
        n = len(result.get("results", [])) if isinstance(result, dict) else 0
        logger.info("  tool:tivaly_search | query={!r} → {} results", query, n)
        return result
    except Exception as exc:
        logger.warning("  tool:tivaly_search | query={!r} failed: {}", query, exc)
        return {"results": [], "query": query}


def exa_search(
    query: str,
    max_results: int = 5,
    include_text: bool = True,
    search_type: Literal["auto", "neural", "keyword"] = "auto",
) -> dict:
    """Run an Exa internet search."""
    logger.info(
        "  tool:exa_search | query={!r} max_results={} type={} text={}",
        query,
        max_results,
        search_type,
        include_text,
    )
    if not _exa_api_key:
        logger.warning(
            "  tool:exa_search | Exa unavailable (no EXA_API_KEY) — returning no results"
        )
        return {"results": [], "query": query}

    payload: dict[str, object] = {
        "query": query,
        "numResults": max_results,
        "type": search_type,
    }
    if include_text:
        payload["contents"] = {"text": True}

    try:
        response = httpx.post(
            "https://api.exa.ai/search",
            headers={
                "x-api-key": _exa_api_key,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        results = _normalize_exa_results(data.get("results", []))
        logger.info("  tool:exa_search | query={!r} → {} results", query, len(results))
        return {"results": results, "query": query}
    except Exception as exc:  # noqa: BLE001 - search failure should not abort a run
        logger.warning("  tool:exa_search | query={!r} failed: {}", query, exc)
        return {"results": [], "query": query}


def _normalize_exa_results(results: object) -> list[dict[str, object]]:
    if not isinstance(results, list):
        return []
    normalized: list[dict[str, object]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        payload = cast(dict[str, object], item)
        title = payload.get("title") or payload.get("url") or ""
        content = payload.get("text") or payload.get("summary") or ""
        normalized.append(
            {
                "title": title,
                "url": payload.get("url", ""),
                "content": content,
                "published_date": payload.get("publishedDate", ""),
                "score": payload.get("score"),
            }
        )
    return normalized


def _search_tools() -> list[Any]:
    tools: list[Any] = []
    if _tavily_client is not None:
        tools.append(tivaly_search)
    if _exa_api_key:
        tools.append(exa_search)
    return tools


def _search_tool_names(tools: Sequence[Any]) -> list[str]:
    return [getattr(tool, "__name__", str(tool)) for tool in tools]


def _run_agent(agent: Any, prompt: str) -> str:
    """Invoke a deep agent with a single user message and return its final text."""
    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    return _extract_text(result)


def _extract_text(result: object) -> str:
    if isinstance(result, dict):
        payload = cast(dict[str, object], result)
        messages = payload.get("messages")
        if isinstance(messages, list) and messages:
            last = messages[-1]
            content = getattr(last, "content", None)
            if content:
                return str(content).strip()
    return str(result).strip()


def _build_chat_model(model: str) -> ChatOpenAI:
    """Build a ChatOpenAI instance that respects OPENAI_BASE_URL.

    When a custom base URL is set (local server), force the chat-completions
    endpoint — local servers implement /v1/chat/completions, not /v1/responses.
    """
    base_url = os.environ.get("OPENAI_BASE_URL")
    kwargs: dict[str, Any] = {"model": model}
    if base_url:
        logger.info("  deepagents llm | model={} base_url={}", model, base_url)
        kwargs["base_url"] = base_url
        kwargs["use_responses_api"] = False
    return ChatOpenAI(**kwargs)


def _create_deep_agent(*, model: str, system_prompt: str, subagents: Sequence[Any]):
    try:
        from deepagents import create_deep_agent
    except ImportError as exc:  # pragma: no cover - runtime dependency guard
        raise RuntimeError(
            "deepagents is not installed. Add it to the environment to enable this feature."
        ) from exc

    tools = _search_tools()
    logger.info(
        "  deepagents search tools: {}",
        ", ".join(_search_tool_names(tools)) if tools else "none",
    )

    return create_deep_agent(
        model=_build_chat_model(model),
        system_prompt=dedent(system_prompt).strip(),
        subagents=subagents,
        tools=tools,
    )
