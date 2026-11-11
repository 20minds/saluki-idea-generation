from __future__ import annotations
import json
import os
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from itertools import combinations
from typing import Any, Callable, Literal, cast

from langchain_openai import ChatOpenAI
from loguru import logger
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .schemas import (
    GeneratedHypothesis,
    ScoreResult,
    CompareResult,
    EvolveResult,
    ResearchPlan,
    VerificationResult,
)
from .state import CoScientistState, Hypothesis, lookup_similarity
from .idea_research import run_idea_research
from .deep_reflect import run_deep_reflection
from .deepagents_helpers import _exa_api_key, exa_search, tivaly_search
from .prompts import (
    RESEARCH_PLAN,
    GENERATE,
    GENERATE_DECISION,
    GENERATE_DEBATE,
    REFLECT,
    SCORE,
    VERIFY,
    COMPARE,
    EVOLVE,
    EVOLVE_DECISION,
    EVOLVE_SIMPLIFY,
    EVOLVE_SIMPLIFY_DECISION,
    EVOLVE_DIVERGE,
    EVOLVE_DIVERGE_DECISION,
    NEGATIVE_SPACE_ADDON,
    META_REVIEW,
    SYSTEM_FEEDBACK,
    FINAL_OVERVIEW,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationProgress:
    """Structured lifecycle update for incremental hypothesis generation."""

    phase: Literal["generating", "debating", "complete"]
    candidate_count: int


NONE_YET_TEXT = "None yet."

SIMILARITY_SKIP_THRESHOLD = 0.7  # tournament: skip pairs at or above this similarity
DUPLICATE_THRESHOLD = (
    0.9  # proximity: prune the lower-Elo hypothesis in a near-duplicate pair
)

# Elo semantics (pseudocode AddToTournament): a hypothesis has no Elo until it
# enters the tournament, at which point it is seeded with TOURNAMENT_START_ELO.
INITIAL_ELO_SENTINEL = 0.0
TOURNAMENT_START_ELO = 1200.0

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


# Structured-output calls against small local models intermittently emit
# malformed JSON or hit the length limit; a bounded retry usually recovers.
STRUCTURED_RETRIES = 2
STRUCTURED_RETRY_MIN_SECONDS = 2.0
STRUCTURED_RETRY_MAX_SECONDS = 30.0

COMPACT_INPUT_CHARS = int(os.environ.get("SALUKI_COMPACT_INPUT_CHARS", "12000"))
IDEA_RESEARCH_SUMMARY_CHARS = int(
    os.environ.get("SALUKI_IDEA_RESEARCH_SUMMARY_CHARS", "3000")
)
REFLECTION_SUMMARY_CHARS = int(os.environ.get("SALUKI_REFLECTION_SUMMARY_CHARS", "900"))
META_REVIEW_SUMMARY_CHARS = int(
    os.environ.get("SALUKI_META_REVIEW_SUMMARY_CHARS", "1200")
)
SYSTEM_FEEDBACK_SUMMARY_CHARS = int(
    os.environ.get("SALUKI_SYSTEM_FEEDBACK_SUMMARY_CHARS", "1200")
)


def _llm(model: str, temperature: float = 0.7) -> ChatOpenAI:
    base_url = os.environ.get("OPENAI_BASE_URL")
    kwargs: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
    }
    if base_url:
        kwargs["base_url"] = base_url
        # Local servers only implement /v1/chat/completions, not /v1/responses.
        kwargs["use_responses_api"] = False
    return ChatOpenAI(**kwargs)


VLLM_DISABLE_THINKING_BODY: dict[str, Any] = {
    "chat_template_kwargs": {"enable_thinking": False},
}


def _compatibility_provider() -> str:
    """Return the explicitly configured OpenAI-compatible server profile."""
    return os.environ.get("SALUKI_OPENAI_COMPAT_PROVIDER", "auto").strip().lower()


def _structured_extra_body() -> dict | None:
    """Extra request-body params applied to structured-output calls only.

    Lets you pass server-specific options the OpenAI schema does not model — most
    usefully, disabling a model's reasoning/thinking pass so
    ``with_structured_output`` calls return JSON directly instead of burning the
    token budget on chain of thought.  Scoped to structured calls so free-form
    nodes (idea research, meta-review, the final overview) keep their reasoning.

    Resolution order:

    * ``SALUKI_LLM_EXTRA_BODY`` set to JSON  → use it verbatim (override).
      Use ``'{}'`` to opt out (send nothing).
    * unset, and ``SALUKI_OPENAI_COMPAT_PROVIDER=vllm`` or ``sglang`` → use
      the documented vLLM-style ``chat_template_kwargs`` thinking toggle.
    * unset for ``auto``, ``openai``, or ``generic`` → send no non-standard
      parameters. This is safe for stock OpenAI and arbitrary compatible APIs.
    """
    raw = os.environ.get("SALUKI_LLM_EXTRA_BODY")
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("ignoring invalid SALUKI_LLM_EXTRA_BODY: {}", exc)
            return None
    if _compatibility_provider() in {"vllm", "sglang"}:
        return dict(VLLM_DISABLE_THINKING_BODY)
    return None


def _with_extra_body(llm: Any) -> Any:
    """Return *llm* with the structured-output extra body applied, if configured."""
    extra = _structured_extra_body()
    if extra and hasattr(llm, "model_copy"):
        merged = {**(getattr(llm, "extra_body", None) or {}), **extra}
        return llm.model_copy(update={"extra_body": merged})
    return llm


def _format_exception(exc: BaseException) -> str:
    """Format an exception with chained causes for concise operational logs."""
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current).strip() or repr(current)
        parts.append(f"{type(current).__name__}: {message}")
        current = current.__cause__ or current.__context__
    return " | caused by ".join(parts)


def _structured(llm: ChatOpenAI, schema: type) -> Any:
    """Return a structured-output chain for *schema*.

    With a local base_url, use json_mode so that langchain routes the response
    through PydanticOutputParser (which calls parse_json_markdown and strips
    markdown fences) instead of the openai SDK's strict json_schema validation
    path, which rejects responses that wrap valid JSON in code fences.

    Applies the optional ``SALUKI_LLM_EXTRA_BODY`` (e.g. to disable model
    reasoning) here so it affects only structured-output calls.
    """
    llm = _with_extra_body(llm)
    if getattr(llm, "openai_api_base", None):
        return llm.with_structured_output(schema, method="json_mode")
    return llm.with_structured_output(schema)


def _log_structured_retry(label: str):
    def _log(retry_state: RetryCallState) -> None:
        outcome = retry_state.outcome
        exc = outcome.exception() if outcome is not None and outcome.failed else None
        if exc is None:
            return
        logger.warning(
            "{} | attempt {}/{} failed; retrying in {:.1f}s: {}",
            label,
            retry_state.attempt_number,
            STRUCTURED_RETRIES,
            retry_state.next_action.sleep if retry_state.next_action else 0.0,
            _format_exception(exc),
        )

    return _log


def _invoke_structured(llm: Any, prompt: str, *, label: str = "structured") -> Any:
    """Invoke a structured-output chain, retrying transient failures.

    Small local models occasionally return malformed JSON; retrying usually
    recovers.  Re-raises the last error if every attempt fails so callers can
    decide whether to skip or abort.
    """
    retryer = Retrying(
        stop=stop_after_attempt(STRUCTURED_RETRIES),
        wait=wait_exponential(
            multiplier=1,
            min=STRUCTURED_RETRY_MIN_SECONDS,
            max=STRUCTURED_RETRY_MAX_SECONDS,
        ),
        retry=retry_if_exception_type(Exception),
        reraise=True,
        before_sleep=_log_structured_retry(label),
    )

    for attempt in retryer:
        with attempt:
            return llm.invoke(prompt)
    raise RuntimeError(f"{label} retry loop exited without result")


def _elo_update(ra: float, rb: float, winner_id: str, h1_id: str, k: float = 32.0):
    """Update ELO scores. winner_id is either h1_id or h2_id."""
    ea = 1 / (1 + 10 ** ((rb - ra) / 400))
    eb = 1 - ea
    if winner_id == h1_id:
        return ra + k * (1 - ea), rb + k * (0 - eb)
    return ra + k * (0 - ea), rb + k * (1 - eb)


def _active(state: CoScientistState) -> list[Hypothesis]:
    pruned = set(state.get("pruned_ids") or [])
    active = [h for h in state["hypotheses"] if h["id"] not in pruned]
    return sorted(
        active, key=lambda h: h.get("elo", INITIAL_ELO_SENTINEL), reverse=True
    )


def _in_tournament(h: Hypothesis) -> bool:
    """A hypothesis is in the tournament once AddToTournament has seeded its Elo."""
    return bool(h.get("in_tournament", False))


def _truncate_text(text: str, max_chars: int = 4000) -> str:
    """Truncate text to max_chars, preserving word boundaries."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def _context_text(state: CoScientistState) -> str:
    if state.get("idea_research"):
        return state["idea_research"]
    text = "\n".join(
        f"- {e['title']}: {e.get('abstract', '')}" for e in state.get("literature", [])
    )
    return text or "None provided."


def _feedback_text(state: CoScientistState) -> str:
    """Return the best available feedback for prompt injection.

    Priority: system_wide_feedback (cross-round) > meta_review (per-round) > None.
    """
    swf = (state.get("system_wide_feedback") or "").strip()
    if swf:
        return swf
    return (state.get("meta_review") or "").strip() or NONE_YET_TEXT


def _message_text(message: object) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    return str(content).strip()


def _compact_for_state(
    text: str,
    *,
    label: str,
    model: str,
    max_chars: int,
    guidance: str,
) -> str:
    """Summarize long text before storing it in shared state.

    State fields are repeatedly fed into later prompts, so storing compact
    summaries is the most effective place to control context growth.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return text

    source = _truncate_text(text, COMPACT_INPUT_CHARS)
    prompt = f"""\
Summarize the following {label} for reuse as compact context in later LLM prompts.
Keep the operational details that affect future hypothesis generation, review, or ranking.
{guidance}

Limit the summary to at most {max_chars} characters. Return only the summary.

{label}:
{source}
"""
    try:
        summary = _message_text(_llm(model, temperature=0.0).invoke(prompt))
    except Exception as exc:  # noqa: BLE001 - compaction must not break the run
        logger.warning(
            "compact | {} summarization failed, falling back to truncation: {}",
            label,
            _format_exception(exc),
        )
        return _truncate_text(text, max_chars)

    summary = summary.strip()
    if not summary:
        return _truncate_text(text, max_chars)
    if len(summary) > max_chars:
        summary = _truncate_text(summary, max_chars)
    logger.info("compact | {} {}→{} chars", label, len(text), len(summary))
    return summary


def _hypothesis_text(h: Hypothesis) -> str:
    return " ".join(
        part for part in [h["mechanism"], h["prediction"], h["experiment"]] if part
    )


def _format_ranked(
    hypotheses: list[Hypothesis], limit: int, empty: str = "  No hypotheses yet."
) -> str:
    """Format the top *limit* hypotheses as a numbered, Elo-tagged list."""
    return (
        "\n".join(
            f"{i + 1}. [{h['id']} elo={h.get('elo', 0.0):.0f}] {h['mechanism']}"
            for i, h in enumerate(hypotheses[:limit])
        )
        or empty
    )


def _format_matches(
    log: list[dict],
    *,
    limit: int | None = None,
    prefix: str = "  ",
    empty: str = "  No matches yet.",
) -> str:
    """Format tournament-log entries as 'winner beat loser: reason' lines."""
    entries = log[-limit:] if limit else log
    return (
        "\n".join(
            f"{prefix}{e['winner']} beat {e['loser']}: {e.get('reason', '')}"
            for e in entries
        )
        or empty
    )


def _similarity(a: Hypothesis, b: Hypothesis) -> float:
    a_text = _hypothesis_text(a).lower()
    b_text = _hypothesis_text(b).lower()
    return SequenceMatcher(None, a_text, b_text).ratio()


def _pair_similarity(h1_id: str, h2_id: str, proximity_graph: dict) -> float:
    """Look up precomputed similarity between two hypotheses from the proximity graph.

    Returns 0.0 if the pair is not in the graph (i.e. similarity was below the
    graph inclusion threshold, so definitely not redundant).
    """
    return lookup_similarity(proximity_graph, h1_id, h2_id)


@lru_cache(maxsize=1)
def _embedder() -> Any:
    """Load the FastEmbed model once and cache it for the process.

    Kept lazy so importing this module never initializes ONNX Runtime or
    downloads the model unless embeddings are actually used.
    """
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=EMBEDDING_MODEL)


def _embed_texts(texts: list[str]) -> list[list[float]] | None:
    """Embed *texts* with the local EmbeddingGemma model.

    Returns ``None`` if embeddings are unavailable (model not installed,
    offline, or any error) so callers can fall back to lexical similarity.
    This keeps the proximity step usable in tests and offline runs.
    """
    if not texts:
        return None
    try:
        return [vector.tolist() for vector in _embedder().embed(texts)]
    except Exception as exc:  # pragma: no cover - exercised via monkeypatch in tests
        logger.debug(
            "embeddings unavailable, falling back to lexical similarity: {}", exc
        )
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _web_context(
    query: str, keywords: list[str] | None = None, max_results: int = 5
) -> str:
    """Retrieve literature snippets via web search (pseudocode: Strategy 1).

    Returns formatted snippets, or an empty string when search is unavailable
    (search helpers already degrade gracefully to an empty result set).
    """
    search_query = query
    if keywords:
        search_query = f"{query} ({', '.join(keywords[:6])})"
    try:
        search = exa_search if _exa_api_key else tivaly_search
        payload = search(search_query, max_results=max_results)
    except Exception:  # pragma: no cover - defensive
        return ""
    results = (payload or {}).get("results", []) if isinstance(payload, dict) else []
    lines = []
    for r in results:
        title = r.get("title", "") if isinstance(r, dict) else ""
        content = r.get("content", "") if isinstance(r, dict) else ""
        if title or content:
            lines.append(f"- {title}: {_truncate_text(content, 300)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# nodes
# ---------------------------------------------------------------------------


def _plan_to_dict(obj: object) -> dict:
    """Coerce a ResearchPlan-like object (pydantic or namespace) to a plain dict."""
    if isinstance(obj, dict):
        return obj
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        return dump()
    return {k: v for k, v in vars(obj).items()} if hasattr(obj, "__dict__") else {}


def parse_research_plan(question: str, *, model: str = "gpt-4o-mini") -> dict:
    """Parse the research goal into a structured ResearchPlan (pseudocode start step).

    Degrades gracefully to an empty plan if the LLM is unavailable so bootstrap
    never hard-fails.
    """
    try:
        llm = _structured(_llm(model, temperature=0.0), ResearchPlan)
        result = _invoke_structured(
            llm, RESEARCH_PLAN.format(question=question), label="research_plan"
        )
        plan = _plan_to_dict(result)
        logger.info("research_plan | objective={:.120}", str(plan.get("objective", "")))
        return plan
    except Exception as exc:  # pragma: no cover - defensive offline guard
        logger.debug("research plan parsing failed: {}", exc)
        return {}


def bootstrap_node(state: CoScientistState, *, model: str = "gpt-4o-mini") -> dict:
    logger.info("bootstrap | question: {}", state["question"])
    plan = state.get("research_plan") or parse_research_plan(
        state["question"], model=model
    )
    return {
        "round": 0,
        "research_plan": plan,
        "pruned_ids": [],
        "tournament_log": [],
        "reviews": [],
        "meta_review": state.get("meta_review", ""),
        "final_overview": "",
        "proximity_graph": {},
        "idea_research": state.get("idea_research", ""),
        "deep_reflect": state.get("deep_reflect", True),
        "hypotheses": [],
        "literature": [],
    }


def idea_research_node(state: CoScientistState, *, model: str = "gpt-4o-mini") -> dict:
    logger.info("idea_research | model={}", model)
    existing_text = (
        "\n".join(f"- {h['mechanism']}" for h in _active(state)) or NONE_YET_TEXT
    )
    feedback = _feedback_text(state)
    brief = run_idea_research(
        question=state["question"],
        model=model,
        existing=existing_text,
        feedback=feedback,
    )
    brief = _compact_for_state(
        brief,
        label="idea research brief",
        model=model,
        max_chars=IDEA_RESEARCH_SUMMARY_CHARS,
        guidance=(
            "Preserve promising directions, key assumptions, risks, citations or named "
            "prior-art signals, and suggested next experiments."
        ),
    )
    logger.info("idea_research | brief={:.120}", brief)
    return {"idea_research": brief}


def _build_hypothesis(obj: Any, h_id: str, generation: int) -> Hypothesis:
    """Build a fresh, not-yet-in-tournament Hypothesis from an LLM result.

    *obj* must expose ``mechanism``/``prediction``/``experiment`` plus the
    optional ``decision``/``for_whom``/``reversal_condition``/``not_doing``
    fields shared by ``GeneratedHypothesis`` and ``EvolveResult``.
    """
    return {
        "id": h_id,
        "mechanism": obj.mechanism,
        "prediction": obj.prediction,
        "experiment": obj.experiment,
        "elo": INITIAL_ELO_SENTINEL,
        "in_tournament": False,
        "matches": 0,
        "reflection": "",
        "verification": "",
        "novelty_score": 0.5,
        "rigor_score": 0.5,
        "testability_score": 0.5,
        "impact_score": 0.5,
        "simplicity_score": 0.5,
        "generation": generation,
        "decision": obj.decision or "",
        "for_whom": obj.for_whom or "",
        "reversal_condition": obj.reversal_condition or "",
        "not_doing": list(obj.not_doing or []),
    }


def generate_node(
    state: CoScientistState,
    *,
    model: str = "gpt-4o-mini",
    on_hypothesis: Callable[[Hypothesis], None] | None = None,
    on_progress: Callable[[GenerationProgress], None] | None = None,
) -> dict:
    """Create initial hypotheses using two strategies (pseudocode CreateInitialHypotheses).

    Strategy 1 — literature-grounded: a web search supplies external context that is
                 injected alongside the idea-research brief.
    Strategy 2 — simulated expert debate: one consensus hypothesis from a debate.
    """
    n_gen = 12
    round_num = state["round"]
    if on_progress:
        on_progress(GenerationProgress(phase="generating", candidate_count=0))
    logger.info("generate | round={} model={} n={}", round_num, model, n_gen)

    # Strategy 1 context: idea-research brief + live web literature.
    plan = state.get("research_plan") or {}
    web_text = _web_context(state["question"], keywords=plan.get("keywords"))
    brief = _context_text(state)
    lit_text = _truncate_text(
        brief + ("\n\nWeb literature:\n" + web_text if web_text else "")
    )
    if web_text:
        logger.info("  web literature retrieved ({} chars)", len(web_text))

    decision_mode = state.get("decision_mode", False)
    negative_space = state.get("negative_space", False)
    logger.info("  decision_mode={} negative_space={}", decision_mode, negative_space)

    prompt_tmpl = GENERATE_DECISION if decision_mode else GENERATE
    if negative_space:
        prompt_tmpl = prompt_tmpl + NEGATIVE_SPACE_ADDON

    # Incremental generation: produce 1 hypothesis per LLM call,
    # accumulating previously generated ones as "existing" context.
    items: list[Hypothesis] = []
    for _ in range(1, n_gen + 1):
        existing_text = "\n".join(f"- {h['mechanism']}" for h in items) or NONE_YET_TEXT
        h_text = f"h{round_num}_{len(items)}"

        llm = _structured(_llm(model), GeneratedHypothesis)
        try:
            obj = cast(
                GeneratedHypothesis,
                _invoke_structured(
                    llm,
                    prompt_tmpl.format(
                        n=1,  # always ask for exactly 1 new hypothesis
                        question=state["question"],
                        literature=lit_text,
                        existing=existing_text,
                        feedback=_feedback_text(state),
                    ),
                    label=f"generate {h_text}",
                ),
            )
        except Exception:  # noqa: BLE001 - skip a flaky generation, keep the rest
            logger.warning("generate | {} failed after retries, skipping", h_text)
            continue
        h = _build_hypothesis(obj, h_text, round_num)
        items.append(h)
        logger.info("  {} | {}", h["id"], h["mechanism"][:120])
        if on_hypothesis:
            on_hypothesis(h)

    # Strategy 2: simulated expert debate → three consensus hypotheses.
    if on_progress:
        on_progress(GenerationProgress(phase="debating", candidate_count=len(items)))
    debate_tmpl = GENERATE_DEBATE + (NEGATIVE_SPACE_ADDON if negative_space else "")
    debate_llm = _structured(_llm(model), GeneratedHypothesis)
    for debate_idx in range(3):
        existing_text = "\n".join(f"- {h['mechanism']}" for h in items) or NONE_YET_TEXT
        child_id = f"d{round_num}_{debate_idx}"
        try:
            debate_obj = cast(
                GeneratedHypothesis,
                _invoke_structured(
                    debate_llm,
                    debate_tmpl.format(
                        question=state["question"],
                        literature=lit_text,
                        existing=existing_text,
                        feedback=_feedback_text(state),
                    ),
                    label=f"generate {child_id}",
                ),
            )
            debate_h = _build_hypothesis(debate_obj, child_id, round_num)
            items.append(debate_h)
            logger.info(
                "  {} (debate) | {}", debate_h["id"], debate_h["mechanism"][:120]
            )
            if on_hypothesis:
                on_hypothesis(debate_h)
        except Exception:  # noqa: BLE001 - debate hypothesis is optional
            logger.warning(
                "generate | debate hypothesis {} failed after retries, skipping",
                child_id,
            )

    logger.info("  generated {} hypotheses", len(items))
    if on_progress:
        on_progress(GenerationProgress(phase="complete", candidate_count=len(items)))
    return {
        "round": round_num,
        "hypotheses": items,
        "new_hypothesis_ids": [h["id"] for h in items],
    }


def reflect_node(state: CoScientistState, *, model: str = "gpt-4o-mini") -> dict:
    active = _active(state)
    if not active:
        logger.warning("reflect | no active hypotheses, skipping")
        return {}

    logger.info("reflect | model={} n={}", model, len(active))

    if state.get("deep_reflect", True):
        lit_text = _truncate_text(_context_text(state))
        feedback = _feedback_text(state)
        cache = state.get("deep_reflection_cache") or {}
        reflections = {}
        for h in active:
            if h["id"] in cache:
                continue  # skip already-reflected hypotheses
            r = run_deep_reflection(
                question=state["question"],
                mechanism=h["mechanism"],
                prediction=h["prediction"],
                experiment=h["experiment"],
                context=lit_text,
                feedback=feedback,
                model=model,
                subject=f"[{h['id']}] {h['mechanism']}",
            )
            r = _compact_for_state(
                r,
                label=f"reflection {h['id']}",
                model=model,
                max_chars=REFLECTION_SUMMARY_CHARS,
                guidance=(
                    "Preserve novelty, feasibility, contradictions, strongest caveats, "
                    "and the final judgment label."
                ),
            )
            reflections[h["id"]] = r
        # Merge: new reflections + cached reflections
        merged = []
        for h in state["hypotheses"]:
            updated = {**h}
            if h["id"] in reflections:
                updated["reflection"] = reflections[h["id"]]
                cache[h["id"]] = reflections[h["id"]]
            elif h.get("reflection"):
                pass  # keep existing reflection
            else:
                updated["reflection"] = ""
            merged.append(updated)
        return {"hypotheses": merged, "deep_reflection_cache": cache}

    llm = _llm(model, temperature=0.3)
    h = active[0]
    review = llm.invoke(
        REFLECT.format(
            mechanism=h["mechanism"],
            prediction=h["prediction"],
            experiment=h["experiment"],
            literature=_truncate_text(_context_text(state)),
            feedback=_feedback_text(state),
        ),
    )
    reflection = _message_text(review)
    reflection = _compact_for_state(
        reflection,
        label=f"reflection {h['id']}",
        model=model,
        max_chars=REFLECTION_SUMMARY_CHARS,
        guidance=(
            "Preserve novelty, feasibility, contradictions, strongest caveats, "
            "and the final judgment label."
        ),
    )
    logger.info("reflect | {:.120}", reflection)

    merged = [{**h, "reflection": reflection} for h in state["hypotheses"]]
    return {"hypotheses": merged}


NOVELTY_PRUNE_THRESHOLD = 0.2


def _apply_scores(h: Hypothesis, scores: ScoreResult) -> Hypothesis:
    """Return *h* with its five score dimensions overwritten from *scores*."""
    return cast(
        Hypothesis,
        {
            **h,
            "novelty_score": scores.novelty_score,
            "rigor_score": scores.rigor_score,
            "testability_score": scores.testability_score,
            "impact_score": scores.impact_score,
            "simplicity_score": scores.simplicity_score,
        },
    )


def _score_hypothesis(
    h: Hypothesis, lit_text: str, feedback: str, model: str
) -> ScoreResult:
    llm = _structured(_llm(model, temperature=0.0), ScoreResult)
    return cast(
        ScoreResult,
        _invoke_structured(
            llm,
            SCORE.format(
                literature=lit_text,
                feedback=feedback,
                hypotheses=(
                    f"[{h['id']}] {h['mechanism']}\n"
                    f"  Prediction: {h['prediction']}\n"
                    f"  Experiment: {h['experiment']}"
                ),
            ),
            label=f"score {h['id']}",
        ),
    )


def _neutral_scores() -> ScoreResult:
    return ScoreResult(
        novelty_score=0.5,
        rigor_score=0.5,
        testability_score=0.5,
        impact_score=0.5,
        simplicity_score=0.5,
        reason="",
    )


def score_hypotheses(
    state: CoScientistState,
    hypotheses: list[Hypothesis],
    *,
    model: str = "gpt-4o-mini",
    lit_text: str | None = None,
    feedback: str | None = None,
    prune_low_novelty: bool = True,
    tolerate_errors: bool = False,
    log_label: str = "score",
) -> tuple[list[Hypothesis], list[str]]:
    """Score hypotheses with the same five dimensions used throughout Saluki."""
    if not hypotheses:
        return [], []

    lit_text = (
        lit_text if lit_text is not None else _truncate_text(_context_text(state))
    )
    feedback = feedback if feedback is not None else _feedback_text(state)
    scored: list[Hypothesis] = []
    for h in hypotheses:
        try:
            scores = _score_hypothesis(h, lit_text, feedback, model)
        except Exception as exc:
            if not tolerate_errors:
                raise
            logger.warning(
                "{} | id={} scoring failed, using neutral scores: {}",
                log_label,
                h["id"],
                _format_exception(exc),
            )
            scores = _neutral_scores()
        scored_h = _apply_scores(h, scores)
        logger.info(
            "{} | id={} scored novelty={:.2f} rigor={:.2f} testability={:.2f} impact={:.2f} simplicity={:.2f}",
            log_label,
            h["id"],
            scored_h["novelty_score"],
            scored_h["rigor_score"],
            scored_h["testability_score"],
            scored_h["impact_score"],
            scored_h["simplicity_score"],
        )
        scored.append(scored_h)

    pruned = (
        [h["id"] for h in scored if h["novelty_score"] < NOVELTY_PRUNE_THRESHOLD]
        if prune_low_novelty
        else []
    )
    return scored, pruned


def score_node(state: CoScientistState, *, model: str = "gpt-4o-mini") -> dict:
    active = _active(state)
    if not active:
        logger.warning("score | no active hypotheses, skipping")
        return {}

    logger.info("score | model={} n={}", model, len(active))

    merged, pruned = score_hypotheses(state, active, model=model)

    logger.info("score | done pruned={}", pruned)
    result: dict = {"hypotheses": merged}
    if pruned:
        result["pruned_ids"] = pruned
    return result


def _verify_hypothesis(h: Hypothesis, lit_text: str, model: str) -> VerificationResult:
    llm = _structured(_llm(model, temperature=0.0), VerificationResult)
    return cast(
        VerificationResult,
        _invoke_structured(
            llm,
            VERIFY.format(
                mechanism=h["mechanism"],
                prediction=h["prediction"],
                experiment=h["experiment"],
                literature=lit_text,
            ),
            label=f"verify {h['id']}",
        ),
    )


def _reflection_text(
    h: Hypothesis, state: CoScientistState, lit_text: str, model: str
) -> str:
    """Produce the FullReview critique text for one hypothesis."""
    feedback = _feedback_text(state)
    if not state.get("deep_reflect", True):
        return ""
    return run_deep_reflection(
        question=state["question"],
        mechanism=h["mechanism"],
        prediction=h["prediction"],
        experiment=h["experiment"],
        context=lit_text,
        feedback=feedback,
        model=model,
        subject=f"[{h['id']}] {h['mechanism']}",
    )


def review_hypothesis_node(
    state: CoScientistState, target_id: str, *, model: str = "gpt-4o-mini"
) -> dict:
    """Reflection.ReviewHypothesis(TargetID): full review + deep verification.

    Produces a FullReview (critique text + 5-dimension scores) and a
    VerificationReview (assumption decomposition), appends both to the
    ReviewsList, applies novelty pruning, and signals readiness for the
    tournament via ``reviewed_id``.
    """
    target = next((h for h in _active(state) if h["id"] == target_id), None)
    if target is None:
        logger.warning("review | target {} not active, skipping", target_id)
        return {}

    logger.info("review | id={} model={}", target_id, model)

    # Pseudocode ReviewHypothesis: perform a web search for evidence related to
    # *this* hypothesis, then review/verify against it plus the existing context.
    plan = state.get("research_plan") or {}
    web_text = _web_context(target["mechanism"], keywords=plan.get("keywords"))
    context = _context_text(state)
    lit_text = _truncate_text(
        context + ("\n\nWeb evidence:\n" + web_text if web_text else "")
    )
    if web_text:
        logger.info(
            "review | id={} web evidence retrieved ({} chars)", target_id, len(web_text)
        )

    # Degrade gracefully if any sub-review fails after retries: a flaky local
    # model shouldn't abort the whole run — fall back to neutral defaults so the
    # hypothesis still proceeds to the tournament.
    try:
        reflection = _reflection_text(target, state, lit_text, model)
        reflection = _compact_for_state(
            reflection,
            label=f"review reflection {target_id}",
            model=model,
            max_chars=REFLECTION_SUMMARY_CHARS,
            guidance=(
                "Preserve novelty, feasibility, contradictions, strongest caveats, "
                "and the final judgment label."
            ),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "review | id={} reflection failed, using empty critique: {}",
            target_id,
            _format_exception(exc),
        )
        reflection = ""
    scored, _ = score_hypotheses(
        state,
        [target],
        model=model,
        lit_text=lit_text,
        feedback=_feedback_text(state),
        prune_low_novelty=False,
        tolerate_errors=True,
        log_label="review",
    )
    scored_target = scored[0]
    try:
        verification = _verify_hypothesis(target, lit_text, model)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "review | id={} verification failed, using neutral verdict: {}",
            target_id,
            _format_exception(exc),
        )
        verification = VerificationResult()

    verification_summary = (
        f"verdict={verification.verdict}: {verification.summary} "
        + "; ".join(
            f"[{'ok' if a.plausible else 'weak'}] {a.assumption}"
            for a in verification.assumptions
        )
    ).strip()

    updated: Hypothesis = {
        **scored_target,
        "reflection": reflection,
        "verification": verification_summary,
    }

    reviews = [
        {
            "type": "FullReview",
            "hypothesis_id": target_id,
            "review": reflection,
            "scores": {
                "novelty": scored_target["novelty_score"],
                "rigor": scored_target["rigor_score"],
                "testability": scored_target["testability_score"],
                "impact": scored_target["impact_score"],
                "simplicity": scored_target["simplicity_score"],
            },
        },
        {
            "type": "VerificationReview",
            "hypothesis_id": target_id,
            "verdict": verification.verdict,
            "summary": verification.summary,
            "assumptions": [
                {"assumption": a.assumption, "plausible": a.plausible, "note": a.note}
                for a in verification.assumptions
            ],
        },
    ]

    result: dict = {
        "hypotheses": [updated],
        "reviews": reviews,
        "reviewed_id": target_id,
    }
    if scored_target["novelty_score"] < NOVELTY_PRUNE_THRESHOLD:
        logger.info("review | pruning low-novelty [{}]", target_id)
        result["pruned_ids"] = list(set(state.get("pruned_ids") or []) | {target_id})
        # A pruned hypothesis does not proceed to the tournament.
        result.pop("reviewed_id")
    return result


def proximity_node(state: CoScientistState) -> dict:
    active = _active(state)
    logger.info("proximity | n={}", len(active))

    graph: dict[str, list[dict[str, Any]]] = {}
    elo_map = {h["id"]: h.get("elo", INITIAL_ELO_SENTINEL) for h in active}

    # Similarity backend: embeddings (cosine) when available, else lexical ratio.
    vectors = _embed_texts([_hypothesis_text(h) for h in active])
    if vectors is not None:
        logger.info("proximity | using embedding similarity")
        index = {h["id"]: i for i, h in enumerate(active)}

        def _sim(a: Hypothesis, b: Hypothesis) -> float:
            return _cosine(vectors[index[a["id"]]], vectors[index[b["id"]]])

    else:
        logger.info("proximity | using lexical (difflib) similarity")
        _sim = _similarity

    # Prune near-duplicates: for each pair above the duplicate threshold, drop
    # the lower-Elo hypothesis so the tournament doesn't waste matches on it.
    pruned: set[str] = set(state.get("pruned_ids") or [])

    for h1, h2 in combinations(active, 2):
        sim = _sim(h1, h2)
        if sim >= DUPLICATE_THRESHOLD:
            loser = h1["id"] if elo_map[h1["id"]] <= elo_map[h2["id"]] else h2["id"]
            if loser not in pruned:
                pruned.add(loser)
                logger.info(
                    "proximity | pruning near-duplicate [{}] sim={:.2f}", loser, sim
                )
        if sim > 0.3:
            graph.setdefault(h1["id"], []).append({"id": h2["id"], "similarity": sim})
            graph.setdefault(h2["id"], []).append({"id": h1["id"], "similarity": sim})

    logger.info(
        "proximity | edges={} pruned={}",
        sum(len(v) for v in graph.values()),
        len(pruned) - len(set(state.get("pruned_ids") or [])),
    )
    return {"proximity_graph": graph, "pruned_ids": list(pruned)}


def _decision_section(h: Hypothesis, decision_mode: bool) -> str:
    """Format the decision block for a hypothesis, or empty string."""
    if not decision_mode or not h.get("decision"):
        return ""
    return (
        f"  Decision: {h['decision']}\n"
        f"  For whom: {h.get('for_whom', '')}\n"
        f"  Reversal: {h.get('reversal_condition', '')}"
    )


def _not_doing_section(h: Hypothesis, _decision_mode: bool = False) -> str:
    """Format the not-doing block for a hypothesis, or empty string."""
    nd = h.get("not_doing", [])
    if nd:
        return "  Not doing:\n" + "\n".join(f"    - {n}" for n in nd)
    return ""


def _normalize_winner_letter(winner: str) -> str:
    """Normalize the winner letter to 'A' or 'B'. Defaults to 'A'."""
    if winner.upper() == "B":
        return "B"
    return "A"


def _select_pairs(
    active: list[Hypothesis],
    scores: dict[str, float],
    n_matches: int,
    proximity_graph: dict | None = None,
    match_history: list[dict] | None = None,
) -> list[tuple[Hypothesis, Hypothesis]]:
    """Select pairs for tournament matches using Elo proximity + recency scoring.

    Pairs are scored by:
    - Elo proximity: closer Elo ratings = more informative match (higher weight)
    - Freshness: pairs that have already played many times are penalized, so
      matchups rotate instead of replaying the same dominant pairing forever
      (which would inflate one Elo monotonically and starve evolution)
    - Recency: newer hypotheses (lower generation) get more matches

    Pairs with precomputed similarity >= SIMILARITY_SKIP_THRESHOLD are excluded
    as redundant (the proximity graph already tells us they're too similar to
    produce an informative match).

    Returns the top N pairs by combined score.
    """
    all_pairs = list(combinations(active, 2))

    # Drop redundant pairs: too similar to yield an informative match
    if proximity_graph:
        all_pairs = [
            (h1, h2)
            for h1, h2 in all_pairs
            if _pair_similarity(h1["id"], h2["id"], proximity_graph)
            < SIMILARITY_SKIP_THRESHOLD
        ]

    if len(all_pairs) <= n_matches:
        return all_pairs

    # Count how many times each unordered pair has already played, so we can
    # deprioritize repeats. Derived from the cumulative tournament log.
    pair_plays: Counter[frozenset[str]] = Counter()
    for m in match_history or []:
        winner_id, loser_id = m.get("winner"), m.get("loser")
        if winner_id and loser_id:
            pair_plays[frozenset((winner_id, loser_id))] += 1

    # Compute Elo range for normalization
    elo_values = [scores.get(h["id"], h["elo"]) for h in active]
    max_elo = max(elo_values)
    min_elo = min(elo_values)
    elo_range = max_elo - min_elo if max_elo != min_elo else 1.0

    # Compute generation range for normalization
    gens = [h.get("generation", 1) for h in active]
    max_gen = max(gens)
    min_gen = min(gens)
    gen_range = max_gen - min_gen if max_gen != min_gen else 1.0

    # Score each pair
    scored: list[tuple[float, tuple[Hypothesis, Hypothesis]]] = []
    for h1, h2 in all_pairs:
        elo1 = scores.get(h1["id"], h1["elo"])
        elo2 = scores.get(h2["id"], h2["elo"])
        gen1 = h1.get("generation", 1)
        gen2 = h2.get("generation", 1)

        # Elo proximity: closer ratings = higher score (0-1)
        elo_diff = abs(elo1 - elo2) / elo_range
        elo_score = 1.0 - elo_diff

        # Recency: newer hypotheses (lower generation) get higher score (0-1)
        max_gen_pair = max(gen1, gen2)
        recency_score = 1.0 - (max_gen_pair - min_gen) / gen_range

        # Freshness: pairs played fewer times score higher (1.0 if never played),
        # so the same matchup doesn't repeat every batch.
        plays = pair_plays[frozenset((h1["id"], h2["id"]))]
        freshness_score = 1.0 / (1.0 + plays)

        # Combined score: 50% Elo proximity, 30% freshness, 20% recency
        combined = 0.5 * elo_score + 0.3 * freshness_score + 0.2 * recency_score
        scored.append((combined, (h1, h2)))

    # Sort by combined score descending, take top N
    scored.sort(key=lambda x: x[0], reverse=True)
    return [pair for _, pair in scored[:n_matches]]


def _play_tournament_match(
    h1: Hypothesis,
    h2: Hypothesis,
    llm: Any,  # ChatOpenAI wrapped by .with_structured_output → Runnable
    scores: dict[str, float],
    question: str,
    decision_mode: bool,
    negative_space: bool,
    feedback: str,
) -> dict:
    prompt = COMPARE.format(
        goal=question,
        criteria="correctness, novelty, feasibility, and testability",
        feedback=feedback,
        mech_a=h1["mechanism"],
        pred_a=h1["prediction"],
        refl_a=h1.get("reflection", ""),
        nov_a=h1.get("novelty_score", 0.5),
        rig_a=h1.get("rigor_score", 0.5),
        test_a=h1.get("testability_score", 0.5),
        imp_a=h1.get("impact_score", 0.5),
        simp_a=h1.get("simplicity_score", 0.5),
        decision_section_a=_decision_section(h1, decision_mode),
        not_doing_section_a=_not_doing_section(h1),
        mech_b=h2["mechanism"],
        pred_b=h2["prediction"],
        refl_b=h2.get("reflection", ""),
        nov_b=h2.get("novelty_score", 0.5),
        rig_b=h2.get("rigor_score", 0.5),
        test_b=h2.get("testability_score", 0.5),
        imp_b=h2.get("impact_score", 0.5),
        simp_b=h2.get("simplicity_score", 0.5),
        decision_section_b=_decision_section(h2, decision_mode),
        not_doing_section_b=_not_doing_section(h2),
    )
    if decision_mode:
        prompt = prompt + "\n\nThis is a decision-framed comparison."
    if negative_space:
        prompt = prompt + NEGATIVE_SPACE_ADDON

    result = cast(
        CompareResult,
        _invoke_structured(llm, prompt, label=f"compare {h1['id']} vs {h2['id']}"),
    )

    winner_id = h1["id"] if result.winner == "A" else h2["id"]
    loser_id = h2["id"] if result.winner == "A" else h1["id"]
    winner_elo, loser_elo = _elo_update(
        scores[h1["id"]], scores[h2["id"]], winner_id, h1["id"]
    )
    scores[winner_id] = winner_elo
    scores[loser_id] = loser_elo

    logger.info(
        "  {} beat {} ({:.1f} vs {:.1f}): {}",
        winner_id,
        loser_id,
        winner_elo,
        loser_elo,
        result.reason,
    )

    return {
        "winner": winner_id,
        "loser": loser_id,
        "reason": result.reason,
    }


def add_to_tournament_node(state: CoScientistState, target_id: str) -> dict:
    """Ranking.AddToTournament(TargetID): seed Elo 1200, guarding re-entry.

    Mirrors the pseudocode: if the hypothesis already has an Elo (is already in
    the tournament) this is a no-op.
    """
    target = next((h for h in state["hypotheses"] if h["id"] == target_id), None)
    if target is None:
        logger.warning("add_to_tournament | unknown id {}", target_id)
        return {}
    if _in_tournament(target):
        logger.info(
            "add_to_tournament | [{}] already in tournament, skipping", target_id
        )
        return {}

    updated: Hypothesis = {
        **target,
        "elo": TOURNAMENT_START_ELO,
        "in_tournament": True,
        "matches": target.get("matches", 0),
    }
    logger.info(
        "add_to_tournament | [{}] seeded Elo {}", target_id, TOURNAMENT_START_ELO
    )
    return {"hypotheses": [updated], "added_id": target_id}


def tournament_node(state: CoScientistState, *, model: str = "gpt-4o-mini") -> dict:
    # Only hypotheses that have entered the tournament participate.  In the
    # linear (deprecated) graph path nothing calls AddToTournament, so we
    # auto-admit any active hypothesis that is not yet in the tournament.
    active = _active(state)
    auto_added: dict[str, Hypothesis] = {}
    for h in active:
        if not _in_tournament(h):
            auto_added[h["id"]] = {
                **h,
                "elo": TOURNAMENT_START_ELO,
                "in_tournament": True,
                "matches": h.get("matches", 0),
            }
    contenders = [
        auto_added.get(h["id"], h)
        for h in active
        if _in_tournament(auto_added.get(h["id"], h))
    ]

    if len(contenders) < 2:
        logger.warning("tournament | fewer than 2 contenders, skipping")
        return {"hypotheses": list(auto_added.values())} if auto_added else {}

    logger.info("tournament | model={} n={}", model, len(contenders))

    n_matches = min(len(contenders) * 2, 10)
    scores = {h["id"]: h.get("elo", TOURNAMENT_START_ELO) for h in contenders}
    match_counts = {h["id"]: h.get("matches", 0) for h in contenders}
    prox_graph = state.get("proximity_graph") or {}
    sampled = _select_pairs(
        contenders,
        scores,
        n_matches,
        proximity_graph=prox_graph,
        match_history=state.get("tournament_log") or [],
    )

    decision_mode = state.get("decision_mode", False)
    negative_space = state.get("negative_space", False)
    llm = _structured(_llm(model, temperature=0.0), CompareResult)
    new_log: list[dict] = []

    for h1, h2 in sampled:
        try:
            match_result = _play_tournament_match(
                h1=h1,
                h2=h2,
                llm=llm,
                scores=scores,
                question=state["question"],
                decision_mode=decision_mode,
                negative_space=negative_space,
                feedback=_feedback_text(state),
            )
        except Exception:  # noqa: BLE001 - skip a flaky match, keep the others
            logger.warning(
                "tournament | match {} vs {} failed after retries, skipping",
                h1["id"],
                h2["id"],
            )
            continue
        match_counts[h1["id"]] += 1
        match_counts[h2["id"]] += 1
        new_log.append(match_result)

    # Merge updated Elo + match counts back, including any auto-added hypotheses.
    by_id = {h["id"]: h for h in contenders}
    merged = [
        {
            **by_id[h_id],
            "elo": scores[h_id],
            "matches": match_counts[h_id],
        }
        for h_id in by_id
    ]
    return {"hypotheses": merged, "tournament_log": new_log}


def _select_diverse_parents(
    pool: list[Hypothesis],
    proximity_graph: dict,
) -> tuple[Hypothesis, Hypothesis]:
    """Pick the parent pair that best balances high Elo with low similarity.

    Scoring per pair: 0.5 * diversity + 0.5 * normalised_elo_average
    where diversity = 1 - precomputed_similarity (0.0 if pair not in graph).
    Falls back to the top-2 by Elo when the graph is empty.
    """
    if not proximity_graph or len(pool) < 2:
        return pool[0], pool[1]

    max_elo = max(h["elo"] for h in pool)
    best_score = -1.0
    best_pair: tuple[Hypothesis, Hypothesis] = (pool[0], pool[1])

    for h1, h2 in combinations(pool, 2):
        sim = _pair_similarity(h1["id"], h2["id"], proximity_graph)
        diversity = 1.0 - sim
        elo_score = (h1["elo"] + h2["elo"]) / (2 * max_elo) if max_elo > 0 else 0.5
        score = 0.5 * diversity + 0.5 * elo_score
        if score > best_score:
            best_score = score
            best_pair = (h1, h2)

    logger.info(
        "evolve | combine parents [{}, {}] (score={:.3f})",
        best_pair[0]["id"],
        best_pair[1]["id"],
        best_score,
    )
    return best_pair


def _evolve_child(
    result: EvolveResult,
    child_id: str,
    round_num: int,
    strategy: str,
) -> Hypothesis:
    """Build (and log) a Hypothesis dict from an EvolveResult."""
    child = _build_hypothesis(result, child_id, round_num)
    logger.info("  child [{}] strategy={} {}", child_id, strategy, child["mechanism"])
    for nd in child["not_doing"]:
        logger.info("    ✗ {}", nd)
    return child


def _evolve_one(
    llm: Any, prompt: str, child_id: str, round_num: int, strategy: str
) -> Hypothesis | None:
    """Run one evolution strategy, returning the child or None if it fails."""
    try:
        result = cast(
            EvolveResult, _invoke_structured(llm, prompt, label=f"evolve {strategy}")
        )
    except Exception:  # noqa: BLE001 - skip a flaky strategy, keep the others
        logger.warning("evolve | {} failed after retries, skipping", strategy)
        return None
    return _evolve_child(result, child_id, round_num, strategy)


def _next_evolved_id(state: CoScientistState, strategy: str) -> str:
    """Return a stable, collision-free evolved hypothesis id for a strategy."""
    existing = {h["id"] for h in state["hypotheses"]}
    idx = state["round"]
    while f"e{idx}_{strategy}" in existing:
        idx += 1
    return f"e{idx}_{strategy}"


def _with_negative_space(prompt: str, enabled: bool) -> str:
    return prompt + NEGATIVE_SPACE_ADDON if enabled else prompt


def _combine_prompt(
    p1: Hypothesis,
    p2: Hypothesis,
    *,
    decision_mode: bool,
    feedback: str,
    notes: str,
) -> str:
    if decision_mode and p1.get("decision"):
        return EVOLVE_DECISION.format(
            mech_1=p1["mechanism"],
            dec_1=p1["decision"],
            whom_1=p1.get("for_whom", ""),
            rev_1=p1.get("reversal_condition", ""),
            pred_1=p1["prediction"],
            mech_2=p2["mechanism"],
            dec_2=p2.get("decision", ""),
            whom_2=p2.get("for_whom", ""),
            rev_2=p2.get("reversal_condition", ""),
            pred_2=p2["prediction"],
            feedback=feedback,
            notes=notes,
        )
    return EVOLVE.format(
        mech_1=p1["mechanism"],
        pred_1=p1["prediction"],
        mech_2=p2["mechanism"],
        pred_2=p2["prediction"],
        feedback=feedback,
        notes=notes,
    )


def _simplify_prompt(
    top: Hypothesis,
    *,
    decision_mode: bool,
    feedback: str,
) -> str:
    if decision_mode and top.get("decision"):
        return EVOLVE_SIMPLIFY_DECISION.format(
            mech=top["mechanism"],
            pred=top["prediction"],
            dec=top["decision"],
            whom=top.get("for_whom", ""),
            rev=top.get("reversal_condition", ""),
            exp=top.get("experiment", ""),
            feedback=feedback,
        )
    return EVOLVE_SIMPLIFY.format(
        mech=top["mechanism"],
        pred=top["prediction"],
        exp=top.get("experiment", ""),
        feedback=feedback,
    )


def _diverge_prompt(
    top5: list[Hypothesis],
    *,
    decision_mode: bool,
    feedback: str,
    notes: str,
) -> str:
    if decision_mode:
        hyp_list = "\n".join(
            f"{i+1}. [{h['id']}] {h['mechanism']} | Decision: {h.get('decision', '')}"
            for i, h in enumerate(top5)
        )
        return EVOLVE_DIVERGE_DECISION.format(
            hypotheses=hyp_list,
            feedback=feedback,
            notes=notes,
        )
    hyp_list = "\n".join(
        f"{i+1}. [{h['id']}] {h['mechanism']}" for i, h in enumerate(top5)
    )
    return EVOLVE_DIVERGE.format(
        hypotheses=hyp_list,
        feedback=feedback,
        notes=notes,
    )


def evolve_node(state: CoScientistState, *, model: str = "gpt-4o-mini") -> dict:
    """Run three evolution strategies on the top 5 hypotheses.

    Strategies:
    - combine:  merge insights from the top 2 parents
    - simplify: refine the best hypothesis for clarity and testability
    - diverge:  propose a fundamentally different approach inspired by the top pool
    """
    active = _active(state)
    if len(active) < 2:
        logger.warning("evolve | fewer than 2 active hypotheses, skipping")
        return {}

    top5 = active[:5]
    round_num = state["round"]
    decision_mode = state.get("decision_mode", False)
    negative_space = state.get("negative_space", False)
    feedback = _feedback_text(state)
    notes = _format_matches(
        state["tournament_log"], limit=10, prefix="- ", empty="No matches yet."
    )

    logger.info(
        "evolve | model={} top={} strategies=combine,simplify,diverge",
        model,
        len(top5),
    )

    prox_graph = state.get("proximity_graph") or {}
    llm = _structured(_llm(model), EvolveResult)
    children: list[Hypothesis] = []

    # ── Strategy 1: combine ───────────────────────────────────────────────────
    # Prefer dissimilar top hypotheses to maximise diversity in the child.
    p1, p2 = _select_diverse_parents(top5, prox_graph)
    prompt = _with_negative_space(
        _combine_prompt(
            p1,
            p2,
            decision_mode=decision_mode,
            feedback=feedback,
            notes=notes,
        ),
        negative_space,
    )
    child = _evolve_one(
        llm, prompt, _next_evolved_id(state, "combine"), round_num, "combine"
    )
    if child is not None:
        children.append(child)

    # ── Strategy 2: simplify ──────────────────────────────────────────────────
    top = top5[0]
    prompt = _with_negative_space(
        _simplify_prompt(top, decision_mode=decision_mode, feedback=feedback),
        negative_space,
    )
    child = _evolve_one(
        llm, prompt, _next_evolved_id(state, "simplify"), round_num, "simplify"
    )
    if child is not None:
        children.append(child)

    # ── Strategy 3: diverge ───────────────────────────────────────────────────
    prompt = _with_negative_space(
        _diverge_prompt(
            top5,
            decision_mode=decision_mode,
            feedback=feedback,
            notes=notes,
        ),
        negative_space,
    )
    child = _evolve_one(
        llm, prompt, _next_evolved_id(state, "diverge"), round_num, "diverge"
    )
    if child is not None:
        children.append(child)

    return {
        "hypotheses": children,
        "new_hypothesis_ids": [c["id"] for c in children],
    }


def meta_review_node(state: CoScientistState, *, model: str = "gpt-4o-mini") -> dict:
    active = _active(state)
    logger.info(
        "meta_review | round={} model={} active={}", state["round"], model, len(active)
    )

    top_text = _format_ranked(active, 5)
    tourney_text = _format_matches(state["tournament_log"], limit=10)

    llm = _llm(model, temperature=0.3)
    review_message = llm.invoke(
        META_REVIEW.format(
            round=state["round"],
            top_hypotheses=top_text,
            tournament_summary=tourney_text,
        )
    )
    review = _message_text(review_message)
    review = _compact_for_state(
        review,
        label="meta-review",
        model=model,
        max_chars=META_REVIEW_SUMMARY_CHARS,
        guidance=(
            "Preserve the emerging patterns, strongest directions, recurring weaknesses, "
            "and gaps that should influence the next round."
        ),
    )
    logger.info("meta_review | {:.120}", review)

    from .display import print_hypotheses

    print_hypotheses(active, header=f"Round {state['round']} standings")
    import click

    click.echo(f"\n  {review}\n")

    return {"meta_review": review}


def system_feedback_node(
    state: CoScientistState, *, model: str = "gpt-4o-mini"
) -> dict:
    """Generate system-wide feedback by aggregating all rounds' meta-reviews
    and tournament transcripts.

    This is a leaf task: it produces feedback but does not spawn follow-up
    tasks.  The Supervisor's decide_next_steps() periodically enqueues this
    task when the queue has been empty for N iterations.
    """
    logger.info("system_feedback | round={}", state["round"])

    # Aggregate tournament transcripts (all rounds)
    tournament_transcripts = _format_matches(
        state.get("tournament_log", []), empty="  No tournament matches yet."
    )

    # Top hypotheses by Elo
    active = _active(state)
    top_hypotheses = _format_ranked(active, 10)

    # Use current round's meta_review as the round-level summary
    meta_reviews = state.get("meta_review") or ""

    llm = _llm(model, temperature=0.3)
    feedback_message = llm.invoke(
        SYSTEM_FEEDBACK.format(
            meta_reviews=meta_reviews,
            tournament_transcripts=tournament_transcripts,
            top_hypotheses=top_hypotheses,
        )
    )
    feedback = _message_text(feedback_message)
    feedback = _compact_for_state(
        feedback,
        label="system feedback",
        model=model,
        max_chars=SYSTEM_FEEDBACK_SUMMARY_CHARS,
        guidance=(
            "Preserve only actionable guidance that should steer future generation, "
            "review, and tournament decisions."
        ),
    )
    logger.info("system_feedback | {:.120}", feedback)

    return {"system_wide_feedback": feedback}


def final_overview_node(state: CoScientistState, *, model: str = "gpt-4o-mini") -> dict:
    """Metareview.GenerateFinalResearchOverview: synthesize the top 10 hypotheses.

    Produces the single coherent research overview returned to the scientist.
    """
    active = _active(state)
    logger.info("final_overview | model={} active={}", model, len(active))

    top_text = _format_ranked(active, 10)

    plan = state.get("research_plan") or {}
    objective = plan.get("objective") or state["question"]

    llm = _llm(model, temperature=0.3)
    overview_message = llm.invoke(
        FINAL_OVERVIEW.format(
            objective=objective,
            top_hypotheses=top_text,
            system_feedback=state.get("system_wide_feedback") or "None.",
        )
    )
    overview = _message_text(overview_message)
    logger.info("final_overview | {:.120}", overview)

    return {"final_overview": overview}


def should_continue(state: CoScientistState) -> str:
    if state["round"] >= state["max_rounds"]:
        logger.info("converged | max rounds ({}) reached", state["max_rounds"])
        return "end"

    logger.info("continue | starting round {}", state["round"] + 1)
    return "generate"
