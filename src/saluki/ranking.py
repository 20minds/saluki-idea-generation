from __future__ import annotations

from typing import cast

from .nodes import (
    INITIAL_ELO_SENTINEL,
    TOURNAMENT_START_ELO,
    bootstrap_node,
    idea_research_node,
    review_hypothesis_node,
    score_hypotheses,
    tournament_node,
)
from .state import CoScientistState, Hypothesis, _upsert_hypotheses, new_state

DEFAULT_RANK_PROMPT = "Rank these predefined ideas by overall quality."


def parse_markdown_ideas(markdown: str) -> list[str]:
    """Extract blank-line-separated Markdown paragraphs as ideas."""
    ideas: list[str] = []
    current: list[str] = []
    in_fence = False

    def flush() -> None:
        if current:
            ideas.append(" ".join(current).strip())
            current.clear()

    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            flush()
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if not stripped:
            flush()
            continue
        if stripped.startswith("#") or stripped in {"---", "***", "___"}:
            flush()
            continue
        current.append(stripped)

    flush()
    return ideas


def ideas_to_hypotheses(ideas: list[str]) -> list[Hypothesis]:
    """Represent predefined ideas in the existing tournament hypothesis shape."""
    return [
        {
            "id": f"idea_{index:03d}",
            "mechanism": idea,
            "prediction": "",
            "experiment": "",
            "elo": INITIAL_ELO_SENTINEL,
            "reflection": "",
            "verification": "",
            "novelty_score": 0.5,
            "rigor_score": 0.5,
            "testability_score": 0.5,
            "impact_score": 0.5,
            "simplicity_score": 0.5,
            "generation": 0,
            "decision": "",
            "for_whom": "",
            "reversal_condition": "",
            "not_doing": [],
            "in_tournament": False,
            "matches": 0,
        }
        for index, idea in enumerate(ideas, 1)
    ]


def rank_predefined_ideas(
    ideas: list[str],
    *,
    prompt: str = DEFAULT_RANK_PROMPT,
    model: str = "gpt-4o-mini",
    max_matches_per_idea: int = 4,
    research: bool = False,
    deep_review: bool = False,
) -> CoScientistState:
    """Run the existing Elo tournament until every idea reaches the match target."""
    state = new_state(prompt, max_rounds=1)
    state["hypotheses"] = ideas_to_hypotheses(ideas)
    if research:
        state = research_predefined_ideas(state, model=model)
    state = (
        deep_review_predefined_ideas(state, model=model)
        if deep_review
        else score_predefined_ideas(state, model=model)
    )

    max_batches = max(1, len(ideas) * max_matches_per_idea * 2)
    for _ in range(max_batches):
        result = tournament_node(state, model=model)
        state = merge_tournament_result(state, result)
        if _rank_complete(state, max_matches_per_idea):
            break
        if not result.get("tournament_log") and len(ideas) >= 2:
            break

    return state


def research_predefined_ideas(
    state: CoScientistState, *, model: str = "gpt-4o-mini"
) -> CoScientistState:
    """Reuse the generate setup steps to add ranking context before scoring."""
    state = merge_node_result(
        state,
        bootstrap_node(state, model=model),
        preserve_hypotheses=True,
        preserve_pruned_ids=True,
    )
    return merge_node_result(
        state,
        idea_research_node(state, model=model),
        preserve_pruned_ids=True,
    )


def score_predefined_ideas(
    state: CoScientistState, *, model: str = "gpt-4o-mini"
) -> CoScientistState:
    """Assign five-dimension scores while preserving the fixed idea set."""
    scored, _ = score_hypotheses(
        state,
        state["hypotheses"],
        model=model,
        prune_low_novelty=False,
        tolerate_errors=True,
        log_label="rank",
    )
    return {**state, "hypotheses": scored}


def deep_review_predefined_ideas(
    state: CoScientistState, *, model: str = "gpt-4o-mini"
) -> CoScientistState:
    """Reuse per-idea review while keeping every predefined idea rankable."""
    reviewed = state
    for hypothesis in list(state["hypotheses"]):
        reviewed = merge_node_result(
            reviewed,
            review_hypothesis_node(reviewed, hypothesis["id"], model=model),
            preserve_pruned_ids=True,
        )
    return reviewed


def merge_node_result(
    state: CoScientistState,
    result: dict,
    *,
    preserve_hypotheses: bool = False,
    preserve_pruned_ids: bool = False,
) -> CoScientistState:
    """Merge a node result into state, using upsert semantics for hypotheses."""
    merged: CoScientistState = {**state}
    for key, value in result.items():
        if key == "hypotheses":
            if preserve_hypotheses:
                continue
            merged["hypotheses"] = _upsert_hypotheses(merged["hypotheses"], value)
        elif key == "tournament_log":
            merged["tournament_log"] = [
                *merged.get("tournament_log", []),
                *value,
            ]
        elif key == "reviews":
            merged["reviews"] = [
                *merged.get("reviews", []),
                *value,
            ]
        elif key == "pruned_ids":
            if not preserve_pruned_ids:
                merged["pruned_ids"] = value
        else:
            cast(dict[str, object], merged)[key] = value
    return merged


def merge_tournament_result(state: CoScientistState, result: dict) -> CoScientistState:
    """Merge a tournament batch result into a state dict."""
    return merge_node_result(state, result)


def ranked_hypotheses(state: CoScientistState) -> list[Hypothesis]:
    pruned = set(state.get("pruned_ids") or [])
    active = [h for h in state["hypotheses"] if h["id"] not in pruned]
    return sorted(
        active,
        key=lambda h: h.get("elo", TOURNAMENT_START_ELO),
        reverse=True,
    )


def _rank_complete(state: CoScientistState, max_matches_per_idea: int) -> bool:
    contenders = [h for h in state["hypotheses"] if h.get("in_tournament", False)]
    return bool(contenders) and all(
        h.get("matches", 0) >= max_matches_per_idea for h in contenders
    )
