from __future__ import annotations

from typing import Any, cast
from types import SimpleNamespace

import pytest

from saluki.state import CoScientistState, Hypothesis


@pytest.fixture(autouse=True)
def _offline_defaults(monkeypatch):
    """Keep tests deterministic and offline by default.

    - Force the lexical (difflib) similarity path (no embeddings).
    - Make web search return nothing.
    Tests that specifically exercise these paths re-patch them.
    """
    import saluki.nodes as nodes_mod

    monkeypatch.setattr(nodes_mod, "_embed_texts", lambda texts: None)
    monkeypatch.setattr(nodes_mod, "_exa_api_key", "")
    monkeypatch.setattr(
        nodes_mod, "tivaly_search", lambda q, **k: {"results": [], "query": q}
    )


class DummyLLM:
    def __init__(self, result: object):
        self.result = result
        self.prompts: list[str] = []

    def with_structured_output(self, _schema: object) -> "DummyLLM":
        return self

    def invoke(self, prompt: str) -> object:
        self.prompts.append(prompt)
        return self.result


def ns(**kwargs: object) -> SimpleNamespace:
    return SimpleNamespace(**kwargs)


def _hypothesis(
    h_id: str,
    *,
    generation: int = 1,
    elo: float = 1200.0,
    in_tournament: bool = True,
    matches: int = 0,
) -> Hypothesis:
    data: dict[str, Any] = {
        "id": h_id,
        "mechanism": f"mechanism-{h_id}",
        "prediction": f"prediction-{h_id}",
        "experiment": f"experiment-{h_id}",
        "elo": elo,
        "in_tournament": in_tournament,
        "matches": matches,
        "reflection": "",
        "verification": "",
        "novelty_score": 0.5,
        "rigor_score": 0.5,
        "testability_score": 0.5,
        "impact_score": 0.5,
        "simplicity_score": 0.5,
        "generation": generation,
        "decision": "",
        "for_whom": "",
        "reversal_condition": "",
        "not_doing": [],
    }
    return cast(Hypothesis, data)


def _state(**overrides: Any) -> CoScientistState:
    data: dict[str, Any] = {
        "question": "",
        "research_plan": {},
        "hypotheses": [],
        "idea_research": "",
        "literature": [],
        "round": 0,
        "max_rounds": 1,
        "tournament_log": [],
        "reviews": [],
        "meta_review": "",
        "system_wide_feedback": "",
        "final_overview": "",
        "proximity_graph": {},
        "pruned_ids": [],
        "decision_mode": False,
        "negative_space": False,
        "deep_reflect": False,
        "verbose": False,
    }
    data.update(overrides)
    return cast(CoScientistState, data)
