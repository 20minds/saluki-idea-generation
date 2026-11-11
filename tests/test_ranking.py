from __future__ import annotations

from saluki import ranking


def test_parse_markdown_ideas_extracts_paragraphs() -> None:
    markdown = """# Ideas

First idea wraps
onto another line.

Second idea.

```
not an idea
```

---

Third idea.
"""

    assert ranking.parse_markdown_ideas(markdown) == [
        "First idea wraps onto another line.",
        "Second idea.",
        "Third idea.",
    ]


def test_rank_predefined_ideas_reuses_tournament_node(monkeypatch) -> None:
    calls: list[str] = []
    captured_score_kwargs: list[dict[str, object]] = []

    def fake_score_hypotheses(
        state: dict,
        hypotheses: list[dict],
        **kwargs: object,
    ) -> tuple[list[dict], list[str]]:
        captured_score_kwargs.append(kwargs)
        return [
            {
                **hypothesis,
                "novelty_score": 0.8,
                "rigor_score": 0.7,
                "testability_score": 0.6,
                "impact_score": 0.5,
                "simplicity_score": 0.4,
            }
            for hypothesis in hypotheses
        ], []

    def fake_tournament_node(state: dict, *, model: str) -> dict:
        calls.append(model)
        updates = []
        for index, hypothesis in enumerate(state["hypotheses"]):
            updates.append(
                {
                    **hypothesis,
                    "elo": 1200.0 + index,
                    "in_tournament": True,
                    "matches": hypothesis.get("matches", 0) + 1,
                }
            )
        return {
            "hypotheses": updates,
            "tournament_log": [
                {"winner": updates[0]["id"], "loser": updates[1]["id"], "reason": "x"}
            ],
        }

    monkeypatch.setattr(ranking, "score_hypotheses", fake_score_hypotheses)
    monkeypatch.setattr(ranking, "tournament_node", fake_tournament_node)

    state = ranking.rank_predefined_ideas(
        ["alpha", "beta"],
        prompt="rank",
        model="m",
        max_matches_per_idea=2,
        research=False,
        deep_review=False,
    )

    assert calls == ["m", "m"]
    assert captured_score_kwargs[0]["model"] == "m"
    assert [h["mechanism"] for h in state["hypotheses"]] == ["alpha", "beta"]
    assert [h["novelty_score"] for h in state["hypotheses"]] == [0.8, 0.8]
    assert [h["matches"] for h in state["hypotheses"]] == [2, 2]
    assert len(state["tournament_log"]) == 2


def test_score_predefined_ideas_uses_shared_scoring_without_pruning(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_score_hypotheses(
        state: dict,
        hypotheses: list[dict],
        **kwargs: object,
    ) -> tuple[list[dict], list[str]]:
        captured.update(kwargs)
        return [{**hypothesis, "novelty_score": 0.1} for hypothesis in hypotheses], [
            "ignored"
        ]

    monkeypatch.setattr(ranking, "score_hypotheses", fake_score_hypotheses)
    state = ranking.new_state("rank", max_rounds=1)
    state["hypotheses"] = ranking.ideas_to_hypotheses(["alpha", "beta"])

    scored = ranking.score_predefined_ideas(state, model="m")

    assert captured["model"] == "m"
    assert captured["prune_low_novelty"] is False
    assert captured["tolerate_errors"] is True
    assert len(scored["hypotheses"]) == 2
    assert scored["hypotheses"][0]["novelty_score"] == 0.1


def test_research_predefined_ideas_reuses_setup_without_dropping_ideas(
    monkeypatch,
) -> None:
    calls: list[str] = []

    def fake_bootstrap_node(state: dict, *, model: str) -> dict:
        calls.append(f"bootstrap:{model}")
        return {
            "research_plan": {"keywords": ["alpha"]},
            "hypotheses": [],
            "pruned_ids": ["would_drop"],
        }

    def fake_idea_research_node(state: dict, *, model: str) -> dict:
        calls.append(f"idea_research:{model}")
        assert state["research_plan"] == {"keywords": ["alpha"]}
        return {"idea_research": "brief"}

    monkeypatch.setattr(ranking, "bootstrap_node", fake_bootstrap_node)
    monkeypatch.setattr(ranking, "idea_research_node", fake_idea_research_node)
    state = ranking.new_state("rank", max_rounds=1)
    state["hypotheses"] = ranking.ideas_to_hypotheses(["alpha", "beta"])

    researched = ranking.research_predefined_ideas(state, model="m")

    assert calls == ["bootstrap:m", "idea_research:m"]
    assert [h["mechanism"] for h in researched["hypotheses"]] == ["alpha", "beta"]
    assert researched["idea_research"] == "brief"
    assert researched["pruned_ids"] == []


def test_deep_review_predefined_ideas_preserves_fixed_set(monkeypatch) -> None:
    reviewed_ids: list[str] = []

    def fake_review_hypothesis_node(state: dict, target_id: str, *, model: str) -> dict:
        reviewed_ids.append(target_id)
        target = next(h for h in state["hypotheses"] if h["id"] == target_id)
        return {
            "hypotheses": [{**target, "reflection": f"reviewed {target_id}"}],
            "reviews": [{"hypothesis_id": target_id}],
            "pruned_ids": [target_id],
        }

    monkeypatch.setattr(ranking, "review_hypothesis_node", fake_review_hypothesis_node)
    state = ranking.new_state("rank", max_rounds=1)
    state["hypotheses"] = ranking.ideas_to_hypotheses(["alpha", "beta"])

    reviewed = ranking.deep_review_predefined_ideas(state, model="m")

    assert reviewed_ids == ["idea_001", "idea_002"]
    assert [h["reflection"] for h in reviewed["hypotheses"]] == [
        "reviewed idea_001",
        "reviewed idea_002",
    ]
    assert len(reviewed["reviews"]) == 2
    assert reviewed["pruned_ids"] == []


def test_ranked_hypotheses_sorts_by_elo() -> None:
    state = ranking.new_state("rank", max_rounds=1)
    state["hypotheses"] = ranking.ideas_to_hypotheses(["alpha", "beta"])
    state["hypotheses"][0]["elo"] = 1100.0
    state["hypotheses"][1]["elo"] = 1300.0

    assert [h["id"] for h in ranking.ranked_hypotheses(state)] == [
        "idea_002",
        "idea_001",
    ]
