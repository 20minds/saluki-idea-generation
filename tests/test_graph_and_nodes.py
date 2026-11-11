from __future__ import annotations

from types import SimpleNamespace

import pytest

import saluki.graph as graph_mod
import saluki.nodes as nodes
from conftest import DummyLLM, ns, _hypothesis, _state
from saluki.nodes import (
    _active,
    _decision_section,
    _feedback_text,
    _not_doing_section,
    _normalize_winner_letter,
    _pair_similarity,
    _play_tournament_match,
    _select_diverse_parents,
    _select_pairs,
    bootstrap_node,
    evolve_node,
    generate_node,
    idea_research_node,
    meta_review_node,
    proximity_node,
    reflect_node,
    score_node,
    should_continue,
    tournament_node,
)
from saluki.state import Hypothesis


def test_bootstrap_and_feedback_helpers(monkeypatch) -> None:
    monkeypatch.setattr(
        nodes, "parse_research_plan", lambda *a, **k: {"objective": "q"}
    )
    state = _state(question="q", meta_review="  review  ", idea_research="x")
    boot = bootstrap_node(state)

    assert boot["round"] == 0
    assert boot["hypotheses"] == []
    assert boot["research_plan"] == {"objective": "q"}
    assert _feedback_text(state) == "review"


def test_active_and_contextual_helpers() -> None:
    state = _state(hypotheses=[_hypothesis("a"), _hypothesis("b")], pruned_ids=["a"])
    active = _active(state)

    assert [h["id"] for h in active] == ["b"]
    assert _decision_section(state["hypotheses"][0], True) == ""
    assert _not_doing_section(state["hypotheses"][0], True) == ""
    assert _normalize_winner_letter("b") == "B"
    assert _normalize_winner_letter("nope") == "A"


def test_generate_node_builds_hypotheses(monkeypatch) -> None:
    result = ns(
        id="x1",
        mechanism="m",
        prediction="p",
        experiment="e",
        decision="",
        for_whom="",
        reversal_condition="",
        not_doing=[],
    )
    fake_llm = DummyLLM(result)
    monkeypatch.setattr(nodes, "_llm", lambda *args, **kwargs: fake_llm)

    out = generate_node(
        _state(question="q", round=1, hypotheses=[], literature=[], meta_review=""),
        model="m",
    )

    assert out["round"] == 1
    assert out["hypotheses"][0]["mechanism"] == "m"
    assert "q" in fake_llm.prompts[0]


def test_idea_and_reflect_nodes(monkeypatch) -> None:
    monkeypatch.setattr(nodes, "run_idea_research", lambda **kwargs: "brief")
    state = _state(question="q", hypotheses=[], meta_review="", idea_research="")
    out = idea_research_node(state, model="m")
    assert out == {"idea_research": "brief"}

    state = _state(
        question="q",
        round=1,
        hypotheses=[_hypothesis("a")],
        meta_review="",
        deep_reflect=False,
    )
    fake_llm = DummyLLM(SimpleNamespace(content=" reflection "))
    monkeypatch.setattr(nodes, "_llm", lambda *args, **kwargs: fake_llm)
    out = reflect_node(state, model="m")
    assert out["hypotheses"][0]["reflection"] == "reflection"


def test_idea_research_node_summarizes_long_brief_before_storing(monkeypatch) -> None:
    long_brief = "Long brief sentence. " * 200
    fake_llm = DummyLLM(SimpleNamespace(content="compact research summary"))
    monkeypatch.setattr(nodes, "run_idea_research", lambda **kwargs: long_brief)
    monkeypatch.setattr(nodes, "_llm", lambda *args, **kwargs: fake_llm)
    monkeypatch.setattr(nodes, "IDEA_RESEARCH_SUMMARY_CHARS", 80)

    out = idea_research_node(
        _state(question="q", hypotheses=[], meta_review="", idea_research=""),
        model="m",
    )

    assert out == {"idea_research": "compact research summary"}
    assert "Long brief sentence" in fake_llm.prompts[0]


def test_compaction_falls_back_to_truncation_when_summary_fails(monkeypatch) -> None:
    class FailingLLM:
        def invoke(self, _prompt: str) -> object:
            raise RuntimeError("no model")

    monkeypatch.setattr(nodes, "_llm", lambda *args, **kwargs: FailingLLM())

    out = nodes._compact_for_state(
        "alpha beta gamma delta",
        label="test text",
        model="m",
        max_chars=12,
        guidance="Keep the important parts.",
    )

    assert out == "alpha beta..."


def test_score_and_proximity_nodes(monkeypatch) -> None:
    score_result = ns(
        novelty_score=0.1,
        rigor_score=0.1,
        testability_score=0.1,
        impact_score=0.1,
        simplicity_score=0.1,
        reason="low",
    )
    monkeypatch.setattr(nodes, "_llm", lambda *args, **kwargs: DummyLLM(score_result))

    state = _state(
        round=1,
        hypotheses=[_hypothesis("a", generation=1)],
        meta_review="",
        pruned_ids=[],
    )
    out = score_node(state, model="m")
    assert out["pruned_ids"] == ["a"]
    assert out["hypotheses"][0]["novelty_score"] == 0.1

    prox = proximity_node(
        _state(hypotheses=[_hypothesis("a"), _hypothesis("b")], pruned_ids=[])
    )
    assert "a" in prox["proximity_graph"]
    assert prox["proximity_graph"]["a"][0]["id"] == "b"


def test_score_hypotheses_logs_each_scored_idea(monkeypatch) -> None:
    score_result = ns(
        novelty_score=0.9,
        rigor_score=0.8,
        testability_score=0.7,
        impact_score=0.6,
        simplicity_score=0.5,
        reason="ok",
    )
    logged: list[str] = []

    monkeypatch.setattr(nodes, "_llm", lambda *args, **kwargs: DummyLLM(score_result))
    monkeypatch.setattr(
        nodes.logger,
        "info",
        lambda msg, *args: logged.append(msg.format(*args)),
    )

    nodes.score_hypotheses(
        _state(question="q"),
        [_hypothesis("a")],
        model="m",
        log_label="rank",
    )

    assert logged == [
        "rank | id=a scored novelty=0.90 rigor=0.80 testability=0.70 impact=0.60 simplicity=0.50"
    ]


def test_tournament_helpers_and_node(monkeypatch) -> None:
    a = _hypothesis("a")
    b = _hypothesis("b")
    active: list[Hypothesis] = [a, b]

    compare_result = ns(winner="B", reason="better")
    monkeypatch.setattr(nodes, "_llm", lambda *args, **kwargs: DummyLLM(compare_result))
    out = tournament_node(
        _state(
            question="q",
            hypotheses=active,
            proximity_graph={},
            decision_mode=False,
            negative_space=False,
            tournament_log=[],
        ),
        model="m",
    )
    assert out["tournament_log"][0]["winner"] == "b"

    match = _play_tournament_match(
        h1=a,
        h2=b,
        llm=DummyLLM(compare_result),
        scores={"a": 1200.0, "b": 1200.0},
        question="q",
        decision_mode=False,
        negative_space=False,
        feedback="f",
    )
    assert match["winner"] == "b"


def test_evolve_and_meta_review_and_continue(monkeypatch) -> None:
    evolve_result = ns(
        mechanism="new mech",
        prediction="new pred",
        experiment="new exp",
        decision="",
        for_whom="",
        reversal_condition="",
        not_doing=[],
    )
    monkeypatch.setattr(nodes, "_llm", lambda *args, **kwargs: DummyLLM(evolve_result))
    state = _state(
        round=2,
        hypotheses=[_hypothesis("a", elo=1300.0), _hypothesis("b", elo=1250.0)],
        tournament_log=[{"winner": "a", "loser": "b", "reason": "x"}],
        meta_review="",
        decision_mode=False,
        negative_space=False,
    )
    out = evolve_node(state, model="m")
    # Three strategies → three children
    assert len(out["hypotheses"]) == 3
    assert out["hypotheses"][0]["mechanism"] == "new mech"

    monkeypatch.setattr(
        nodes,
        "_llm",
        lambda *args, **kwargs: DummyLLM(SimpleNamespace(content="review text")),
    )
    review = meta_review_node(
        _state(
            round=2, hypotheses=[_hypothesis("a")], tournament_log=[], meta_review=""
        ),
        model="m",
    )
    assert review["meta_review"] == "review text"
    assert should_continue(_state(round=1, max_rounds=2)) == "generate"
    assert should_continue(_state(round=2, max_rounds=2)) == "end"


def test_evolve_returns_three_strategies(monkeypatch) -> None:
    """evolve_node always produces exactly 3 children: combine, simplify, diverge."""
    evolve_result = ns(
        mechanism="child mech",
        prediction="child pred",
        experiment="child exp",
        decision="",
        for_whom="",
        reversal_condition="",
        not_doing=[],
    )
    monkeypatch.setattr(nodes, "_llm", lambda *args, **kwargs: DummyLLM(evolve_result))
    state = _state(
        round=3,
        hypotheses=[
            _hypothesis("a", elo=1400.0),
            _hypothesis("b", elo=1350.0),
            _hypothesis("c", elo=1300.0),
        ],
        tournament_log=[],
        decision_mode=False,
        negative_space=False,
    )
    out = evolve_node(state, model="m")
    assert len(out["hypotheses"]) == 3
    ids = [h["id"] for h in out["hypotheses"]]
    assert "e3_combine" in ids
    assert "e3_simplify" in ids
    assert "e3_diverge" in ids
    for h in out["hypotheses"]:
        assert h["generation"] == 3
        assert h["elo"] == 0.0  # not yet in tournament
        assert h["in_tournament"] is False


def test_evolve_avoids_existing_child_ids(monkeypatch) -> None:
    """Repeated task-queue evolution must append new child ids, not overwrite e0_*."""
    evolve_result = ns(
        mechanism="child mech",
        prediction="child pred",
        experiment="child exp",
        decision="",
        for_whom="",
        reversal_condition="",
        not_doing=[],
    )
    monkeypatch.setattr(nodes, "_llm", lambda *args, **kwargs: DummyLLM(evolve_result))
    state = _state(
        round=0,
        hypotheses=[
            _hypothesis("h0_2", elo=1300.0),
            _hypothesis("h0_4", elo=1250.0),
            _hypothesis("e0_combine", elo=1200.0),
            _hypothesis("e0_simplify", elo=1200.0),
            _hypothesis("e0_diverge", elo=1200.0),
        ],
        tournament_log=[],
        decision_mode=False,
        negative_space=False,
    )
    out = evolve_node(state, model="m")
    assert [h["id"] for h in out["hypotheses"]] == [
        "e1_combine",
        "e1_simplify",
        "e1_diverge",
    ]


def test_evolve_uses_top_5_max(monkeypatch) -> None:
    """evolve_node uses at most 5 hypotheses even when more are active."""
    prompts_seen: list[str] = []

    class CapturingLLM:
        def with_structured_output(self, _schema):
            return self

        def invoke(self, prompt: str):
            prompts_seen.append(prompt)
            return ns(
                mechanism="m",
                prediction="p",
                experiment="e",
                decision="",
                for_whom="",
                reversal_condition="",
                not_doing=[],
            )

    monkeypatch.setattr(nodes, "_llm", lambda *args, **kwargs: CapturingLLM())
    hypotheses = [_hypothesis(f"h{i}", elo=float(1400 - i * 10)) for i in range(7)]
    state = _state(
        round=1,
        hypotheses=hypotheses,
        tournament_log=[],
        decision_mode=False,
        negative_space=False,
    )
    out = evolve_node(state, model="m")
    assert len(out["hypotheses"]) == 3
    # The diverge prompt should mention at most 5 hypothesis IDs
    diverge_prompt = prompts_seen[2]
    assert "h5" not in diverge_prompt
    assert "h6" not in diverge_prompt


def test_evolve_skips_when_fewer_than_two(monkeypatch) -> None:
    state = _state(
        round=1,
        hypotheses=[_hypothesis("only")],
        tournament_log=[],
    )
    out = evolve_node(state, model="m")
    assert out == {}


def test_evolve_decision_mode(monkeypatch) -> None:
    """In decision mode with decision fields set, EVOLVE_DECISION prompt is used."""
    prompts_seen: list[str] = []

    class CapturingLLM:
        def with_structured_output(self, _schema):
            return self

        def invoke(self, prompt: str):
            prompts_seen.append(prompt)
            return ns(
                mechanism="m",
                prediction="p",
                experiment="e",
                decision="d",
                for_whom="f",
                reversal_condition="r",
                not_doing=[],
            )

    monkeypatch.setattr(nodes, "_llm", lambda *args, **kwargs: CapturingLLM())
    h1 = {
        **_hypothesis("a", elo=1400.0),
        "decision": "launch X",
        "for_whom": "users",
        "reversal_condition": "if CTR < 2%",
    }
    h2 = {
        **_hypothesis("b", elo=1350.0),
        "decision": "try Y",
        "for_whom": "admins",
        "reversal_condition": "if churn > 5%",
    }
    state = _state(
        round=1,
        hypotheses=[h1, h2],
        tournament_log=[],
        decision_mode=True,
        negative_space=False,
    )
    out = evolve_node(state, model="m")
    assert len(out["hypotheses"]) == 3
    combine_prompt = prompts_seen[0]
    assert "Decision:" in combine_prompt  # EVOLVE_DECISION format


# ---------------------------------------------------------------------------
# 3.1 proximity graph integration tests
# ---------------------------------------------------------------------------


def test_pair_similarity_lookup() -> None:
    prox = {
        "a": [{"id": "b", "similarity": 0.85}, {"id": "c", "similarity": 0.4}],
        "b": [{"id": "a", "similarity": 0.85}],
    }
    assert _pair_similarity("a", "b", prox) == 0.85
    assert _pair_similarity("a", "c", prox) == 0.4
    assert _pair_similarity("a", "d", prox) == 0.0  # not in graph
    assert _pair_similarity("x", "y", {}) == 0.0  # empty graph


def test_select_pairs_filters_high_similarity_pairs() -> None:
    """Pairs with proximity similarity >= 0.7 are excluded from tournament."""
    a = _hypothesis("a")
    b = _hypothesis("b")
    c = _hypothesis("c")
    scores = {"a": 1200.0, "b": 1200.0, "c": 1200.0}
    prox = {
        "a": [{"id": "b", "similarity": 0.8}, {"id": "c", "similarity": 0.4}],
        "b": [{"id": "a", "similarity": 0.8}, {"id": "c", "similarity": 0.4}],
        "c": [{"id": "a", "similarity": 0.4}, {"id": "b", "similarity": 0.4}],
    }
    pairs = _select_pairs([a, b, c], scores, n_matches=10, proximity_graph=prox)
    pair_ids = {frozenset([h1["id"], h2["id"]]) for h1, h2 in pairs}
    assert frozenset(["a", "b"]) not in pair_ids  # filtered: sim 0.8 >= 0.7
    assert frozenset(["a", "c"]) in pair_ids  # kept: sim 0.4 < 0.7
    assert frozenset(["b", "c"]) in pair_ids  # kept: sim 0.4 < 0.7


def test_select_pairs_no_filter_without_proximity_graph() -> None:
    """Without a proximity graph, all pairs are eligible (existing behaviour)."""
    a = _hypothesis("a")
    b = _hypothesis("b")
    c = _hypothesis("c")
    scores = {"a": 1200.0, "b": 1200.0, "c": 1200.0}
    pairs = _select_pairs([a, b, c], scores, n_matches=10)
    assert len(pairs) == 3


def test_proximity_node_prunes_near_duplicates() -> None:
    """proximity_node adds the lower-Elo hypothesis of a near-duplicate pair to pruned_ids."""
    high = _hypothesis("a", elo=1400.0)
    low = _hypothesis("b", elo=1200.0)
    # "mechanism-a prediction-a experiment-a" vs "mechanism-b ..." — very similar strings
    state = _state(hypotheses=[high, low], pruned_ids=[])
    out = proximity_node(state)
    # lower-Elo "b" should be pruned; "a" should not
    assert "b" in out["pruned_ids"]
    assert "a" not in out["pruned_ids"]


def test_proximity_node_preserves_existing_pruned_ids() -> None:
    """pruned_ids already in state are preserved in the output."""
    a = _hypothesis("a", elo=1400.0)
    b = _hypothesis("b", elo=1200.0)
    state = _state(hypotheses=[a, b], pruned_ids=["x"])
    out = proximity_node(state)
    assert "x" in out["pruned_ids"]


def test_select_diverse_parents_prefers_dissimilar() -> None:
    """_select_diverse_parents picks quality + diversity over just top-2 Elo."""
    a = _hypothesis("a", elo=1400.0)
    b = _hypothesis("b", elo=1380.0)
    c = _hypothesis("c", elo=1100.0)
    # a-b near-duplicate; b-c moderately similar; a-c dissimilar
    # Scores: (a,b)=0.52, (b,c)=0.69, (a,c)=0.90 → (a,c) wins
    prox = {
        "a": [{"id": "b", "similarity": 0.95}, {"id": "c", "similarity": 0.1}],
        "b": [{"id": "a", "similarity": 0.95}, {"id": "c", "similarity": 0.5}],
        "c": [{"id": "a", "similarity": 0.1}, {"id": "b", "similarity": 0.5}],
    }
    p1, p2 = _select_diverse_parents([a, b, c], prox)
    assert {p1["id"], p2["id"]} == {"a", "c"}


def test_select_diverse_parents_falls_back_without_graph() -> None:
    """Without a proximity graph, the top-2 by Elo are returned."""
    a = _hypothesis("a", elo=1400.0)
    b = _hypothesis("b", elo=1380.0)
    c = _hypothesis("c", elo=1100.0)
    p1, p2 = _select_diverse_parents([a, b, c], {})
    assert {p1["id"], p2["id"]} == {"a", "b"}


def test_evolve_combine_uses_proximity_for_parent_selection(monkeypatch) -> None:
    """evolve_node's combine strategy picks dissimilar parents when a graph is present."""
    prompts_seen: list[str] = []

    class CapturingLLM:
        def with_structured_output(self, _schema):
            return self

        def invoke(self, prompt: str):
            prompts_seen.append(prompt)
            return ns(
                mechanism="m",
                prediction="p",
                experiment="e",
                decision="",
                for_whom="",
                reversal_condition="",
                not_doing=[],
            )

    monkeypatch.setattr(nodes, "_llm", lambda *args, **kwargs: CapturingLLM())

    a = _hypothesis("a", elo=1400.0)
    b = _hypothesis("b", elo=1350.0)
    c = _hypothesis("c", elo=1200.0)
    # a-b are near-duplicates; a-c are dissimilar → combine should pick (a, c)
    prox = {
        "a": [{"id": "b", "similarity": 0.95}],
        "b": [{"id": "a", "similarity": 0.95}],
    }
    state = _state(
        round=1,
        hypotheses=[a, b, c],
        tournament_log=[],
        proximity_graph=prox,
        decision_mode=False,
        negative_space=False,
    )
    evolve_node(state, model="m")
    combine_prompt = prompts_seen[0]
    # The combine prompt must reference "c"'s mechanism as one of the parents
    assert "mechanism-c" in combine_prompt


def test_build_graph_constructs_compiled_graph() -> None:
    graph = graph_mod.build_graph("gpt-4o-mini")
    assert graph is not None


# ---------------------------------------------------------------------------
# Pseudocode-alignment features
# ---------------------------------------------------------------------------


def test_parse_research_plan(monkeypatch) -> None:
    plan = ns(
        objective="cure X",
        sub_questions=["q1", "q2"],
        constraints=[],
        keywords=["x", "y"],
    )
    monkeypatch.setattr(nodes, "_llm", lambda *a, **k: DummyLLM(plan))
    out = nodes.parse_research_plan("how to cure X", model="m")
    assert out["objective"] == "cure X"
    assert out["keywords"] == ["x", "y"]


def test_parse_research_plan_degrades_gracefully(monkeypatch) -> None:
    def boom(*a, **k):
        raise RuntimeError("no api")

    monkeypatch.setattr(nodes, "_llm", boom)
    assert nodes.parse_research_plan("q", model="m") == {}


def test_generate_node_injects_web_search_and_debate(monkeypatch) -> None:
    calls = {"search": 0}

    def fake_search(query, **kwargs):
        calls["search"] += 1
        return {"results": [{"title": "Paper", "content": "evidence here"}]}

    monkeypatch.setattr(nodes, "_exa_api_key", "")
    monkeypatch.setattr(nodes, "tivaly_search", fake_search)

    gen_result = ns(
        mechanism="m",
        prediction="p",
        experiment="e",
        decision="",
        for_whom="",
        reversal_condition="",
        not_doing=[],
    )
    seen: list[str] = []

    class CapLLM:
        def with_structured_output(self, _s):
            return self

        def invoke(self, prompt):
            seen.append(prompt)
            return gen_result

    monkeypatch.setattr(nodes, "_llm", lambda *a, **k: CapLLM())
    state = _state(question="Q", round=0, research_plan={"keywords": ["k1"]})
    streamed: list[str] = []
    progress: list[tuple[str, int]] = []
    out = generate_node(
        state,
        model="m",
        on_hypothesis=lambda hypothesis: streamed.append(hypothesis["id"]),
        on_progress=lambda event: progress.append((event.phase, event.candidate_count)),
    )

    # 12 standard + 3 debate strategy = 15 ideas, none yet in the tournament.
    assert len(out["hypotheses"]) == 15
    assert out["new_hypothesis_ids"] == [h["id"] for h in out["hypotheses"]]
    assert streamed == out["new_hypothesis_ids"]
    assert progress == [("generating", 0), ("debating", 12), ("complete", 15)]
    assert all(
        h["in_tournament"] is False and h["elo"] == 0.0 for h in out["hypotheses"]
    )
    debate_ids = [h["id"] for h in out["hypotheses"] if h["id"].startswith("d0_")]
    assert debate_ids == ["d0_0", "d0_1", "d0_2"]
    # Web search ran and its snippet reached the generation prompt.
    assert calls["search"] == 1
    assert any("evidence here" in p for p in seen)


def test_review_hypothesis_full_and_verification(monkeypatch) -> None:
    combined = ns(
        novelty_score=0.6,
        rigor_score=0.6,
        testability_score=0.6,
        impact_score=0.6,
        simplicity_score=0.6,
        reason="ok",
        assumptions=[ns(assumption="A1", plausible=True, note="fine")],
        verdict="sound",
        summary="looks good",
    )
    monkeypatch.setattr(nodes, "_llm", lambda *a, **k: DummyLLM(combined))
    state = _state(
        question="Q",
        hypotheses=[_hypothesis("a", in_tournament=False, elo=0.0)],
        deep_reflect=False,
    )
    out = nodes.review_hypothesis_node(state, "a", model="m")

    assert out["reviewed_id"] == "a"
    assert out["hypotheses"][0]["reflection"] == ""
    assert "sound" in out["hypotheses"][0]["verification"]
    types = {r["type"] for r in out["reviews"]}
    assert types == {"FullReview", "VerificationReview"}


def test_review_hypothesis_uses_empty_review_when_deep_reflection_fails(
    monkeypatch,
) -> None:
    combined = ns(
        novelty_score=0.6,
        rigor_score=0.6,
        testability_score=0.6,
        impact_score=0.6,
        simplicity_score=0.6,
        reason="ok",
        assumptions=[],
        verdict="sound",
        summary="ok",
    )

    def fail_deep_reflection(**_kwargs):
        raise ConnectionError("connection refused")

    monkeypatch.setattr(nodes, "run_deep_reflection", fail_deep_reflection)
    monkeypatch.setattr(nodes, "_llm", lambda *a, **k: DummyLLM(combined))
    state = _state(
        question="Q",
        hypotheses=[_hypothesis("a", in_tournament=False, elo=0.0)],
        deep_reflect=True,
    )

    out = nodes.review_hypothesis_node(state, "a", model="m")

    assert out["reviewed_id"] == "a"
    assert out["hypotheses"][0]["reflection"] == ""
    assert out["reviews"][0]["review"] == ""


def test_review_hypothesis_searches_web_for_evidence(monkeypatch) -> None:
    """ReviewHypothesis performs a per-hypothesis web search and injects evidence."""
    queries: list[str] = []

    def fake_search(query, **kwargs):
        queries.append(query)
        return {"results": [{"title": "Paper", "content": "supporting evidence"}]}

    monkeypatch.setattr(nodes, "_exa_api_key", "")
    monkeypatch.setattr(nodes, "tivaly_search", fake_search)

    seen_prompts: list[str] = []

    class CapLLM:
        content = "critique"
        novelty_score = rigor_score = testability_score = 0.6
        impact_score = simplicity_score = 0.6
        reason = "ok"
        assumptions: list = []
        verdict = "sound"
        summary = "ok"

        def with_structured_output(self, _s):
            return self

        def invoke(self, prompt):
            seen_prompts.append(prompt)
            return self

    monkeypatch.setattr(nodes, "_llm", lambda *a, **k: CapLLM())
    state = _state(
        question="Q",
        hypotheses=[_hypothesis("a", in_tournament=False, elo=0.0)],
        research_plan={"keywords": ["k1"]},
        deep_reflect=False,
    )
    nodes.review_hypothesis_node(state, "a", model="m")

    # Web search ran for this hypothesis and its evidence reached the review prompts.
    assert queries and "mechanism-a" in queries[0]
    assert any("supporting evidence" in p for p in seen_prompts)


def test_review_hypothesis_prunes_low_novelty(monkeypatch) -> None:
    combined = ns(
        content="weak",
        novelty_score=0.05,
        rigor_score=0.5,
        testability_score=0.5,
        impact_score=0.5,
        simplicity_score=0.5,
        reason="low novelty",
        assumptions=[],
        verdict="questionable",
        summary="meh",
    )
    monkeypatch.setattr(nodes, "_llm", lambda *a, **k: DummyLLM(combined))
    state = _state(
        question="Q",
        hypotheses=[_hypothesis("a", in_tournament=False, elo=0.0)],
        deep_reflect=False,
    )
    out = nodes.review_hypothesis_node(state, "a", model="m")
    assert "a" in out["pruned_ids"]
    assert "reviewed_id" not in out  # pruned → does not enter the tournament


def test_add_to_tournament_seeds_and_guards() -> None:
    state = _state(hypotheses=[_hypothesis("a", in_tournament=False, elo=0.0)])
    out = nodes.add_to_tournament_node(state, "a")
    assert out["hypotheses"][0]["elo"] == 1200.0
    assert out["hypotheses"][0]["in_tournament"] is True
    assert out["added_id"] == "a"

    # Re-entry guard: already in tournament → no-op.
    state2 = _state(hypotheses=[_hypothesis("a", in_tournament=True, elo=1300.0)])
    assert nodes.add_to_tournament_node(state2, "a") == {}


def test_tournament_auto_adds_and_counts_matches(monkeypatch) -> None:
    monkeypatch.setattr(
        nodes, "_llm", lambda *a, **k: DummyLLM(ns(winner="A", reason="x"))
    )
    state = _state(
        question="Q",
        hypotheses=[
            _hypothesis("a", in_tournament=False, elo=0.0),
            _hypothesis("b", in_tournament=False, elo=0.0),
        ],
        tournament_log=[],
    )
    out = tournament_node(state, model="m")
    assert all(h["in_tournament"] for h in out["hypotheses"])
    assert sum(h["matches"] for h in out["hypotheses"]) > 0


def test_proximity_uses_embeddings_when_available(monkeypatch) -> None:
    # a,b identical vectors (dup → prune); c orthogonal (no edge).
    vectors = {"a": [1.0, 0.0], "b": [1.0, 0.0], "c": [0.0, 1.0]}
    monkeypatch.setattr(
        nodes,
        "_embed_texts",
        lambda texts: [vectors[t.split()[0].split("-")[-1]] for t in texts],
    )
    state = _state(
        hypotheses=[
            _hypothesis("a", elo=1200.0),
            _hypothesis("b", elo=1100.0),
            _hypothesis("c", elo=1000.0),
        ],
        pruned_ids=[],
    )
    out = proximity_node(state)
    # a-b are near-identical → the lower-Elo one (b) is pruned.
    assert "b" in out["pruned_ids"]
    # a-c orthogonal → no edge between them.
    assert all(e["id"] != "c" for e in out["proximity_graph"].get("a", []))


def test_cosine_similarity() -> None:
    assert nodes._cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert nodes._cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert nodes._cosine([0.0, 0.0], [1.0, 1.0]) == 0.0  # zero vector


def test_final_overview_node(monkeypatch) -> None:
    monkeypatch.setattr(
        nodes,
        "_llm",
        lambda *a, **k: SimpleNamespace(
            invoke=lambda p: SimpleNamespace(content="the overview")
        ),
    )
    state = _state(
        question="Q",
        hypotheses=[_hypothesis("a", elo=1300.0), _hypothesis("b", elo=1200.0)],
        research_plan={"objective": "obj"},
    )
    out = nodes.final_overview_node(state, model="m")
    assert out["final_overview"] == "the overview"


def test_web_context_handles_empty_search(monkeypatch) -> None:
    monkeypatch.setattr(nodes, "_exa_api_key", "")
    monkeypatch.setattr(nodes, "tivaly_search", lambda q, **k: {"results": []})
    assert nodes._web_context("query", keywords=["k"]) == ""


def test_structured_extra_body_explicit_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8080/v1")
    monkeypatch.setenv("SALUKI_LLM_EXTRA_BODY", '{"reasoning_effort": "none"}')
    assert nodes._structured_extra_body() == {"reasoning_effort": "none"}


def test_structured_extra_body_empty_opts_out(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8080/v1")
    monkeypatch.setenv("SALUKI_LLM_EXTRA_BODY", "{}")
    assert nodes._structured_extra_body() == {}


def test_structured_extra_body_invalid_env_is_ignored(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8080/v1")
    monkeypatch.setenv("SALUKI_LLM_EXTRA_BODY", "{not valid json")
    assert nodes._structured_extra_body() is None


def test_structured_extra_body_defaults_to_none_for_generic_compatible_server(
    monkeypatch,
) -> None:
    monkeypatch.delenv("SALUKI_LLM_EXTRA_BODY", raising=False)
    monkeypatch.delenv("SALUKI_OPENAI_COMPAT_PROVIDER", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8080/v1")
    assert nodes._structured_extra_body() is None


def test_structured_extra_body_uses_vllm_profile(monkeypatch) -> None:
    monkeypatch.delenv("SALUKI_LLM_EXTRA_BODY", raising=False)
    monkeypatch.setenv("SALUKI_OPENAI_COMPAT_PROVIDER", "vllm")
    assert nodes._structured_extra_body() == nodes.VLLM_DISABLE_THINKING_BODY


def test_structured_extra_body_none_for_stock_openai(monkeypatch) -> None:
    monkeypatch.delenv("SALUKI_LLM_EXTRA_BODY", raising=False)
    monkeypatch.delenv("SALUKI_OPENAI_COMPAT_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    assert nodes._structured_extra_body() is None


def test_extra_body_applied_only_to_structured_calls(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("SALUKI_LLM_EXTRA_BODY", '{"reasoning_effort": "none"}')

    # Plain (free-form) llm: reasoning left intact.
    plain = nodes._llm("m")
    assert (plain.extra_body or None) is None
    # Structured path: extra body (reasoning toggle) is applied.
    wrapped = nodes._with_extra_body(plain)
    assert wrapped.extra_body == {"reasoning_effort": "none"}


def test_format_exception_includes_chained_cause() -> None:
    try:
        try:
            raise OSError("server closed connection")
        except OSError as exc:
            raise RuntimeError("Connection error.") from exc
    except RuntimeError as exc:
        text = nodes._format_exception(exc)

    assert "RuntimeError: Connection error." in text
    assert "OSError: server closed connection" in text


def test_invoke_structured_retries_then_succeeds(monkeypatch) -> None:
    monkeypatch.setattr(nodes, "STRUCTURED_RETRY_MIN_SECONDS", 0)
    monkeypatch.setattr(nodes, "STRUCTURED_RETRY_MAX_SECONDS", 0)
    calls = {"n": 0}

    class FlakyLLM:
        def invoke(self, prompt):
            calls["n"] += 1
            if calls["n"] < nodes.STRUCTURED_RETRIES:
                raise ValueError("boom")
            return "ok"

    assert nodes._invoke_structured(FlakyLLM(), "p") == "ok"
    assert calls["n"] == nodes.STRUCTURED_RETRIES


def test_invoke_structured_raises_after_exhausting_retries(monkeypatch) -> None:
    monkeypatch.setattr(nodes, "STRUCTURED_RETRY_MIN_SECONDS", 0)
    monkeypatch.setattr(nodes, "STRUCTURED_RETRY_MAX_SECONDS", 0)
    calls = {"n": 0}

    class BoomLLM:
        def invoke(self, prompt):
            calls["n"] += 1
            raise ValueError("always")

    with pytest.raises(ValueError):
        nodes._invoke_structured(BoomLLM(), "p")
    assert calls["n"] == nodes.STRUCTURED_RETRIES


def test_generate_skips_hypothesis_that_fails_after_retries(monkeypatch) -> None:
    monkeypatch.setattr(nodes, "STRUCTURED_RETRY_MIN_SECONDS", 0)
    monkeypatch.setattr(nodes, "STRUCTURED_RETRY_MAX_SECONDS", 0)
    monkeypatch.setattr(nodes, "_llm", lambda *a, **k: object())
    monkeypatch.setattr(nodes, "_web_context", lambda *a, **k: "")

    # Step 1 succeeds in 1 call; step 2's retry attempts (calls 2..1+R) all fail,
    # so that one hypothesis is skipped. Remaining steps + debate succeed.
    fail_window = set(range(2, 2 + nodes.STRUCTURED_RETRIES))

    class PartialLLM:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, prompt):
            self.calls += 1
            if self.calls in fail_window:
                raise ValueError("flaky")
            return ns(
                mechanism="m",
                prediction="p",
                experiment="e",
                decision="",
                for_whom="",
                reversal_condition="",
                not_doing=[],
            )

    stub = PartialLLM()
    monkeypatch.setattr(nodes, "_structured", lambda llm, schema: stub)

    out = generate_node(_state(question="q", idea_research="brief", round=0), model="m")
    # 12 generation steps + 3 debate = 15, minus the one skipped step = 14.
    assert len(out["hypotheses"]) == 14
    assert out["new_hypothesis_ids"] == [h["id"] for h in out["hypotheses"]]
