"""Tests for the task-queue architecture (step 1.1)."""

from __future__ import annotations

from types import SimpleNamespace

import saluki.agents as agents_mod
import saluki.task_queue as tq
from saluki.state import (
    ProximityGraph,
    SystemFeedback,
    GlobalState as GS,
)
from conftest import DummyLLM, ns, _hypothesis, _state


# ------------------------------------------------------------------
# Task & TaskQueue
# ------------------------------------------------------------------


def test_task_attributes() -> None:
    task = tq.Task(agent="generate", action="CreateHypotheses")
    assert task.agent == "generate"
    assert task.action == "CreateHypotheses"
    assert task.target_id is None
    assert task.payload == {}


def test_task_with_payload() -> None:
    task = tq.Task(agent="generate", action="X", payload={"model": "gpt-4"})
    assert task.payload == {"model": "gpt-4"}


def test_task_queue_enqueue_dequeue() -> None:
    queue = tq.TaskQueue()
    assert queue.is_empty()
    queue.enqueue(tq.Task(agent="a", action="1"))
    assert not queue.is_empty()
    task = queue.dequeue()
    assert task is not None
    assert task.agent == "a"
    assert queue.is_empty()


def test_task_queue_dequeue_empty() -> None:
    assert tq.TaskQueue().dequeue() is None


def test_task_queue_enqueue_priority() -> None:
    queue = tq.TaskQueue()
    queue.enqueue(tq.Task(agent="a", action="1"))
    queue.enqueue(tq.Task(agent="b", action="2"))
    queue.enqueue_priority(tq.Task(agent="urgent", action="0"))
    first = queue.dequeue()
    assert first is not None
    assert first.agent == "urgent"  # runs first


def test_task_queue_len() -> None:
    queue = tq.TaskQueue()
    assert len(queue) == 0
    queue.enqueue(tq.Task(agent="a", action="1"))
    assert len(queue) == 1


def test_collect_ready_tasks_no_parallel_takes_one_task() -> None:
    sup = tq.Supervisor(model="m")
    sup.queue.enqueue(tq.Task(agent="a", action="1"))
    sup.queue.enqueue(tq.Task(agent="b", action="2"))
    sup.queue.enqueue(tq.Task(agent="c", action="3"))

    tasks = sup._collect_ready_tasks(no_parallel=True)

    assert [(task.agent, task.action) for task in tasks] == [("a", "1")]
    assert len(sup.queue) == 2
    next_task = sup.queue.dequeue()
    assert next_task is not None
    assert next_task.agent == "b"


def test_collect_ready_tasks_default_batches_independent_tasks() -> None:
    sup = tq.Supervisor(model="m")
    sup.queue.enqueue(tq.Task(agent="a", action="1"))
    sup.queue.enqueue(tq.Task(agent="b", action="2"))
    sup.queue.enqueue(tq.Task(agent="c", action="3"))

    tasks = sup._collect_ready_tasks()

    assert [(task.agent, task.action) for task in tasks] == [
        ("a", "1"),
        ("b", "2"),
        ("c", "3"),
    ]
    assert sup.queue.is_empty()


# ------------------------------------------------------------------
# Supervisor follow-up task generation
# ------------------------------------------------------------------


def test_supervisor_follow_up_bootstrap() -> None:
    sup = tq.Supervisor(model="m")
    task = tq.Task(agent="bootstrap", action="Init")
    follow_ups = sup._generate_follow_up_tasks(task, {})
    assert len(follow_ups) == 1
    assert follow_ups[0].agent == "idea_research"


def test_supervisor_follow_up_idea_research() -> None:
    sup = tq.Supervisor(model="m")
    task = tq.Task(agent="idea_research", action="ResearchIdeaSpace")
    follow_ups = sup._generate_follow_up_tasks(task, {})
    assert len(follow_ups) == 1
    assert follow_ups[0].agent == "generate"


def test_supervisor_unknown_agent_returns_empty() -> None:
    sup = tq.Supervisor(model="m")
    task = tq.Task(agent="unknown", action="X")
    follow_ups = sup._generate_follow_up_tasks(task, {})
    assert follow_ups == []


def test_supervisor_generate_routes_each_new_hyp_to_review() -> None:
    """Each created hypothesis spawns a per-id ReviewHypothesis task."""
    sup = tq.Supervisor(model="m")
    task = tq.Task(agent="generate", action="CreateInitialHypotheses")
    result = {"new_hypothesis_ids": ["h0_0", "h0_1"]}
    follow_ups = sup._generate_follow_up_tasks(task, result)
    assert [(t.agent, t.action, t.target_id) for t in follow_ups] == [
        ("reflect", "ReviewHypothesis", "h0_0"),
        ("reflect", "ReviewHypothesis", "h0_1"),
    ]


def test_supervisor_review_routes_to_add_to_tournament() -> None:
    """A completed review enqueues AddToTournament for that hypothesis."""
    sup = tq.Supervisor(model="m")
    task = tq.Task(agent="reflect", action="ReviewHypothesis", target_id="h0_0")
    follow_ups = sup._generate_follow_up_tasks(task, {"reviewed_id": "h0_0"})
    assert len(follow_ups) == 1
    assert follow_ups[0].agent == "ranking"
    assert follow_ups[0].action == "AddToTournament"
    assert follow_ups[0].target_id == "h0_0"


def test_supervisor_pruned_review_does_not_enter_tournament() -> None:
    """A review that prunes the hypothesis emits no reviewed_id, so no AddToTournament."""
    sup = tq.Supervisor(model="m")
    task = tq.Task(agent="reflect", action="ReviewHypothesis", target_id="h0_0")
    # No reviewed_id signal → no follow-up.
    follow_ups = sup._generate_follow_up_tasks(task, {"pruned_ids": ["h0_0"]})
    assert follow_ups == []


def test_supervisor_evolve_routes_children_to_review() -> None:
    """Evolved children are treated as brand-new ideas → ReviewHypothesis each."""
    sup = tq.Supervisor(model="m")
    task = tq.Task(agent="evolve", action="EvolveTopHypotheses")
    result = {"new_hypothesis_ids": ["e1_combine", "e1_simplify"]}
    follow_ups = sup._generate_follow_up_tasks(task, result)
    assert all(
        t.agent == "reflect" and t.action == "ReviewHypothesis" for t in follow_ups
    )
    assert {t.target_id for t in follow_ups} == {"e1_combine", "e1_simplify"}


def test_supervisor_decide_next_steps() -> None:
    sup = tq.Supervisor(model="m")
    sup.state = GS(_state(max_rounds=5))
    tasks = sup.decide_next_steps()
    assert len(tasks) >= 1
    # Tournament is always enqueued to keep refining scores.
    assert any(t.agent == "ranking" and t.action == "RunTournamentBatch" for t in tasks)


# ------------------------------------------------------------------
# Agents (unit tests that mock the LLM)
# ------------------------------------------------------------------


def test_bootstrap_agent_executes(monkeypatch) -> None:
    monkeypatch.setattr(
        agents_mod.nodes,
        "parse_research_plan",
        lambda *a, **k: {"objective": "Q"},
    )
    agent = agents_mod.BootstrapAgent(model="m")
    state = _state(question="Q", meta_review="review", idea_research="brief")
    result = agent.execute(tq.Task(agent="bootstrap", action="Init"), state)
    assert result["round"] == 0
    assert result["hypotheses"] == []
    assert result["research_plan"] == {"objective": "Q"}


def test_proximity_agent_executes() -> None:
    agent = agents_mod.ProximityAgent(model="m")
    state = _state(
        hypotheses=[_hypothesis("a"), _hypothesis("b")],
        pruned_ids=[],
    )
    out = agent.execute(tq.Task(agent="proximity", action="Prox"), state)
    assert "a" in out["proximity_graph"]


def test_ranking_agent_runs_tournament(monkeypatch) -> None:
    compare_result = ns(winner="B", reason="better")
    monkeypatch.setattr(
        agents_mod.nodes, "_llm", lambda *args, **kwargs: DummyLLM(compare_result)
    )
    monkeypatch.setattr(agents_mod.nodes, "_embed_texts", lambda texts: None)

    agent = agents_mod.RankingAgent(model="m")
    state = _state(
        question="Q",
        hypotheses=[_hypothesis("a"), _hypothesis("b")],
        proximity_graph={},
        tournament_log=[],
    )
    out = agent.execute(tq.Task(agent="ranking", action="RunTournamentBatch"), state)
    assert "tournament_log" in out


def test_ranking_agent_add_to_tournament(monkeypatch) -> None:
    agent = agents_mod.RankingAgent(model="m")
    state = _state(hypotheses=[_hypothesis("a", in_tournament=False, elo=0.0)])
    out = agent.execute(
        tq.Task(agent="ranking", action="AddToTournament", target_id="a"), state
    )
    assert out["hypotheses"][0]["elo"] == 1200.0
    assert out["hypotheses"][0]["in_tournament"] is True
    assert out["added_id"] == "a"


def test_evolution_agent_executes(monkeypatch) -> None:
    evolve_result = ns(
        mechanism="new mech",
        prediction="new pred",
        experiment="new exp",
        decision="",
        for_whom="",
        reversal_condition="",
        not_doing=[],
    )
    monkeypatch.setattr(
        agents_mod.nodes, "_llm", lambda *args, **kwargs: DummyLLM(evolve_result)
    )

    agent = agents_mod.EvolutionAgent(model="m")
    state = _state(
        round=2,
        hypotheses=[_hypothesis("a", elo=1300.0), _hypothesis("b", elo=1250.0)],
        tournament_log=[{"winner": "a", "loser": "b", "reason": "x"}],
        meta_review="",
    )
    out = agent.execute(tq.Task(agent="evolve", action="EvolveTopHypotheses"), state)
    assert out["hypotheses"][0]["mechanism"] == "new mech"
    assert out["new_hypothesis_ids"]


def test_metareview_agent_meta_review(monkeypatch) -> None:
    monkeypatch.setattr(
        agents_mod.nodes,
        "_llm",
        lambda *args, **kwargs: SimpleNamespace(
            invoke=lambda prompt: SimpleNamespace(content="review text"),
        ),
    )
    agent = agents_mod.MetareviewAgent(model="m")
    state = _state(
        round=2, hypotheses=[_hypothesis("a")], tournament_log=[], meta_review=""
    )
    out = agent.execute(tq.Task(agent="metareview", action="MetaReview"), state)
    assert out["meta_review"] == "review text"


def test_metareview_agent_final_overview(monkeypatch) -> None:
    monkeypatch.setattr(
        agents_mod.nodes,
        "_llm",
        lambda *args, **kwargs: SimpleNamespace(
            invoke=lambda prompt: SimpleNamespace(content="the overview"),
        ),
    )
    agent = agents_mod.MetareviewAgent(model="m")
    state = _state(hypotheses=[_hypothesis("a")])
    out = agent.execute(
        tq.Task(agent="metareview", action="GenerateFinalResearchOverview"), state
    )
    assert out["final_overview"] == "the overview"


def test_metareview_agent_system_feedback(monkeypatch) -> None:
    monkeypatch.setattr(
        agents_mod.nodes,
        "_llm",
        lambda *args, **kwargs: SimpleNamespace(
            invoke=lambda prompt: SimpleNamespace(content="cross-round feedback"),
        ),
    )
    agent = agents_mod.MetareviewAgent(model="m")
    state = _state(
        round=2,
        hypotheses=[_hypothesis("a"), _hypothesis("b")],
        tournament_log=[{"winner": "a", "loser": "b", "reason": "better"}],
        meta_review="round 1 review",
    )
    out = agent.execute(
        tq.Task(agent="metareview", action="GenerateSystemFeedback"), state
    )
    assert out["system_wide_feedback"] == "cross-round feedback"


def test_idea_research_agent_executes(monkeypatch) -> None:
    monkeypatch.setattr(agents_mod.nodes, "run_idea_research", lambda **kwargs: "brief")

    agent = agents_mod.IdeaResearchAgent(model="m")
    state = _state(question="Q", hypotheses=[], meta_review="", idea_research="")
    out = agent.execute(tq.Task(agent="idea_research", action="Research"), state)
    assert out == {"idea_research": "brief"}


def test_supervisor_decide_next_steps_enqueues_system_feedback(monkeypatch) -> None:
    """When enough iterations have passed, decide_next_steps enqueues system feedback."""
    sup = tq.Supervisor(model="m")
    sup.state = GS(_state(max_rounds=5))
    sup._iterations = 5
    sup._last_system_feedback_iteration = 0
    sup._last_proximity_iteration = 5  # suppress proximity to isolate this assertion

    tasks = sup.decide_next_steps()
    actions = [t.action for t in tasks]
    assert "GenerateSystemFeedback" in actions


def test_supervisor_decide_next_steps_no_system_feedback_when_recent(
    monkeypatch,
) -> None:
    """When system feedback was generated recently, do not enqueue another."""
    sup = tq.Supervisor(model="m")
    sup.state = GS(_state(max_rounds=5))
    sup._iterations = 5
    sup._last_system_feedback_iteration = 4  # only 1 iteration since last
    sup._last_proximity_iteration = 5

    tasks = sup.decide_next_steps()
    actions = [t.action for t in tasks]
    assert "GenerateSystemFeedback" not in actions


# ------------------------------------------------------------------
# Completion + convergence gating
# ------------------------------------------------------------------


def test_is_complete_requires_ideas_and_matches() -> None:
    # A requested idea target should not be cut short by the match threshold.
    sup = tq.Supervisor(model="m")
    sup._max_ideas = 2
    sup._max_matches_per_idea = 3

    # Enough matches (min>=3) but not enough ideas → not complete.
    sup._num_ideas = 1
    sup.state = GS(
        _state(
            hypotheses=[
                _hypothesis("a", matches=3),
                _hypothesis("b", matches=3),
            ]
        )
    )
    assert not sup._is_complete()

    # Enough ideas but a contender short on matches → not complete.
    sup._num_ideas = 2
    sup.state = GS(
        _state(
            hypotheses=[
                _hypothesis("a", matches=3),
                _hypothesis("b", matches=1),
            ]
        )
    )
    assert not sup._is_complete()

    # Both targets reached → complete.
    sup._num_ideas = 2
    sup.state = GS(
        _state(
            hypotheses=[
                _hypothesis("a", matches=3),
                _hypothesis("b", matches=3),
            ]
        )
    )
    assert sup._is_complete()

    # Neither target reached → not complete.
    sup._num_ideas = 1
    sup.state = GS(
        _state(
            hypotheses=[
                _hypothesis("a", matches=3),
                _hypothesis("b", matches=1),
            ]
        )
    )
    assert not sup._is_complete()


def test_decide_next_steps_emits_final_overview_when_complete() -> None:
    sup = tq.Supervisor(model="m")
    sup._max_ideas = 1
    sup._max_matches_per_idea = 1
    sup._num_ideas = 2
    sup.state = GS(
        _state(
            hypotheses=[
                _hypothesis("a", matches=2),
                _hypothesis("b", matches=2),
            ]
        )
    )
    tasks = sup.decide_next_steps()
    assert [(t.agent, t.action) for t in tasks] == [
        ("metareview", "GenerateFinalResearchOverview")
    ]
    # Emitted once only.
    assert sup.decide_next_steps() == []


def test_decide_next_steps_does_not_stop_before_max_ideas() -> None:
    """Regression for --max-ideas 100 stopping after the match target at ~9 ideas."""
    sup = tq.Supervisor(model="m")
    sup._max_ideas = 100
    sup._max_matches_per_idea = 4
    sup._num_ideas = 9
    sup.state = GS(
        _state(
            max_rounds=100,
            hypotheses=[
                _hypothesis(f"h{i}", matches=4, elo=1200.0 + i) for i in range(9)
            ],
        )
    )

    tasks = sup.decide_next_steps()
    assert all(t.action != "GenerateFinalResearchOverview" for t in tasks)
    assert any(t.action == "RunTournamentBatch" for t in tasks)
    assert not sup._final_overview_done


def test_decide_next_steps_idle_backstop_forces_final_overview() -> None:
    # Targets never met (1 contender can't be a real tournament; never complete),
    # but the idle-cycle backstop must still terminate by emitting the overview.
    sup = tq.Supervisor(model="m")
    sup._max_ideas = 1000
    sup._max_matches_per_idea = 1000
    sup.state = GS(_state(hypotheses=[_hypothesis("a", matches=0)]))
    assert not sup._is_complete()

    emitted = None
    for _ in range(sup.MAX_IDLE_CYCLES):
        tasks = sup.decide_next_steps()
        if any(t.action == "GenerateFinalResearchOverview" for t in tasks):
            emitted = tasks
            break
    assert emitted is not None
    assert [(t.agent, t.action) for t in emitted] == [
        ("metareview", "GenerateFinalResearchOverview")
    ]
    assert sup.decide_next_steps() == []  # nothing more after termination


def test_should_evolve_triggers_after_stall() -> None:
    sup = tq.Supervisor(model="m")
    sup._max_ideas = 100  # don't let the idea cap interfere
    sup.state = GS(
        _state(
            max_rounds=5,
            hypotheses=[_hypothesis("a", elo=1200.0), _hypothesis("b", elo=1100.0)],
        )
    )
    # Stable top Elo across calls; STALL_LIMIT=2 → evolve on the 3rd check.
    assert sup._should_evolve() is False
    assert sup._should_evolve() is False
    assert sup._should_evolve() is True
    assert sup._evolve_cycles == 1


def test_should_evolve_respects_max_rounds_cap() -> None:
    sup = tq.Supervisor(model="m")
    sup._max_ideas = 100
    sup.state = GS(
        _state(
            max_rounds=0,  # no evolution cycles allowed
            hypotheses=[_hypothesis("a"), _hypothesis("b")],
        )
    )
    for _ in range(5):
        assert sup._should_evolve() is False


def test_decide_next_steps_evolves_on_cadence_despite_climbing_elo() -> None:
    # Regression: a dominant hypothesis whose Elo keeps climbing never looks
    # "stalled", so stall-only logic never evolved and the run generated no new
    # ideas.  Evolution must now fire on a cadence regardless.
    sup = tq.Supervisor(model="m")
    sup._max_ideas = 100
    sup._max_matches_per_idea = 1000  # keep matches target out of the way
    top = _hypothesis("a", elo=1500.0, matches=0)
    sup.state = GS(
        _state(
            max_rounds=1000,
            hypotheses=[top, _hypothesis("b", elo=1100.0, matches=0)],
        )
    )

    evolved = False
    for _ in range(sup.EVOLVE_INTERVAL + 1):
        top["elo"] += 25.0  # top Elo always improving → never "stalled"
        tasks = sup.decide_next_steps()
        if any(t.action == "EvolveTopHypotheses" for t in tasks):
            evolved = True
            break
    assert evolved, "evolution should fire on cadence even when Elo keeps climbing"


def test_decide_next_steps_evolves_on_first_idle_cycle_below_idea_target() -> None:
    sup = tq.Supervisor(model="m")
    sup._max_ideas = 100
    sup._max_matches_per_idea = 1000
    sup.state = GS(
        _state(
            max_rounds=100,
            hypotheses=[
                _hypothesis("a", elo=1300.0, matches=0),
                _hypothesis("b", elo=1200.0, matches=0),
            ],
        )
    )

    tasks = sup.decide_next_steps()
    assert any(t.action == "EvolveTopHypotheses" for t in tasks)


def test_decide_next_steps_prints_verbose_standings(monkeypatch) -> None:
    lines: list[str] = []
    monkeypatch.setattr("saluki.display.click.echo", lambda msg="": lines.append(msg))

    sup = tq.Supervisor(model="m")
    sup._max_ideas = 100
    sup._max_matches_per_idea = 1000
    sup._num_ideas = 2
    sup._iterations = sup.STANDINGS_INTERVAL
    sup.state = GS(
        _state(
            verbose=True,
            max_rounds=100,
            hypotheses=[
                _hypothesis("a", elo=1300.0, matches=0),
                _hypothesis("b", elo=1200.0, matches=0),
            ],
        )
    )

    sup.decide_next_steps()
    output = "\n".join(lines)
    assert "Current standings (2/100 ideas)" in output
    assert "[a]" in output
    assert output.index("[a]") < output.index("[b]")


def test_decide_next_steps_idle_reset_on_new_ideas() -> None:
    # Productive cycles (new ideas added) must reset the idle backstop so a
    # generating run isn't killed at MAX_IDLE_CYCLES.
    sup = tq.Supervisor(model="m")
    sup._max_ideas = 1000
    sup._max_matches_per_idea = 1000
    sup._num_ideas = 2
    sup._last_productive_ideas = 2
    sup.state = GS(
        _state(
            hypotheses=[
                _hypothesis("a", matches=0),
                _hypothesis("b", matches=0),
            ]
        )
    )

    # Run well past the backstop, simulating one new idea per cycle.
    for i in range(sup.MAX_IDLE_CYCLES * 2):
        sup._num_ideas += 1  # productive: new idea each cycle
        tasks = sup.decide_next_steps()
        assert all(
            t.action != "GenerateFinalResearchOverview" for t in tasks
        ), f"backstop fired at cycle {i} despite continued idea generation"
    assert not sup._final_overview_done


# ------------------------------------------------------------------
# Agent registry
# ------------------------------------------------------------------


def test_agent_registry_contains_all_agents() -> None:
    expected = {
        "bootstrap",
        "idea_research",
        "generate",
        "reflect",
        "ranking",
        "proximity",
        "evolve",
        "metareview",
    }
    assert set(agents_mod.AGENT_REGISTRY.keys()) == expected


# ------------------------------------------------------------------
# _feedback_text priority
# ------------------------------------------------------------------


def test_feedback_text_prefers_system_wide_feedback() -> None:
    """_feedback_text should prefer system_wide_feedback over meta_review."""
    from saluki.nodes import _feedback_text

    state = _state(
        meta_review="per-round review",
        system_wide_feedback="cross-round feedback",
    )
    assert _feedback_text(state) == "cross-round feedback"

    state2 = _state(
        meta_review="per-round review",
        system_wide_feedback="",
    )
    assert _feedback_text(state2) == "per-round review"

    state3 = _state(meta_review="", system_wide_feedback="")
    assert _feedback_text(state3) == "None yet."


# ------------------------------------------------------------------
# _select_pairs (step 2.1)
# ------------------------------------------------------------------


def test_select_pairs_fewer_than_n_matches_returns_all(
    monkeypatch,
) -> None:
    from saluki.nodes import _select_pairs

    a = _hypothesis("a", elo=1200.0, generation=1)
    b = _hypothesis("b", elo=1250.0, generation=1)
    c = _hypothesis("c", elo=1300.0, generation=1)
    active = [a, b, c]
    scores = {"a": 1200.0, "b": 1250.0, "c": 1300.0}

    pairs = _select_pairs(active, scores, n_matches=10)
    assert len(pairs) == 3  # all pairs: (a,b), (a,c), (b,c)


def test_select_pairs_prioritizes_close_elo(monkeypatch) -> None:
    from saluki.nodes import _select_pairs

    # Create hypotheses with varying Elo spreads
    a = _hypothesis("a", elo=1200.0, generation=1)
    b = _hypothesis("b", elo=1210.0, generation=1)  # close to a
    c = _hypothesis("c", elo=1500.0, generation=1)  # far from a
    d = _hypothesis("d", elo=1510.0, generation=1)  # close to c

    active = [a, b, c, d]
    scores = {"a": 1200.0, "b": 1210.0, "c": 1500.0, "d": 1510.0}

    # Request only 3 matches — should prefer close-Elo pairs
    pairs = _select_pairs(active, scores, n_matches=3)
    assert len(pairs) == 3

    # The top pairs should be (a,b) and (c,d) — closest Elo pairs
    pair_ids = {(h1["id"], h2["id"]) for h1, h2 in pairs}
    assert ("a", "b") in pair_ids or ("b", "a") in pair_ids
    assert ("c", "d") in pair_ids or ("d", "c") in pair_ids


def test_select_pairs_includes_recency_for_same_elo(
    monkeypatch,
) -> None:
    from saluki.nodes import _select_pairs

    # Same Elo, different generations
    a = _hypothesis("a", elo=1200.0, generation=1)
    b = _hypothesis("b", elo=1200.0, generation=1)
    c = _hypothesis("c", elo=1200.0, generation=3)
    d = _hypothesis("d", elo=1200.0, generation=3)

    active = [a, b, c, d]
    scores = {"a": 1200.0, "b": 1200.0, "c": 1200.0, "d": 1200.0}

    pairs = _select_pairs(active, scores, n_matches=3)
    assert len(pairs) == 3

    # With same Elo, recency breaks ties: (a,c) and (b,d) involve newer hypotheses
    # so they should be preferred over (a,b) and (c,d)
    pair_ids = {(h1["id"], h2["id"]) for h1, h2 in pairs}
    # At least one pair should mix generations (recency bonus)
    cross_gen = [
        p
        for p in pair_ids
        if (p[0] in ("a", "b") and p[1] in ("c", "d"))
        or (p[1] in ("a", "b") and p[0] in ("c", "d"))
    ]
    assert len(cross_gen) >= 1


# ------------------------------------------------------------------
# 4.1 data model: ProximityGraph, SystemFeedback, GlobalState
# ------------------------------------------------------------------


def test_proximity_graph_update_and_query() -> None:
    pg = ProximityGraph()
    assert pg.is_empty()
    pg.update(
        {"a": [{"id": "b", "similarity": 0.8}], "b": [{"id": "a", "similarity": 0.8}]}
    )
    assert not pg.is_empty()
    assert pg.get_similarity("a", "b") == 0.8
    assert pg.get_similarity("a", "c") == 0.0  # unknown pair
    assert "a" in pg.get_all()


def test_proximity_graph_replace_on_update() -> None:
    pg = ProximityGraph({"x": [{"id": "y", "similarity": 0.5}]})
    pg.update({"a": [{"id": "b", "similarity": 0.9}]})
    assert pg.get_similarity("x", "y") == 0.0  # old data gone
    assert pg.get_similarity("a", "b") == 0.9


def test_system_feedback_update_and_get() -> None:
    sf = SystemFeedback()
    assert not sf
    sf.update("  great progress  ")
    assert sf.get() == "great progress"  # stripped
    assert bool(sf)


def test_system_feedback_update_clears() -> None:
    sf = SystemFeedback("old feedback")
    sf.update("new feedback")
    assert sf.get() == "new feedback"


def test_global_state_uses_structured_fields() -> None:
    from saluki.state import GlobalState

    state = _state(
        system_wide_feedback="some feedback",
        proximity_graph={"a": [{"id": "b", "similarity": 0.6}]},
    )
    gs = GlobalState(state)
    assert isinstance(gs.proximity_graph, ProximityGraph)
    assert isinstance(gs.system_feedback, SystemFeedback)
    assert gs.proximity_graph.get_similarity("a", "b") == 0.6
    assert gs.system_feedback.get() == "some feedback"


def test_global_state_to_dict_round_trips() -> None:
    from saluki.state import GlobalState

    state = _state(
        system_wide_feedback="feedback",
        proximity_graph={"x": [{"id": "y", "similarity": 0.7}]},
        hypotheses=[_hypothesis("h1")],
    )
    gs = GlobalState(state)
    d = gs.to_dict()
    assert d["system_wide_feedback"] == "feedback"
    assert d["proximity_graph"] == {"x": [{"id": "y", "similarity": 0.7}]}
    assert d["hypotheses"][0]["id"] == "h1"


def test_merge_result_routes_proximity_graph_to_class() -> None:
    from saluki.state import GlobalState

    sup = tq.Supervisor(model="m")
    sup.state = GlobalState(_state())
    sup._merge_result({"proximity_graph": {"a": [{"id": "b", "similarity": 0.5}]}})
    assert sup.state.proximity_graph.get_similarity("a", "b") == 0.5


def test_merge_result_routes_system_feedback_to_class() -> None:
    from saluki.state import GlobalState

    sup = tq.Supervisor(model="m")
    sup.state = GlobalState(_state())
    sup._merge_result({"system_wide_feedback": "cross-round insight"})
    assert sup.state.system_feedback.get() == "cross-round insight"


def test_merge_result_counts_only_new_hypothesis_ids() -> None:
    from saluki.state import GlobalState

    sup = tq.Supervisor(model="m")
    sup.state = GlobalState(_state(hypotheses=[_hypothesis("e0_simplify")]))
    sup._num_ideas = 1

    sup._merge_result(
        {
            "hypotheses": [_hypothesis("e0_simplify"), _hypothesis("e1_simplify")],
            "new_hypothesis_ids": ["e0_simplify", "e1_simplify", "e1_simplify"],
        }
    )

    assert sup._num_ideas == 2
    assert {h["id"] for h in sup.state.store.get_all()} == {
        "e0_simplify",
        "e1_simplify",
    }


# ------------------------------------------------------------------
# End-to-end routing through the supervisor (mocked agents)
# ------------------------------------------------------------------


def test_supervisor_end_to_end_routes_and_terminates(monkeypatch) -> None:
    """Drive the supervisor with stubbed agents and assert event-driven routing:
    generate → review(per id) → AddToTournament(per id) → tournament → final overview.
    """
    seen_actions: list[tuple[str, str, str | None]] = []

    class StubAgent:
        def __init__(self, model):
            self.model = model

        def execute(self, task, state):
            seen_actions.append((task.agent, task.action, task.target_id))
            if task.agent == "bootstrap":
                return {"research_plan": {"objective": "o"}}
            if task.agent == "idea_research":
                return {"idea_research": "brief"}
            if task.agent == "generate":
                hyps = [
                    _hypothesis(f"h{i}", in_tournament=False, elo=0.0) for i in range(2)
                ]
                return {
                    "hypotheses": hyps,
                    "new_hypothesis_ids": [h["id"] for h in hyps],
                }
            if task.agent == "reflect":
                return {"reviewed_id": task.target_id}
            if task.agent == "ranking" and task.action == "AddToTournament":
                tid = task.target_id
                h = next(x for x in state.store.get_all() if x["id"] == tid)
                return {"hypotheses": [{**h, "in_tournament": True, "elo": 1200.0}]}
            if task.agent == "ranking":  # RunTournamentBatch
                bumped = [
                    {**h, "matches": h.get("matches", 0) + 5}
                    for h in state.store.get_all()
                    if h.get("in_tournament")
                ]
                return {
                    "hypotheses": bumped,
                    "tournament_log": [{"winner": "h0", "loser": "h1", "reason": "x"}],
                }
            if task.agent == "proximity":
                return {"proximity_graph": {}}
            if (
                task.agent == "metareview"
                and task.action == "GenerateFinalResearchOverview"
            ):
                return {"final_overview": "DONE"}
            return {}

    registry = {
        name: StubAgent
        for name in (
            "bootstrap",
            "idea_research",
            "generate",
            "reflect",
            "ranking",
            "proximity",
            "evolve",
            "metareview",
        )
    }
    monkeypatch.setattr("saluki.agents.AGENT_REGISTRY", registry)
    # run() parses the research plan up front — stub it to stay offline.
    monkeypatch.setattr(
        agents_mod.nodes, "parse_research_plan", lambda *a, **k: {"objective": "o"}
    )

    sup = tq.Supervisor(model="m")
    final = sup.run(
        question="Q",
        max_rounds=1,
        max_ideas=2,
        max_matches_per_idea=4,
    )

    # Final overview was produced → run terminated cleanly.
    assert final["final_overview"] == "DONE"

    # Per-hypothesis routing happened for both ideas.
    review_targets = {t for a, act, t in seen_actions if act == "ReviewHypothesis"}
    add_targets = {t for a, act, t in seen_actions if act == "AddToTournament"}
    assert review_targets == {"h0", "h1"}
    assert add_targets == {"h0", "h1"}
