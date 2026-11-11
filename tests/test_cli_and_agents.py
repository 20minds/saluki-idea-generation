from __future__ import annotations

import runpy
from typing import Any, cast
from types import SimpleNamespace

from click.testing import CliRunner

import saluki.cli as cli
import saluki.deep_reflect as deep_reflect
import saluki.idea_research as idea_research
import saluki.ranking as ranking


def test_cli_main_runs_with_fake_graph(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(cli, "_print_results", lambda final: None)

    class FakeGraph:
        def invoke(self, initial_state: dict) -> dict:
            return {
                **initial_state,
                "hypotheses": [],
                "pruned_ids": [],
                "meta_review": "",
            }

    monkeypatch.setattr(
        "saluki.graph.build_graph", lambda model, base_url=None: FakeGraph()
    )

    result = CliRunner().invoke(
        cli.main,
        ["generate", "-p", "question", "-m", "gpt-4o-mini", "--no-task-queue"],
    )
    assert result.exit_code == 0


def test_cli_no_parallel_reaches_supervisor(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(cli, "_print_results", lambda final: None)
    captured: dict[str, Any] = {}

    def fake_run_with_supervisor(options: cli.SupervisorOptions) -> dict:
        captured["options"] = options
        return {
            "hypotheses": [],
            "pruned_ids": [],
            "meta_review": "",
            "final_overview": "",
        }

    monkeypatch.setattr(cli, "_run_with_supervisor", fake_run_with_supervisor)

    result = CliRunner().invoke(
        cli.main, ["generate", "-p", "question", "--no-parallel"]
    )

    assert result.exit_code == 0
    assert captured["options"].no_parallel is True


def test_cli_rank_reads_idea_file(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(cli, "_print_rank_results", lambda final: None)
    idea_file = tmp_path / "ideas.md"
    idea_file.write_text("# Ideas\n\nAlpha idea.\n\nBeta idea.\n", encoding="utf-8")
    captured: dict[str, Any] = {}

    def fake_rank_predefined_ideas(
        ideas: list[str],
        *,
        prompt: str,
        model: str,
        max_matches_per_idea: int,
        research: bool,
        deep_review: bool,
    ) -> ranking.CoScientistState:
        captured["ideas"] = ideas
        captured["prompt"] = prompt
        captured["model"] = model
        captured["max_matches_per_idea"] = max_matches_per_idea
        captured["research"] = research
        captured["deep_review"] = deep_review
        state = ranking.new_state(prompt, max_rounds=1)
        state["hypotheses"] = ranking.ideas_to_hypotheses(ideas)
        return state

    monkeypatch.setattr(ranking, "rank_predefined_ideas", fake_rank_predefined_ideas)

    result = CliRunner().invoke(
        cli.main,
        [
            "rank",
            "--idea-file",
            str(idea_file),
            "--model",
            "m",
            "--max-matches-per-idea",
            "2",
            "--research",
            "--deep-review",
        ],
    )

    assert result.exit_code == 0
    assert captured["ideas"] == ["Alpha idea.", "Beta idea."]
    assert captured["model"] == "m"
    assert captured["max_matches_per_idea"] == 2
    assert captured["research"] is True
    assert captured["deep_review"] is True


def test_run_idea_research_uses_agent_and_extracts_text(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class FakeAgent:
        def invoke(self, payload: dict) -> dict:
            captured["payload"] = payload
            return {"messages": [SimpleNamespace(content="brief text")]}

    def fake_create_deep_agent(**kwargs: object) -> FakeAgent:
        captured["kwargs"] = kwargs
        return FakeAgent()

    monkeypatch.setattr(idea_research, "_create_deep_agent", fake_create_deep_agent)

    text = idea_research.run_idea_research(
        question="Q",
        model="m",
        existing="existing",
        feedback="feedback",
    )

    assert text == "brief text"
    assert "kwargs" in captured
    payload = cast(dict[str, Any], captured["payload"])
    assert payload["messages"][0]["content"]


def test_run_deep_reflection_uses_agent_and_logs(monkeypatch) -> None:
    captured: dict[str, Any] = {}
    logged: list[str] = []

    class FakeAgent:
        def invoke(self, payload: dict) -> dict:
            captured["payload"] = payload
            return {"messages": [SimpleNamespace(content="critique text")]}

    def fake_create_deep_agent(**kwargs: object) -> FakeAgent:
        captured["kwargs"] = kwargs
        return FakeAgent()

    monkeypatch.setattr(deep_reflect, "_create_deep_agent", fake_create_deep_agent)
    monkeypatch.setattr(
        deep_reflect.logger,
        "info",
        lambda msg, *args: logged.append(msg.format(*args)),
    )

    text = deep_reflect.run_deep_reflection(
        question="Q",
        mechanism="M",
        prediction="P",
        experiment="E",
        context="C",
        feedback="F",
        model="m",
        subject="[h1] M",
    )

    assert text == "critique text"
    assert "kwargs" in captured
    payload = cast(dict[str, Any], captured["payload"])
    assert payload["messages"][0]["content"]
    assert logged[0] == "deep_reflect | start subject=[h1] M"


def test_package_main_entrypoint_executes(monkeypatch) -> None:
    called: dict[str, Any] = {}

    monkeypatch.setattr("saluki.cli.main", lambda: called.setdefault("ran", True))
    runpy.run_module("saluki.__main__", run_name="__main__")

    assert called["ran"] is True
