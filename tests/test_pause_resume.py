"""Tests for 2.3 Pause/Resume capability."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from click.testing import CliRunner

from saluki.state import CoScientistState, Hypothesis
from saluki.task_queue import (
    Supervisor,
    _find_latest_state,
    _persist_state,
    restore_state,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
        "question": "test question",
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


def _write_state_file(persist_dir: str, filename: str, state: CoScientistState) -> str:
    """Write a state file in the expected directory structure."""
    # filename is like "2026-06-15-180000-bbbb2222"
    parts = filename.split("-")  # ["2026", "06", "15", "180000", ...]
    year = parts[0]
    month = parts[1]
    day = parts[2]
    target = Path(persist_dir) / year / month / day / f"{filename}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(state, f, default=str)
    return str(target)


# ---------------------------------------------------------------------------
# _find_latest_state
# ---------------------------------------------------------------------------


class TestFindLatestState:
    def test_returns_none_when_dir_missing(self, tmp_path: Path) -> None:
        assert _find_latest_state(str(tmp_path / "nonexistent")) is None

    def test_returns_none_when_no_json_files(self, tmp_path: Path) -> None:
        (tmp_path / "2026").mkdir()
        assert _find_latest_state(str(tmp_path)) is None

    def test_returns_most_recent_file(self, tmp_path: Path) -> None:
        state = _state()
        _write_state_file(str(tmp_path), "2026-01-01-120000-aaaa1111", state)
        _write_state_file(str(tmp_path), "2026-06-15-180000-bbbb2222", state)

        latest = _find_latest_state(str(tmp_path))
        assert latest is not None
        assert "2026-06-15-180000-bbbb2222" in latest

    def test_returns_none_for_non_json_files(self, tmp_path: Path) -> None:
        (tmp_path / "2026" / "06" / "14").mkdir(parents=True)
        (tmp_path / "2026" / "06" / "14" / "readme.txt").write_text("hello")
        assert _find_latest_state(str(tmp_path)) is None


# ---------------------------------------------------------------------------
# restore_state
# ---------------------------------------------------------------------------


class TestRestoreState:
    def test_returns_none_when_no_state(self, tmp_path: Path) -> None:
        assert restore_state(str(tmp_path)) is None

    def test_restores_valid_state(self, tmp_path: Path) -> None:
        state = _state(question="restored question", round=1)
        _write_state_file(str(tmp_path), "2026-06-14-120000-cccc3333", state)

        restored = restore_state(str(tmp_path))
        assert restored is not None
        assert restored["question"] == "restored question"
        assert restored["round"] == 1

    def test_returns_latest_file(self, tmp_path: Path) -> None:
        old_state = _state(question="old", round=0)
        new_state = _state(question="new", round=2)
        _write_state_file(str(tmp_path), "2026-01-01-120000-dddd4444", old_state)
        _write_state_file(str(tmp_path), "2026-06-15-180000-eeee5555", new_state)

        restored = restore_state(str(tmp_path))
        assert restored is not None
        assert restored["question"] == "new"
        assert restored["round"] == 2


# ---------------------------------------------------------------------------
# _persist_state
# ---------------------------------------------------------------------------


class TestPersistState:
    def test_creates_file_in_expected_structure(self, tmp_path: Path) -> None:
        state = _state(question="persisted")
        _persist_state(state, str(tmp_path))

        files = list(tmp_path.rglob("*.json"))
        assert len(files) == 1
        with open(files[0]) as f:
            loaded = json.load(f)
        assert loaded["question"] == "persisted"

    def test_noop_when_persist_dir_is_none(self) -> None:
        _persist_state(_state(), None)  # should not raise


# ---------------------------------------------------------------------------
# Supervisor.resume_from
# ---------------------------------------------------------------------------


class TestResumeFrom:
    def test_re_enters_at_bootstrap_when_no_hypotheses(self) -> None:
        state = _state()
        sup = Supervisor()
        # We can't actually run the full loop without an LLM,
        # but we can verify _next_task_for_state
        task = sup._next_task_for_state(state)
        assert task is not None
        assert task.agent == "bootstrap"
        assert task.action == "InitializeState"

    def test_re_enters_at_generate_when_idea_research_done_but_no_hypotheses(
        self,
    ) -> None:
        # Setup completed (idea_research persisted) but generation produced no
        # hypotheses: resume at generation, NOT bootstrap (which would re-run
        # the expensive idea-research step).
        state = _state(idea_research="a persisted research brief", hypotheses=[])
        sup = Supervisor()
        task = sup._next_task_for_state(state)
        assert task is not None
        assert task.agent == "generate"
        assert task.action == "CreateInitialHypotheses"

    def test_re_enters_at_review_for_unreviewed_hypotheses(self) -> None:
        # A hypothesis that has not yet entered the tournament still needs review.
        state = _state(
            hypotheses=[_hypothesis("h1", in_tournament=False, elo=0.0)],
            idea_research="some research",
        )
        sup = Supervisor()
        task = sup._next_task_for_state(state)
        assert task is not None
        assert task.agent == "reflect"
        assert task.action == "ReviewHypothesis"
        assert task.target_id == "h1"

    def test_re_enters_at_tournament_when_all_reviewed(self) -> None:
        state = _state(
            hypotheses=[_hypothesis("h1", in_tournament=True)],
            idea_research="some research",
        )
        sup = Supervisor()
        task = sup._next_task_for_state(state)
        assert task is not None
        assert task.agent == "ranking"
        assert task.action == "RunTournamentBatch"

    def test_re_enters_at_meta_review_when_tournament_exists(self) -> None:
        state = _state(
            hypotheses=[_hypothesis("h1", in_tournament=True)],
            idea_research="some research",
            tournament_log=[{"match": "A vs B"}],
        )
        sup = Supervisor()
        task = sup._next_task_for_state(state)
        assert task is not None
        assert task.agent == "metareview"
        assert task.action == "MetaReview"

    def test_re_enters_at_tournament_when_fully_progressed(self) -> None:
        state = _state(
            hypotheses=[_hypothesis("h1", in_tournament=True)],
            idea_research="some research",
            tournament_log=[{"match": "A vs B"}],
            meta_review="done",
        )
        sup = Supervisor()
        task = sup._next_task_for_state(state)
        assert task is not None
        assert task.agent == "ranking"
        assert task.action == "RunTournamentBatch"

    def test_re_enters_at_idea_research_when_no_literature(self) -> None:
        state = _state(
            hypotheses=[_hypothesis("h1")],
            idea_research="",
        )
        sup = Supervisor()
        task = sup._next_task_for_state(state)
        assert task is not None
        assert task.agent == "idea_research"


# ---------------------------------------------------------------------------
# CLI --pause and --continue flags
# ---------------------------------------------------------------------------


class TestCLIPauseContinue:
    def test_pause_defaults_to_default_persist_dir(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """--pause no longer requires --persist-dir; it defaults to the configured dir."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        import saluki.cli as cli

        # Redirect the default persist dir to a tmp dir so the test never writes
        # into the user's real ~/.saluki/states, and stub the supervisor so no
        # real API call is made.  The CLI argument parsing is what's under test.
        monkeypatch.setattr(cli, "_DEFAULT_PERSIST_DIR", str(tmp_path))
        monkeypatch.setattr(
            cli, "_run_with_supervisor", lambda **kw: _state(question="q")
        )

        result = CliRunner().invoke(cli.main, ["generate", "-p", "q", "--pause"])
        assert "--persist-dir is required" not in result.output

    def test_continue_defaults_to_default_persist_dir(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """--continue no longer requires --persist-dir; it defaults to the configured dir."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        import saluki.cli as cli

        # Redirect the default persist dir to an (empty) tmp dir: --continue should
        # report no state found rather than resuming a stray run from the real dir.
        monkeypatch.setattr(cli, "_DEFAULT_PERSIST_DIR", str(tmp_path))

        result = CliRunner().invoke(cli.main, ["generate", "-p", "q", "--continue"])
        assert "--persist-dir is required" not in result.output
        assert "No persisted state found" in result.output

    def test_continue_with_no_state_file(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        import saluki.cli as cli

        result = CliRunner().invoke(
            cli.main,
            ["generate", "-p", "q", "--continue", "--persist-dir", str(tmp_path)],
        )
        assert result.exit_code != 0
        assert "No persisted state found" in result.output
