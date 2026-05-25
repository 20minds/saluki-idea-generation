from __future__ import annotations

from typing import Any, cast

from saluki.nodes import Hypothesis
from saluki.display import print_hypotheses


def test_print_hypotheses_renders_expected_blocks(monkeypatch) -> None:
    lines: list[str] = []
    monkeypatch.setattr("saluki.display.click.echo", lambda msg="": lines.append(msg))

    data: dict[str, Any] = {
        "id": "h1",
        "elo": 1234.0,
        "novelty_score": 0.11,
        "rigor_score": 0.22,
        "testability_score": 0.33,
        "impact_score": 0.44,
        "simplicity_score": 0.55,
        "generation": 1,
        "decision": "Use the smallest viable experiment.",
        "for_whom": "bench scientists",
        "reversal_condition": "If controls fail.",
        "not_doing": ["Do not expand scope."],
        "mechanism": "Mechanism text",
        "prediction": "Prediction text",
        "experiment": "Experiment text",
        "reflection": "x" * 130,
    }

    print_hypotheses(
        [cast(Hypothesis, data)],
        header="Round 1 standings",
    )

    joined = "\n".join(lines)
    assert "Round 1 standings" in joined
    assert "Decision:" in joined
    assert "Will not:" in joined
    assert "Mechanism text" in joined
    assert "Critique:" in joined
