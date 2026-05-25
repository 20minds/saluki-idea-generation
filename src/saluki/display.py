from __future__ import annotations
import click

from .state import Hypothesis


def _print_header(header: str) -> None:
    click.echo("\n" + "─" * 70)
    click.echo(f"  {header}")
    click.echo("─" * 70)


def _print_decision_block(h: Hypothesis) -> None:
    click.echo(f"  Decision:   {h['decision']}")
    click.echo(f"  For whom:   {h['for_whom']}")
    click.echo(f"  Reverse if: {h['reversal_condition']}")


def _print_not_doing_block(h: Hypothesis) -> None:
    click.echo("  Will not:")
    for nd in h["not_doing"]:
        click.echo(f"    ✗ {nd}")


def _print_critique(h: Hypothesis) -> None:
    critique = h["reflection"]
    click.echo(f"  Critique:   {critique[:120]}{'…' if len(critique) > 120 else ''}")


def _print_hypothesis(h: Hypothesis, index: int) -> None:
    click.echo(f"\n#{index}  [{h['id']}]  Elo: {h['elo']:.0f}")
    click.echo(
        f"  Scores — novelty: {h['novelty_score']:.2f}  "
        f"rigor: {h['rigor_score']:.2f}  "
        f"testability: {h['testability_score']:.2f}  "
        f"impact: {h['impact_score']:.2f}  "
        f"simplicity: {h['simplicity_score']:.2f}"
    )
    if h.get("decision"):
        _print_decision_block(h)
    if h.get("not_doing"):
        _print_not_doing_block(h)
    if h.get("decision") or h.get("not_doing"):
        click.echo("  ---")
    click.echo(f"  Mechanism:  {h['mechanism']}")
    click.echo(f"  Prediction: {h['prediction']}")
    click.echo(f"  Experiment: {h['experiment']}")
    if h.get("reflection"):
        _print_critique(h)


def print_hypotheses(hypotheses: list[Hypothesis], header: str | None = None) -> None:
    """Print a ranked hypothesis list. Assumes hypotheses are already sorted."""
    if header:
        _print_header(header)

    for i, h in enumerate(hypotheses, 1):
        _print_hypothesis(h, i)
