from __future__ import annotations
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Any, cast

import click
from dotenv import load_dotenv
from loguru import logger

from .schemas import DEFAULT_MODEL
from .state import CoScientistState, new_state
from .task_queue import restore_state, _DEFAULT_PERSIST_DIR

load_dotenv()


@dataclass(frozen=True)
class CliOptions:
    prompt: str
    model: str
    rounds: int
    verbose: bool
    decision_mode: bool
    negative_space: bool
    deep_reflect: bool
    base_url: str | None
    api_key: str | None
    no_task_queue: bool
    no_parallel: bool
    persist_dir: str | None
    pause: bool
    continue_from_pause: bool
    max_tasks: int | None
    max_ideas: int | None
    max_matches_per_idea: int | None


@dataclass(frozen=True)
class RunOptions:
    prompt: str
    rounds: int
    model: str
    base_url: str | None
    decision_mode: bool
    negative_space: bool
    deep_reflect: bool
    verbose: bool


@dataclass(frozen=True)
class SupervisorOptions:
    run: RunOptions
    persist_dir: str | None
    pause: bool = False
    resume_state: CoScientistState | None = None
    max_tasks: int | None = None
    max_ideas: int | None = None
    max_matches_per_idea: int | None = None
    no_parallel: bool = False


@dataclass(frozen=True)
class RankOptions:
    idea_file: Path
    prompt: str
    model: str
    verbose: bool
    base_url: str | None
    api_key: str | None
    max_matches_per_idea: int
    research: bool
    deep_review: bool


def _configure_logging(verbose: bool) -> None:
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}",
        colorize=True,
    )


def _require_api_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise click.ClickException(
            "OPENAI_API_KEY not set. Export it or add it to a .env file."
        )


def _print_results(final: CoScientistState) -> None:
    from .nodes import _active
    from .display import print_hypotheses

    click.echo("\n" + "=" * 70)
    click.echo("  RESULTS")
    click.echo("=" * 70)

    print_hypotheses(_active(final))

    if final.get("meta_review"):
        click.echo("\n" + "-" * 70)
        click.echo("Meta-review:")
        click.echo(final["meta_review"])

    if final.get("final_overview"):
        click.echo("\n" + "-" * 70)
        click.echo("Final research overview:")
        click.echo(final["final_overview"])

    click.echo("=" * 70)


def _print_rank_results(final: CoScientistState) -> None:
    from .ranking import ranked_hypotheses

    click.echo("\n" + "=" * 70)
    click.echo("  RANKED IDEAS")
    click.echo("=" * 70)

    for index, hypothesis in enumerate(ranked_hypotheses(final), 1):
        click.echo(
            f"\n#{index}  [{hypothesis['id']}]  "
            f"Elo: {hypothesis.get('elo', 0.0):.0f}  "
            f"Matches: {hypothesis.get('matches', 0)}"
        )
        click.echo(
            f"Scores — novelty: {hypothesis['novelty_score']:.2f}  "
            f"rigor: {hypothesis['rigor_score']:.2f}  "
            f"testability: {hypothesis['testability_score']:.2f}  "
            f"impact: {hypothesis['impact_score']:.2f}  "
            f"simplicity: {hypothesis['simplicity_score']:.2f}"
        )
        click.echo(hypothesis["mechanism"])

    click.echo("=" * 70)


@click.group()
def main() -> None:
    """Saluki command line interface."""


@main.command()
@click.option("--prompt", "-p", required=True, help="Research question to explore.")
@click.option(
    "--model",
    "-m",
    default=DEFAULT_MODEL,
    show_default=True,
    help="OpenAI model name (e.g. gpt-4o, gpt-4o-mini, o1-mini).",
)
@click.option(
    "--rounds",
    "-r",
    default=2,
    show_default=True,
    type=click.IntRange(1),
    help="Number of generate→evolve rounds.",
)
@click.option("--verbose", "-v", is_flag=True, help="DEBUG-level logs.")
@click.option(
    "--decision-mode/--no-decision-mode",
    default=False,
    show_default=True,
    help="Enable drafted-decision framing.",
)
@click.option(
    "--negative-space/--no-negative-space",
    default=False,
    show_default=True,
    help="Enable explicit negative-space constraints.",
)
@click.option(
    "--deep-reflect/--no-deep-reflect",
    default=True,
    show_default=True,
    help="Use Deep Agents for the reflection step.",
)
@click.option(
    "--base-url",
    default=None,
    show_default=False,
    help="OpenAI-compatible API base URL (e.g. http://localhost:8080/v1).",
)
@click.option(
    "--api-key",
    default=None,
    show_default=False,
    help="OpenAI API key (sk-...). Overrides OPENAI_API_KEY env var.",
)
@click.option(
    "--no-task-queue",
    is_flag=True,
    default=False,
    help="Use LangGraph pipeline instead of task-queue architecture (Supervisor).",
)
@click.option(
    "--no-parallel",
    is_flag=True,
    default=False,
    help="Execute task-queue work one task at a time instead of batching independent tasks.",
)
@click.option(
    "--persist-dir",
    default=None,
    show_default=_DEFAULT_PERSIST_DIR,
    help="Directory to persist session state after every step "
    f"(default: {_DEFAULT_PERSIST_DIR}).",
)
@click.option(
    "--pause",
    is_flag=True,
    default=False,
    help="Run to completion, then pause.  State is persisted "
    "so the run can be resumed later with --continue.",
)
@click.option(
    "--continue",
    "continue_from_pause",
    is_flag=True,
    default=False,
    help="Resume the most recently paused run from persist-dir.",
)
@click.option(
    "--max-tasks",
    "max_tasks",
    default=None,
    type=click.IntRange(1),
    help="Stop after executing this many individual tasks (overrides --rounds).",
)
@click.option(
    "--max-ideas",
    "max_ideas",
    default=None,
    type=click.IntRange(1),
    help="Target number of ideas to generate before finishing (pseudocode MaxIdeas).",
)
@click.option(
    "--max-matches-per-idea",
    "max_matches_per_idea",
    default=None,
    type=click.IntRange(1),
    help="Target tournament matches per idea before finishing (pseudocode MaxMatchesPerIdea).",
)
def generate(**kwargs: Any) -> None:
    """Saluki — AI Co-Scientist (LangGraph pipeline or task-queue supervisor)."""
    options = CliOptions(**kwargs)
    _run_cli(options)


@main.command()
@click.option(
    "--idea-file",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Markdown file containing blank-line-separated idea paragraphs.",
)
@click.option(
    "--prompt",
    "-p",
    default=None,
    show_default=False,
    help="Ranking goal or criteria. Defaults to a general quality ranking.",
)
@click.option(
    "--model",
    "-m",
    default=DEFAULT_MODEL,
    show_default=True,
    help="OpenAI model name (e.g. gpt-4o, gpt-4o-mini, o1-mini).",
)
@click.option("--verbose", "-v", is_flag=True, help="DEBUG-level logs.")
@click.option(
    "--base-url",
    default=None,
    show_default=False,
    help="OpenAI-compatible API base URL (e.g. http://localhost:8080/v1).",
)
@click.option(
    "--api-key",
    default=None,
    show_default=False,
    help="OpenAI API key (sk-...). Overrides OPENAI_API_KEY env var.",
)
@click.option(
    "--max-matches-per-idea",
    default=4,
    show_default=True,
    type=click.IntRange(1),
    help="Run tournament batches until every idea has at least this many matches.",
)
@click.option(
    "--research",
    is_flag=True,
    default=False,
    help="Run the shared bootstrap and idea-research setup before scoring.",
)
@click.option(
    "--deep-review",
    is_flag=True,
    default=False,
    help="Use the per-idea review path before ranking instead of lightweight scoring.",
)
def rank(**kwargs: Any) -> None:
    """Rank predefined ideas from a Markdown file with the Elo tournament."""
    from .ranking import DEFAULT_RANK_PROMPT

    if kwargs["prompt"] is None:
        kwargs["prompt"] = DEFAULT_RANK_PROMPT
    options = RankOptions(**kwargs)
    _run_rank_cli(options)


def _run_cli(options: CliOptions) -> None:
    _configure_logging(options.verbose)
    # Resolve the default at runtime (not decoration time) so it stays patchable.
    persist_dir = options.persist_dir or _DEFAULT_PERSIST_DIR
    if options.api_key:
        os.environ["OPENAI_API_KEY"] = options.api_key

    _require_api_key()

    if options.base_url:
        os.environ["OPENAI_BASE_URL"] = options.base_url

    logger.info(
        "starting | question='{}' model={} rounds={} max_tasks={} task_queue={} no_parallel={} decision_mode={} negative_space={} deep_reflect={}",
        options.prompt,
        options.model,
        options.rounds,
        options.max_tasks,
        not options.no_task_queue,
        options.no_parallel,
        options.decision_mode,
        options.negative_space,
        options.deep_reflect,
    )

    run_options = RunOptions(
        prompt=options.prompt,
        rounds=options.rounds,
        model=options.model,
        base_url=options.base_url,
        decision_mode=options.decision_mode,
        negative_space=options.negative_space,
        deep_reflect=options.deep_reflect,
        verbose=options.verbose,
    )

    if not options.no_task_queue:
        supervisor_options = _supervisor_options(
            options,
            run_options,
            persist_dir,
        )
        final = _run_with_supervisor(supervisor_options)
    else:
        final = _run_with_graph(run_options)

    logger.info(
        "done | active hypotheses={}",
        len(
            [
                h
                for h in final["hypotheses"]
                if h["id"] not in (final.get("pruned_ids") or [])
            ]
        ),
    )
    _print_results(final)


def _run_rank_cli(options: RankOptions) -> None:
    from .ranking import parse_markdown_ideas, rank_predefined_ideas

    _configure_logging(options.verbose)
    if options.api_key:
        os.environ["OPENAI_API_KEY"] = options.api_key

    _require_api_key()

    if options.base_url:
        os.environ["OPENAI_BASE_URL"] = options.base_url

    ideas = parse_markdown_ideas(options.idea_file.read_text(encoding="utf-8"))
    if len(ideas) < 2:
        raise click.ClickException("Need at least two idea paragraphs to rank.")

    logger.info(
        "ranking | ideas={} model={} max_matches_per_idea={} research={} deep_review={}",
        len(ideas),
        options.model,
        options.max_matches_per_idea,
        options.research,
        options.deep_review,
    )
    final = rank_predefined_ideas(
        ideas,
        prompt=options.prompt,
        model=options.model,
        max_matches_per_idea=options.max_matches_per_idea,
        research=options.research,
        deep_review=options.deep_review,
    )
    _print_rank_results(final)


def _supervisor_options(
    options: CliOptions,
    run_options: RunOptions,
    persist_dir: str,
) -> SupervisorOptions:
    if not options.continue_from_pause:
        return SupervisorOptions(
            run=run_options,
            persist_dir=persist_dir,
            pause=options.pause,
            max_tasks=options.max_tasks,
            max_ideas=options.max_ideas,
            max_matches_per_idea=options.max_matches_per_idea,
            no_parallel=options.no_parallel,
        )

    restored = restore_state(persist_dir)
    if restored is None:
        raise click.ClickException(
            f"No persisted state found in {persist_dir}. Run with --pause first."
        )
    restored_question = restored.get("question", "")
    logger.info(
        "resuming from persisted state | question='{}' round={} hypotheses={}",
        restored_question,
        restored["round"],
        len(restored["hypotheses"]),
    )
    if options.prompt and options.prompt != restored_question:
        logger.warning(
            "--continue ignores --prompt: resuming question '{}', NOT '{}'. "
            "Drop --continue to start a fresh run with your prompt.",
            restored_question,
            options.prompt,
        )
    return SupervisorOptions(
        run=run_options,
        persist_dir=persist_dir,
        resume_state=restored,
        max_tasks=options.max_tasks,
        max_ideas=options.max_ideas,
        max_matches_per_idea=options.max_matches_per_idea,
        no_parallel=options.no_parallel,
    )


def _run_with_supervisor(
    options: SupervisorOptions | None = None, **kwargs: Any
) -> CoScientistState:
    """Run using the task-queue Supervisor (paper architecture)."""
    from .task_queue import Supervisor

    if options is None:
        options = _supervisor_options_from_kwargs(kwargs)

    run = options.run
    supervisor = Supervisor(model=run.model, base_url=run.base_url)

    if options.resume_state is not None:
        return supervisor.resume_from(
            state=options.resume_state,
            persist_dir=options.persist_dir,
            rounds_override=run.rounds,
            max_ideas=options.max_ideas,
            max_matches_per_idea=options.max_matches_per_idea,
            no_parallel=options.no_parallel,
        )

    return supervisor.run(
        question=run.prompt,
        max_rounds=run.rounds,
        decision_mode=run.decision_mode,
        negative_space=run.negative_space,
        deep_reflect=run.deep_reflect,
        verbose=run.verbose,
        persist_dir=options.persist_dir,
        pause=options.pause,
        max_tasks=options.max_tasks,
        max_ideas=options.max_ideas,
        max_matches_per_idea=options.max_matches_per_idea,
        no_parallel=options.no_parallel,
    )


def _supervisor_options_from_kwargs(kwargs: dict[str, Any]) -> SupervisorOptions:
    run = RunOptions(
        prompt=kwargs["prompt"],
        rounds=kwargs["rounds"],
        model=kwargs["model"],
        base_url=kwargs["base_url"],
        decision_mode=kwargs["decision_mode"],
        negative_space=kwargs["negative_space"],
        deep_reflect=kwargs["deep_reflect"],
        verbose=kwargs["verbose"],
    )
    return SupervisorOptions(
        run=run,
        persist_dir=kwargs["persist_dir"],
        pause=kwargs.get("pause", False),
        resume_state=kwargs.get("resume_state"),
        max_tasks=kwargs.get("max_tasks"),
        max_ideas=kwargs.get("max_ideas"),
        max_matches_per_idea=kwargs.get("max_matches_per_idea"),
        no_parallel=kwargs.get("no_parallel", False),
    )


def _run_with_graph(options: RunOptions) -> CoScientistState:
    """Run using the LangGraph pipeline (original architecture)."""
    from .graph import build_graph

    graph = build_graph(model=options.model)

    initial_state = new_state(
        options.prompt,
        max_rounds=options.rounds,
        decision_mode=options.decision_mode,
        negative_space=options.negative_space,
        deep_reflect=options.deep_reflect,
        verbose=options.verbose,
    )

    try:
        return cast(CoScientistState, graph.invoke(initial_state))
    except Exception as exc:
        logger.exception("graph execution failed")
        raise click.ClickException(str(exc)) from exc
