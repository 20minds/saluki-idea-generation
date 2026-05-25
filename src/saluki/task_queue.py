from __future__ import annotations
from typing import Any, cast
from dataclasses import dataclass, field
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json
import uuid
import glob
from datetime import datetime
from pathlib import Path

from loguru import logger

from .schemas import DEFAULT_MODEL
from .state import CoScientistState, GlobalState, Hypothesis, new_state


def _persist_state(state_dict: CoScientistState, persist_dir: str | None) -> None:
    if persist_dir is None:
        return

    now = datetime.now()
    path = (
        Path(persist_dir).expanduser()
        / now.strftime("%Y")
        / now.strftime("%m")
        / now.strftime("%d")
    )
    path.mkdir(parents=True, exist_ok=True)

    timestamp = now.strftime("%Y-%m-%d-%H%M%S")
    uuid_str = str(uuid.uuid4())
    filename = f"{timestamp}-{uuid_str}.json"

    with open(path / filename, "w") as f:
        json.dump(state_dict, f, indent=4)


def _find_latest_state(persist_dir: str | None) -> str | None:
    if persist_dir is None:
        return None

    path = Path(persist_dir).expanduser()
    if not path.exists():
        return None

    files = glob.glob(str(path / "**" / "*.json"), recursive=True)
    if not files:
        return None

    # Sort by filename (which starts with timestamp) to find most recent
    files.sort(key=lambda f: os.path.basename(f))
    return files[-1]


def restore_state(persist_dir: str | None) -> CoScientistState | None:
    latest_file = _find_latest_state(persist_dir)
    if latest_file is None:
        return None

    with open(latest_file, "r") as f:
        state_dict = json.load(f)

    return state_dict


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_PERSIST_DIR = os.path.expanduser("~/.saluki/states")

# ---------------------------------------------------------------------------
# Task & TaskQueue
# ---------------------------------------------------------------------------


@dataclass
class Task:
    """A unit of work to be executed by an agent."""

    agent: str
    action: str
    payload: dict[str, Any] = field(default_factory=dict)
    independent: bool = True
    target_id: str | None = None


class TaskQueue:
    """A FIFO queue for managing tasks."""

    def __init__(self) -> None:
        self._queue: deque[Task] = deque()

    def enqueue(self, task: Task) -> None:
        """Add a task to the end of the queue."""
        self._queue.append(task)

    def dequeue(self) -> Task | None:
        """Remove and return a task from the front of the queue."""
        try:
            return self._queue.popleft()
        except IndexError:
            return None

    def push_front(self, task: Task) -> None:
        """Add a task to the front of the queue."""
        self._queue.appendleft(task)

    def enqueue_priority(self, task: Task) -> None:
        """Enqueue a high-priority task that will run before all pending tasks."""
        self._queue.appendleft(task)

    def is_empty(self) -> bool:
        """Check if the queue is empty."""
        return len(self._queue) == 0

    def __len__(self) -> int:
        return len(self._queue)


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------


class Supervisor:
    """Central orchestrator that processes tasks from a queue.

    Matches the paper's StartCoScientist pseudocode:
        - Parses the research goal into a structured plan
        - Creates initial tasks and adds them to the queue
        - Main loop: dequeue → execute → manage follow-ups
        - DecideNextSteps when the queue drains
    """

    # Tuning constants for DecideNextSteps.
    SYSTEM_FEEDBACK_INTERVAL = 3
    PROXIMITY_INTERVAL = 2
    STALL_LIMIT = 2  # consecutive stalled checks before evolving
    ELO_STALL_EPS = 1.0  # top-Elo improvement below this counts as "stalled"
    # Backstop matching the pseudocode's "periodically... IF enough time has
    # passed" final-report step: after this many idle refinement cycles, emit the
    # final overview and stop even if the idea/match targets are never reached
    # (e.g. when the similarity filter starves some contenders of matches).
    MAX_IDLE_CYCLES = 10
    # Evolve whenever the queue drains while below the idea target.  Tournament
    # refinement still runs every idle cycle, but --max-ideas should primarily
    # drive continued generation rather than waiting through several tournament
    # batches between evolution steps.
    EVOLVE_INTERVAL = 1
    # In verbose mode, periodically print the current ranked hypotheses during
    # long task-queue runs.  The graph path prints via meta_review_node; the
    # task-queue path needs its own lightweight progress display.
    STANDINGS_INTERVAL = 3

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = model if model is not None else DEFAULT_MODEL
        self.base_url = base_url
        self.state: GlobalState | None = None
        self.queue: TaskQueue = TaskQueue()
        self._iterations: int = 0
        self._last_system_feedback_iteration: int = 0
        self._last_proximity_iteration: int = 0
        # Pseudocode loop bounds + progress tracking.
        self._max_ideas: int = 10
        self._max_matches_per_idea: int = 4
        self._num_ideas: int = 0
        self._evolve_cycles: int = 0
        self._last_top_elo: float | None = None
        self._stall: int = 0
        self._idle_cycles: int = 0
        self._idle_since_evolve: int = 0
        self._last_productive_ideas: int = 0
        self._final_overview_done: bool = False
        self._last_standings_iteration: int = 0

    # -- public API --------------------------------------------------------

    def resume_from(
        self,
        state_dict: CoScientistState | None = None,
        *,
        state: CoScientistState | None = None,
        persist_dir: str | None = None,
        rounds_override: int | None = None,
        max_ideas: int | None = None,
        max_matches_per_idea: int | None = None,
        no_parallel: bool = False,
    ) -> CoScientistState:
        """Resume a previously-paused run from *state_dict*.

        Determines where to re-enter the task queue based on what's
        already in the state, then runs the supervisor loop until
        completion.
        """
        if state_dict is None:
            state_dict = state
        if state_dict is None:
            raise ValueError("state_dict or state must be provided")
        self.state = GlobalState(state_dict)
        self.queue = TaskQueue()
        self._iterations = 0
        self._last_system_feedback_iteration = 0
        self._last_proximity_iteration = 0
        self._idle_cycles = 0
        self._idle_since_evolve = 0
        self._final_overview_done = False
        self._last_standings_iteration = 0
        self._num_ideas = len(self.state.store.get_all())
        self._last_productive_ideas = self._num_ideas
        # Honor an explicit --max-ideas; otherwise allow at least a little
        # headroom above the resumed idea count so evolution has room to run.
        if max_ideas is not None:
            self._max_ideas = max(max_ideas, self._num_ideas)
        else:
            self._max_ideas = max(self._max_ideas, self._num_ideas + 6)
        if max_matches_per_idea is not None:
            self._max_matches_per_idea = max_matches_per_idea

        if rounds_override is not None and rounds_override > self.state.max_rounds:
            self.state.max_rounds = rounds_override

        next_task = self._next_task_for_state(self.state)
        if next_task is not None:
            self.queue.enqueue(next_task)

        return self._run_loop(persist_dir, no_parallel=no_parallel)

    def run(
        self,
        question: str,
        max_rounds: int,
        *,
        decision_mode: bool = False,
        negative_space: bool = False,
        deep_reflect: bool = True,
        verbose: bool = False,
        persist_dir: str | None = None,
        pause: bool = False,
        max_tasks: int | None = None,
        max_ideas: int | None = None,
        max_matches_per_idea: int | None = None,
        no_parallel: bool = False,
    ) -> CoScientistState:
        """Run the full supervisor loop and return the final state.

        Per the pseudocode the loop is productive until ``NumIdeas`` and
        ``MatchesPerIdea`` targets are met, at which point a final research
        overview is generated.  ``max_rounds`` acts as a soft cap on the number
        of evolution cycles.  ``--max-tasks`` and ``--pause`` remain as
        additional stop controls.

        If *persist_dir* is provided, state is written after every step.
        """
        self._max_ideas = (
            max_ideas if max_ideas is not None else max(15, max_rounds * 15)
        )
        self._max_matches_per_idea = (
            max_matches_per_idea if max_matches_per_idea is not None else 4
        )

        # 1. Parse the research goal into a structured ResearchPlan *before* the
        #    loop and save it to shared memory (pseudocode: StartCoScientist).
        from .nodes import parse_research_plan

        research_plan = parse_research_plan(question, model=self.model)

        # 2. Initialize state with the parsed plan already in shared memory.
        self.state = GlobalState(
            new_state(
                question,
                max_rounds=max_rounds,
                research_plan=research_plan,
                decision_mode=decision_mode,
                negative_space=negative_space,
                deep_reflect=deep_reflect,
                verbose=verbose,
            )
        )

        # 3. Create initial task (bootstrap → idea_research → generation chain).
        self.queue.enqueue(Task(agent="bootstrap", action="InitializeState"))

        # 4. Main loop
        return self._run_loop(
            persist_dir,
            pause=pause,
            max_tasks=max_tasks,
            no_parallel=no_parallel,
        )

    def _run_loop(
        self,
        persist_dir: str | None,
        *,
        pause: bool = False,
        max_tasks: int | None = None,
        no_parallel: bool = False,
    ) -> CoScientistState:
        """Execute the supervisor main loop until the queue drains.

        When *pause* is ``True``, stop after the current round completes
        (i.e. when ``round`` increments and a new bootstrap task would be
        queued).
        When *max_tasks* is set, stop after that many individual task
        executions regardless of rounds remaining.
        """
        assert self.state is not None
        safety_limit = max(200, self._max_ideas * (self._max_matches_per_idea + 2) * 4)
        iterations = 0
        tasks_executed = 0
        while not self.queue.is_empty():
            tasks_executed += self._execute_ready_tasks(
                self._collect_ready_tasks(no_parallel=no_parallel)
            )
            self._enqueue_idle_tasks()

            iterations += 1
            self._iterations = iterations
            _persist_state(self.state.to_dict(), persist_dir)

            if self._should_stop_loop(
                pause=pause,
                max_tasks=max_tasks,
                tasks_executed=tasks_executed,
                iterations=iterations,
                safety_limit=safety_limit,
            ):
                break

        assert self.state is not None
        # Return a plain CoScientistState dict so callers (CLI, persistence) can
        # subscript the result.
        return self.state.to_dict()

    # -- internal ----------------------------------------------------------

    def _collect_ready_tasks(self, *, no_parallel: bool = False) -> list[Task]:
        tasks = []
        first_task = self.queue.dequeue()
        if first_task:
            tasks.append(first_task)

        if no_parallel:
            return tasks

        while not self.queue.is_empty():
            task = self.queue.dequeue()
            if task is None:
                break
            if not task.independent:
                self.queue.push_front(task)
                break
            tasks.append(task)
        return tasks

    def _execute_ready_tasks(self, tasks: list[Task]) -> int:
        if len(tasks) == 1:
            task = tasks[0]
            result = self._execute_task(task)
            self._merge_result(result)
            self._enqueue_follow_ups(task, result)
            return 1

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._execute_task_wrapper, task) for task in tasks
            ]
            for future in as_completed(futures):
                task, result = future.result()
                self._merge_result(result)
                self._enqueue_follow_ups(task, result)
        return len(tasks)

    def _enqueue_follow_ups(self, task: Task, result: dict) -> None:
        for follow_up in self._generate_follow_up_tasks(task, result):
            self.queue.enqueue(follow_up)

    def _enqueue_idle_tasks(self) -> None:
        if not self.queue.is_empty():
            return
        for task in self.decide_next_steps():
            self.queue.enqueue(task)

    def _should_stop_loop(
        self,
        *,
        pause: bool,
        max_tasks: int | None,
        tasks_executed: int,
        iterations: int,
        safety_limit: int,
    ) -> bool:
        assert self.state is not None
        if pause and self.state.store.tournament_log and not self._is_complete():
            logger.info(
                "supervisor | pause requested — stopping after first productive cycle"
            )
            return True
        if max_tasks is not None and tasks_executed >= max_tasks:
            logger.info(
                "supervisor | reached max_tasks limit (%d), stopping", max_tasks
            )
            return True
        if iterations > safety_limit:
            logger.warning(
                "supervisor | exceeded safety limit (%d), stopping", safety_limit
            )
            return True
        return False

    def _execute_task_wrapper(self, task: Task) -> tuple[Task, dict]:
        """Wrapper for parallel execution to return task and result."""
        result = self._execute_task(task)
        return task, result

    def _execute_task(self, task: Task) -> dict:
        """Look up the agent class and execute the task."""
        from .agents import AGENT_REGISTRY

        agent_cls = AGENT_REGISTRY.get(task.agent)
        if agent_cls is None:
            logger.warning("supervisor | unknown agent '%s', skipping", task.agent)
            return {}

        agent = agent_cls(model=task.payload.get("model", self.model))
        return agent.execute(task, self.state)  # type: ignore[arg-type]

    # Signal keys are consumed by ManageFollowUpTasks, not stored on state.
    _SIGNAL_KEYS = frozenset({"new_hypothesis_ids", "reviewed_id", "added_id"})

    def _merge_result(self, result: dict) -> None:
        """Merge a task result into the shared state.

        Structured fields are routed to their dedicated store objects:
        - hypotheses       → HypothesisStore.upsert  (merge by id)
        - tournament_log   → HypothesisStore.add_tournament_log  (append)
        - reviews          → HypothesisStore.add_reviews  (append)
        - proximity_graph  → ProximityGraph.update  (replace)
        - system_wide_feedback → SystemFeedback.update  (replace)
        Signal keys (``new_hypothesis_ids`` etc.) are skipped here.
        All other fields: direct attribute overwrite.
        """
        if not result:
            return
        assert self.state is not None

        # Count newly created ideas for the loop bound.
        new_ids = result.get("new_hypothesis_ids")
        if isinstance(new_ids, list):
            existing_ids = {h["id"] for h in self.state.store.get_all()}
            self._num_ideas += len(set(new_ids) - existing_ids)

        for key, value in result.items():
            if key in self._SIGNAL_KEYS:
                continue
            self._merge_state_value(key, value)

    def _merge_state_value(self, key: str, value: object) -> None:
        assert self.state is not None
        if key == "hypotheses" and isinstance(value, list):
            self.state.store.upsert(cast(list[Hypothesis], value))
            return
        if key == "tournament_log" and isinstance(value, list):
            self.state.store.add_tournament_log(cast(list[dict], value))
            return
        if key == "reviews" and isinstance(value, list):
            self.state.store.add_reviews(cast(list[dict], value))
            return
        if key == "proximity_graph" and isinstance(value, dict):
            self.state.proximity_graph.update(cast(dict[str, list[dict]], value))
            return
        if key == "system_wide_feedback" and isinstance(value, str):
            self.state.system_feedback.update(value)
            return
        setattr(self.state, key, value)  # type: ignore[attr-defined]

    def _generate_follow_up_tasks(
        self, completed_task: Task, result: dict
    ) -> list[Task]:
        """ManageFollowUpTasks from the pseudocode — result-signal driven.

        - A new hypothesis (``new_hypothesis_ids``) → ReviewHypothesis(TargetID)
        - A completed review (``reviewed_id``)      → AddToTournament(TargetID)
        Plus the fixed setup chain bootstrap → idea_research → generate.
        """
        agent = completed_task.agent
        tasks: list[Task] = []

        # Setup chain (one-time, at the start of a run).
        if agent == "bootstrap":
            return [Task(agent="idea_research", action="ResearchIdeaSpace")]
        if agent == "idea_research":
            return [Task(agent="generate", action="CreateInitialHypotheses")]

        # Result-type routing (applies to generation, evolution, reflection).
        for h_id in result.get("new_hypothesis_ids", []) or []:
            tasks.append(
                Task(agent="reflect", action="ReviewHypothesis", target_id=h_id)
            )

        reviewed_id = result.get("reviewed_id")
        if reviewed_id:
            tasks.append(
                Task(agent="ranking", action="AddToTournament", target_id=reviewed_id)
            )

        return tasks

    def _contenders(self) -> list[Hypothesis]:
        """Active hypotheses that have entered the tournament."""
        assert self.state is not None
        pruned = set(self.state.pruned_ids or [])
        return [
            h
            for h in self.state.store.get_all()
            if h["id"] not in pruned and h.get("in_tournament", False)
        ]

    def _active_hypotheses(self) -> list[Hypothesis]:
        """Active hypotheses sorted by current Elo."""
        assert self.state is not None
        pruned = set(self.state.pruned_ids or [])
        active = [h for h in self.state.store.get_all() if h["id"] not in pruned]
        return sorted(active, key=lambda h: h.get("elo", 0.0), reverse=True)

    def _maybe_print_standings(self) -> None:
        """Print current standings periodically during verbose queue runs."""
        assert self.state is not None
        if not self.state.verbose:
            return
        if (
            self._iterations - self._last_standings_iteration
        ) < self.STANDINGS_INTERVAL:
            return

        active = self._active_hypotheses()
        if not active:
            return

        from .display import print_hypotheses

        print_hypotheses(
            active,
            header=f"Current standings ({self._num_ideas}/{self._max_ideas} ideas)",
        )
        self._last_standings_iteration = self._iterations

    def _is_complete(self) -> bool:
        """Pseudocode loop guard: enough ideas AND enough matches per idea."""
        contenders = self._contenders()
        if not contenders:
            return False
        enough_ideas = self._num_ideas >= self._max_ideas
        enough_matches = (
            min(h.get("matches", 0) for h in contenders) >= self._max_matches_per_idea
        )
        # Honor both pseudocode targets before finalizing.  A run with
        # --max-ideas 100 should not stop merely because the current contenders
        # have reached the default match target.
        return enough_ideas and enough_matches

    def _stall_detected(self) -> bool:
        """True when top-hypothesis Elo has stopped meaningfully improving.

        Updates the stall counter as a side effect.  Note this can be fooled by
        a dominant hypothesis that keeps inflating its Elo against a weak field
        (it never looks "stalled"), which is why evolution also fires on a
        cadence in :meth:`_should_evolve`.
        """
        contenders = self._contenders()
        top = max(h.get("elo", 0.0) for h in contenders)
        if (
            self._last_top_elo is not None
            and (top - self._last_top_elo) <= self.ELO_STALL_EPS
        ):
            self._stall += 1
        else:
            self._stall = 0
        self._last_top_elo = top
        if self._stall >= self.STALL_LIMIT:
            self._stall = 0
            return True
        return False

    def _should_evolve(self) -> bool:
        """Decide whether to evolve top hypotheses this idle cycle.

        Evolves when quality has stalled OR periodically (every
        ``EVOLVE_INTERVAL`` idle cycles), as long as we're below the idea and
        round caps and have at least two contenders.  The periodic trigger is
        what keeps new ideas flowing when the Elo-stall signal never fires.
        """
        assert self.state is not None
        if self._evolve_cycles >= self.state.max_rounds:
            return False
        if self._num_ideas >= self._max_ideas:
            return False
        if len(self._contenders()) < 2:
            return False

        stalled = self._stall_detected()
        due = self._idle_since_evolve >= self.EVOLVE_INTERVAL
        if stalled or due:
            self._evolve_cycles += 1
            self._idle_since_evolve = 0
            return True
        return False

    def _next_task_for_state(self, state) -> Task | None:
        """Determine the next task to queue when resuming from *state*.

        Accepts either a plain dict (CoScientistState) or a GlobalState object.
        Returns the first task whose preconditions are not yet met, or ``None``
        if the run is already complete.
        """
        if isinstance(state, dict):
            hyps = state.get("hypotheses", [])
            idea_research = state.get("idea_research", "")
            tournament_log = state.get("tournament_log", [])
            meta_review = state.get("meta_review", "")
        else:
            hyps = state.store.get_all()
            idea_research = state.idea_research
            tournament_log = state.store.tournament_log
            meta_review = state.meta_review

        # Re-enter at the earliest setup step that has not completed.
        if not idea_research:
            # Setup incomplete.  With no hypotheses either, run the full setup
            # chain from bootstrap; otherwise just (re)run idea_research.
            if not hyps:
                return Task(agent="bootstrap", action="InitializeState")
            return Task(agent="idea_research", action="ResearchIdeaSpace")
        if not hyps:
            # Setup is done (idea_research present) but no hypotheses exist yet —
            # go straight to generation rather than restarting the setup chain
            # and re-running the expensive idea-research step.
            return Task(agent="generate", action="CreateInitialHypotheses")
        # Any hypothesis not yet in the tournament still needs reviewing.
        for h in hyps:
            if not h.get("in_tournament", False):
                return Task(
                    agent="reflect", action="ReviewHypothesis", target_id=h["id"]
                )
        if not tournament_log:
            return Task(agent="ranking", action="RunTournamentBatch")
        if not meta_review:
            return Task(agent="metareview", action="MetaReview")
        return Task(agent="ranking", action="RunTournamentBatch")

    def decide_next_steps(self) -> list[Task]:
        """DecideNextSteps from the pseudocode.

        Called when the queue drains.  Keeps the tournament running, refreshes
        the proximity graph, evolves top hypotheses when quality stalls,
        periodically synthesizes system feedback, and — once the idea/match
        targets are met — generates the final research overview and stops.
        """
        if self.state is None:
            return []

        # Terminal: emit the final overview exactly once, then let the queue drain.
        if self._final_overview_done:
            return []
        # Reset the idle backstop whenever the run is genuinely productive (new
        # ideas were added since the last check) so the limit only fires when
        # we're truly stuck, not while evolution is still generating ideas.
        if self._num_ideas > self._last_productive_ideas:
            self._last_productive_ideas = self._num_ideas
            self._idle_cycles = 0
        self._idle_cycles += 1
        self._idle_since_evolve += 1
        # Stop when the idea/match targets are met, or — as a backstop that
        # guarantees termination — after enough idle refinement cycles.
        if self._is_complete() or self._idle_cycles >= self.MAX_IDLE_CYCLES:
            self._final_overview_done = True
            reason = (
                "targets met"
                if self._is_complete()
                else f"idle limit reached ({self.MAX_IDLE_CYCLES} cycles)"
            )
            logger.info("supervisor | {} — generating final research overview", reason)
            return [Task(agent="metareview", action="GenerateFinalResearchOverview")]

        tasks: list[Task] = []

        self._maybe_print_standings()

        # Refresh the proximity graph periodically (informs pairing + evolution).
        if (
            self._iterations - self._last_proximity_iteration
        ) >= self.PROXIMITY_INTERVAL:
            tasks.append(Task(agent="proximity", action="UpdateProximityGraph"))
            self._last_proximity_iteration = self._iterations

        # Keep the tournament running to refine scores.
        tasks.append(Task(agent="ranking", action="RunTournamentBatch"))

        # Evolve top hypotheses when quality stalls or on a periodic cadence,
        # so new ideas keep flowing toward the idea target.
        if self._should_evolve():
            logger.info(
                "supervisor | evolving top hypotheses ({}/{} ideas)",
                self._num_ideas,
                self._max_ideas,
            )
            tasks.append(Task(agent="evolve", action="EvolveTopHypotheses"))

        # Periodically synthesize system-wide feedback.
        if (
            self._iterations - self._last_system_feedback_iteration
        ) >= self.SYSTEM_FEEDBACK_INTERVAL:
            tasks.append(Task(agent="metareview", action="GenerateSystemFeedback"))
            self._last_system_feedback_iteration = self._iterations

        return tasks
