"""Independent agents matching the pseudocode's agent roster.

Each agent owns one role from the pseudocode (Generation, Reflection, Ranking,
Evolution, Proximity, Meta-review) plus the two setup steps (Bootstrap, Idea
research).  Agents dispatch on ``task.action`` and operate on a specific
``task.target_id`` where the pseudocode routes per-hypothesis.

The agents are thin: they normalise the shared state to a plain dict and call
the corresponding node function in ``nodes.py``.  All store mutation happens in
the Supervisor's ``_merge_result``.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Callable, cast

from .state import CoScientistState
from .task_queue import Task
from .nodes import (
    bootstrap_node,
    idea_research_node,
    generate_node,
    review_hypothesis_node,
    add_to_tournament_node,
    tournament_node,
    evolve_node,
    proximity_node,
    meta_review_node,
    system_feedback_node,
    final_overview_node,
)

nodes = import_module("saluki.nodes")


# ---------------------------------------------------------------------------
# Agent base class
# ---------------------------------------------------------------------------


class Agent:
    """Base class for all agents.

    A trivial agent just runs one ``(state_dict, *, model)`` node function: set
    ``_node`` and the base ``execute`` dispatches to it.  Agents that branch on
    ``task.action`` or need ``task.target_id`` override ``execute`` instead.
    """

    _node: Callable[..., dict] | None = None

    def __init__(self, model: str) -> None:
        self.model = model

    def _state_dict(self, state: Any) -> CoScientistState:
        """Normalize state to a plain CoScientistState dict for node functions."""
        if isinstance(state, dict):
            return cast(CoScientistState, state)
        return state.to_dict()

    def execute(self, task: Task, state: Any) -> dict:
        _ = task
        if self._node is None:
            raise NotImplementedError
        return self._node(self._state_dict(state), model=self.model)


# ---------------------------------------------------------------------------
# Setup agents
# ---------------------------------------------------------------------------


class BootstrapAgent(Agent):
    _node = staticmethod(bootstrap_node)


class IdeaResearchAgent(Agent):
    _node = staticmethod(idea_research_node)


# ---------------------------------------------------------------------------
# Pseudocode agents
# ---------------------------------------------------------------------------


class GenerationAgent(Agent):
    """CreateInitialHypotheses (and per-round regeneration)."""

    _node = staticmethod(generate_node)


class ReflectionAgent(Agent):
    """ReviewHypothesis(TargetID): full review + deep verification."""

    def execute(self, task: Task, state: Any) -> dict:
        if not task.target_id:
            return {}
        return review_hypothesis_node(
            self._state_dict(state), task.target_id, model=self.model
        )


class RankingAgent(Agent):
    """AddToTournament(TargetID) and RunTournamentBatch."""

    def execute(self, task: Task, state: Any) -> dict:
        if task.action == "AddToTournament":
            if not task.target_id:
                return {}
            return add_to_tournament_node(self._state_dict(state), task.target_id)
        # RunTournamentBatch / RunTournament
        return tournament_node(self._state_dict(state), model=self.model)


class EvolutionAgent(Agent):
    """EvolveTopHypotheses."""

    _node = staticmethod(evolve_node)


class ProximityAgent(Agent):
    """UpdateProximityGraph."""

    def execute(self, task: Task, state: Any) -> dict:
        return proximity_node(self._state_dict(state))


class MetareviewAgent(Agent):
    """MetaReview, GenerateSystemFeedback, GenerateFinalResearchOverview."""

    def execute(self, task: Task, state: Any) -> dict:
        if task.action == "GenerateSystemFeedback":
            return system_feedback_node(self._state_dict(state), model=self.model)
        if task.action == "GenerateFinalResearchOverview":
            return final_overview_node(self._state_dict(state), model=self.model)
        return meta_review_node(self._state_dict(state), model=self.model)


# ---------------------------------------------------------------------------
# Registry: maps Task.agent string → Agent class
# ---------------------------------------------------------------------------

AGENT_REGISTRY: dict[str, type[Agent]] = {
    "bootstrap": BootstrapAgent,
    "idea_research": IdeaResearchAgent,
    "generate": GenerationAgent,
    "reflect": ReflectionAgent,
    "ranking": RankingAgent,
    "evolve": EvolutionAgent,
    "proximity": ProximityAgent,
    "metareview": MetareviewAgent,
}
