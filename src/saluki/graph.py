from __future__ import annotations
import warnings
from functools import partial
from typing import Any, cast

from langgraph.graph import StateGraph, START, END

from .state import CoScientistState
from .nodes import (
    bootstrap_node,
    idea_research_node,
    generate_node,
    reflect_node,
    score_node,
    proximity_node,
    tournament_node,
    evolve_node,
    meta_review_node,
    should_continue,
)


def build_graph(model: str) -> Any:
    """Build the LangGraph pipeline.

    .. deprecated::
        Use the task-queue architecture instead (pass ``--task-queue``
        to the CLI).  This function will be removed in a future release.
    """
    warnings.warn(
        "build_graph is deprecated. Use the task-queue architecture "
        "instead (pass --task-queue to the CLI).",
        DeprecationWarning,
        stacklevel=2,
    )
    bind = partial  # alias for clarity

    builder = StateGraph(cast(Any, CoScientistState))

    builder.add_node("bootstrap", bootstrap_node)
    builder.add_node("idea_research", bind(idea_research_node, model=model))
    builder.add_node("generate", bind(generate_node, model=model))
    builder.add_node("reflect", bind(reflect_node, model=model))
    builder.add_node("score", bind(score_node, model=model))
    builder.add_node("proximity", proximity_node)
    builder.add_node("tournament", bind(tournament_node, model=model))
    builder.add_node("evolve", bind(evolve_node, model=model))
    builder.add_node("meta_review", bind(meta_review_node, model=model))

    builder.add_edge(START, "bootstrap")
    builder.add_edge("bootstrap", "idea_research")
    builder.add_edge("idea_research", "generate")
    builder.add_edge("generate", "reflect")
    builder.add_edge("reflect", "score")
    builder.add_edge("score", "proximity")
    builder.add_edge("proximity", "tournament")
    builder.add_edge("tournament", "evolve")
    builder.add_edge("evolve", "meta_review")
    builder.add_conditional_edges(
        "meta_review",
        should_continue,
        {"generate": "generate", "end": END},
    )

    return builder.compile()
