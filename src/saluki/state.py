from __future__ import annotations
from typing import TypedDict
import threading


class Hypothesis(TypedDict, total=False):
    id: str
    mechanism: str
    prediction: str
    experiment: str
    elo: float
    reflection: str
    verification: str
    novelty_score: float
    rigor_score: float
    testability_score: float
    impact_score: float
    simplicity_score: float
    generation: int
    decision: str
    for_whom: str
    reversal_condition: str
    not_doing: list[str]
    # Per the pseudocode AddToTournament step: a hypothesis is only assigned an
    # Elo (1200) once it enters the tournament.  Until then in_tournament is
    # False and elo is a sentinel.  ``matches`` counts tournament matches played.
    in_tournament: bool
    matches: int


def _upsert_hypotheses(
    existing: list[Hypothesis], update: list[Hypothesis]
) -> list[Hypothesis]:
    """Merge by id: new ids are appended, existing ids are updated in place."""
    by_id = {h["id"]: h for h in existing}
    for h in update:
        by_id[h["id"]] = h
    return list(by_id.values())


def lookup_similarity(graph: dict, h1_id: str, h2_id: str) -> float:
    """Return the precomputed similarity between two hypotheses (0.0 if unknown).

    *graph* maps a hypothesis id to a list of ``{"id", "similarity"}`` edges.
    """
    for edge in graph.get(h1_id, []):
        if edge["id"] == h2_id:
            return edge["similarity"]
    return 0.0


class SystemFeedback:
    """Stores system-wide cross-round feedback, periodically updated by GenerateSystemFeedback."""

    def __init__(self, feedback: str = "") -> None:
        self._feedback = feedback.strip()
        self._lock = threading.Lock()

    def update(self, feedback: str) -> None:
        with self._lock:
            self._feedback = feedback.strip()

    def get(self) -> str:
        with self._lock:
            return self._feedback

    def __bool__(self) -> bool:
        with self._lock:
            return bool(self._feedback)


class ProximityGraph:
    """Stores precomputed similarity edges between hypotheses.

    Written by proximity_node; queried by tournament pair selection and
    the evolve combine strategy.
    """

    def __init__(self, graph: dict[str, list[dict]] | None = None) -> None:
        self._graph: dict[str, list[dict]] = dict(graph) if graph else {}
        self._lock = threading.Lock()

    def update(self, graph: dict[str, list[dict]]) -> None:
        with self._lock:
            self._graph = dict(graph)

    def get_similarity(self, h1_id: str, h2_id: str) -> float:
        """Return the precomputed similarity between two hypotheses (0.0 if unknown)."""
        with self._lock:
            return lookup_similarity(self._graph, h1_id, h2_id)

    def get_all(self) -> dict[str, list[dict]]:
        """Return the full graph as a plain dict (safe copy)."""
        with self._lock:
            return dict(self._graph)

    def is_empty(self) -> bool:
        with self._lock:
            return not self._graph


class HypothesisStore:
    """Manages hypotheses, the reviews list, and tournament logs."""

    def __init__(
        self,
        hypotheses: list[Hypothesis] | None = None,
        tournament_log: list[dict] | None = None,
        reviews: list[dict] | None = None,
    ) -> None:
        self.hypotheses: list[Hypothesis] = hypotheses or []
        self.tournament_log: list[dict] = tournament_log or []
        # ReviewsList from the pseudocode: each entry is a FullReview or a
        # VerificationReview produced by the Reflection agent.
        self.reviews: list[dict] = reviews or []
        self._lock = threading.Lock()

    def get_all(self) -> list[Hypothesis]:
        with self._lock:
            return list(self.hypotheses)

    def get(self, h_id: str) -> Hypothesis | None:
        with self._lock:
            return next((h for h in self.hypotheses if h["id"] == h_id), None)

    def upsert(self, updates: list[Hypothesis]) -> None:
        with self._lock:
            self.hypotheses = _upsert_hypotheses(self.hypotheses, updates)

    def add_tournament_log(self, entries: list[dict]) -> None:
        with self._lock:
            self.tournament_log.extend(entries)

    def add_reviews(self, entries: list[dict]) -> None:
        with self._lock:
            self.reviews.extend(entries)


class CoScientistState(TypedDict):
    question: str
    research_plan: dict
    hypotheses: list[Hypothesis]
    idea_research: str
    literature: list[dict]
    round: int
    max_rounds: int
    tournament_log: list[dict]
    reviews: list[dict]
    meta_review: str
    system_wide_feedback: str
    final_overview: str
    proximity_graph: dict[str, list[dict]]
    pruned_ids: list[str]
    decision_mode: bool
    negative_space: bool
    deep_reflect: bool
    deep_reflection_cache: dict[str, str]
    verbose: bool


def new_state(
    question: str,
    *,
    max_rounds: int,
    research_plan: dict | None = None,
    decision_mode: bool = False,
    negative_space: bool = False,
    deep_reflect: bool = True,
    verbose: bool = False,
) -> CoScientistState:
    """Build a fresh CoScientistState with every field initialised to defaults."""
    return {
        "question": question,
        "research_plan": research_plan or {},
        "hypotheses": [],
        "idea_research": "",
        "literature": [],
        "round": 0,
        "max_rounds": max_rounds,
        "tournament_log": [],
        "reviews": [],
        "meta_review": "",
        "system_wide_feedback": "",
        "final_overview": "",
        "proximity_graph": {},
        "pruned_ids": [],
        "decision_mode": decision_mode,
        "negative_space": negative_space,
        "deep_reflect": deep_reflect,
        "deep_reflection_cache": {},
        "verbose": verbose,
    }


class GlobalState:
    """The shared state object used by the Supervisor and Agents.

    Structured storage:
    - store           → HypothesisStore   (hypotheses + tournament log)
    - proximity_graph → ProximityGraph    (similarity edges)
    - system_feedback → SystemFeedback    (cross-round feedback)
    All other fields are plain scalars.
    """

    def __init__(self, state_dict: CoScientistState) -> None:
        self.store = HypothesisStore(
            hypotheses=list(state_dict.get("hypotheses", [])),
            tournament_log=list(state_dict.get("tournament_log", [])),
            reviews=list(state_dict.get("reviews", [])),
        )
        self.proximity_graph = ProximityGraph(state_dict.get("proximity_graph") or {})
        self.system_feedback = SystemFeedback(
            state_dict.get("system_wide_feedback") or ""
        )

        self.question = state_dict["question"]
        self.research_plan = state_dict.get("research_plan", {})
        self.idea_research = state_dict.get("idea_research", "")
        self.literature = state_dict.get("literature", [])
        self.round = state_dict.get("round", 0)
        self.max_rounds = state_dict.get("max_rounds", 1)
        self.meta_review = state_dict.get("meta_review", "")
        self.final_overview = state_dict.get("final_overview", "")
        self.pruned_ids = state_dict.get("pruned_ids", [])
        self.decision_mode = state_dict.get("decision_mode", False)
        self.negative_space = state_dict.get("negative_space", False)
        self.deep_reflect = state_dict.get("deep_reflect", True)
        self.deep_reflection_cache = state_dict.get("deep_reflection_cache", {})
        self.verbose = state_dict.get("verbose", False)

    def to_dict(self) -> CoScientistState:
        """Serialise back to a plain dict for node functions and persistence."""
        return {
            "question": self.question,
            "research_plan": self.research_plan,
            "hypotheses": self.store.get_all(),
            "idea_research": self.idea_research,
            "literature": self.literature,
            "round": self.round,
            "max_rounds": self.max_rounds,
            "tournament_log": self.store.tournament_log,
            "reviews": self.store.reviews,
            "meta_review": self.meta_review,
            "system_wide_feedback": self.system_feedback.get(),
            "final_overview": self.final_overview,
            "proximity_graph": self.proximity_graph.get_all(),
            "pruned_ids": self.pruned_ids,
            "decision_mode": self.decision_mode,
            "negative_space": self.negative_space,
            "deep_reflect": self.deep_reflect,
            "deep_reflection_cache": self.deep_reflection_cache,
            "verbose": self.verbose,
        }
