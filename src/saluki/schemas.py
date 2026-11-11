from __future__ import annotations
import os
from typing import Optional
from pydantic import BaseModel, Field

# Default model used when no --model flag is provided. Decide's deployment-wide
# LLM_MODEL takes precedence; SALUKI_MODEL remains available for standalone use.
DEFAULT_MODEL = (
    os.environ.get("LLM_MODEL") or os.environ.get("SALUKI_MODEL") or "gpt-5.6-luna"
)


class GeneratedHypothesis(BaseModel):
    id: Optional[str] = Field(
        default=None,
        description="Optional short unique slug; ignored if provided—unique IDs are auto-generated",
    )
    mechanism: str = Field(
        description="One sentence: the core insight or causal mechanism"
    )
    prediction: str = Field(description="One sentence: what this predicts or expects")
    experiment: str = Field(
        description="One sentence: the simplest way to test or validate it"
    )
    decision: Optional[str] = Field(
        default="", description="One sentence: the specific action to take"
    )
    for_whom: Optional[str] = Field(
        default="", description="One sentence: who this targets and serves"
    )
    reversal_condition: Optional[str] = Field(
        default="",
        description="One sentence: what would cause us to reverse this decision",
    )
    not_doing: Optional[list[str]] = Field(
        default_factory=list,
        description="2–4 explicit statements of what this idea will NOT do, each formatted as 'Will not [action] — [reason]'",
    )


class HypothesisList(BaseModel):
    items: list[GeneratedHypothesis]


class ScoreResult(BaseModel):
    novelty_score: float = Field(ge=0.0, le=1.0)
    rigor_score: float = Field(ge=0.0, le=1.0)
    testability_score: float = Field(ge=0.0, le=1.0)
    impact_score: float = Field(ge=0.0, le=1.0)
    simplicity_score: float = Field(ge=0.0, le=1.0)
    reason: str = Field(description="One sentence covering all five dimensions")


class CompareResult(BaseModel):
    winner: str = Field(description="'A' or 'B'")
    reason: str = Field(description="One sentence justification")


class ResearchPlan(BaseModel):
    """Structured parse of the scientist's research goal (pseudocode: ResearchPlan)."""

    objective: str = Field(
        description="One sentence restating the core research objective"
    )
    sub_questions: list[str] = Field(
        default_factory=list,
        description="2–5 concrete sub-questions that decompose the objective",
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="Known constraints, scope boundaries, or success criteria",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="3–6 search keywords for literature retrieval",
    )


class Assumption(BaseModel):
    assumption: str = Field(description="One core assumption the hypothesis depends on")
    plausible: bool = Field(
        description="Whether the assumption is scientifically plausible"
    )
    note: str = Field(default="", description="One-sentence justification")


class VerificationResult(BaseModel):
    """Deep-verification review: decompose a hypothesis into assumptions and check each."""

    assumptions: list[Assumption] = Field(
        default_factory=list,
        description="The hypothesis broken into its core assumptions, each checked",
    )
    verdict: str = Field(
        default="neutral",
        description="Overall: 'sound', 'questionable', or 'implausible'",
    )
    summary: str = Field(
        default="", description="One-sentence synthesis of the verification"
    )


class EvolveResult(BaseModel):
    strategy: str = Field(
        default="combine",
        description="The strategy used: 'combine', 'simplify', or 'diverge'",
    )
    mechanism: str
    prediction: str
    experiment: str
    decision: Optional[str] = ""
    for_whom: Optional[str] = ""
    reversal_condition: Optional[str] = ""
    not_doing: Optional[list[str]] = Field(
        default_factory=list,
        description="2–4 explicit statements of what this idea will NOT do",
    )
