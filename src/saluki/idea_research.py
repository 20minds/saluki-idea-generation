from __future__ import annotations

from textwrap import dedent

from .deepagents_helpers import _create_deep_agent, _run_agent


def run_idea_research(
    *,
    question: str,
    model: str,
    existing: str = "",
    feedback: str = "",
) -> str:
    """Use a Deep Agents harness to research the idea space for a prompt."""

    system_prompt = dedent(
        """
        You are an idea-research coordinator.

        Explore the idea space around the user's goal: identify promising directions, surface
        hidden assumptions, and synthesize a concise research brief that can guide hypothesis
        generation. Use the tivaly_search or exa_search tool to ground your brief in
        real literature when it helps, and cite what you find.

        Use planning and subagents if helpful. Prefer short, concrete, testable ideas over
        broad summaries. Focus on novelty, feasibility, and what would be worth testing next.
        """
    ).strip()

    subagents = [
        {
            "name": "explorer",
            "description": "Explores adjacent ideas, mechanisms, and testable angles.",
            "system_prompt": (
                "Generate a diverse set of candidate directions for the idea space. "
                "Look for adjacent mechanisms, plausible analogies, and concrete experiments."
            ),
        },
        {
            "name": "skeptic",
            "description": "Finds weak assumptions, failure modes, and missing pieces.",
            "system_prompt": (
                "Attack the candidate directions. Identify weak assumptions, confounds, "
                "false analogies, and the smallest experiments that would falsify them."
            ),
        },
    ]

    agent = _create_deep_agent(
        model=model,
        system_prompt=system_prompt,
        subagents=subagents,
    )

    prompt = dedent(
        f"""
        Research goal:
        {question}

        Existing ideas to avoid repeating:
        {existing or "None yet."}

        Recent feedback to fold in:
        {feedback or "None yet."}

        Produce a concise idea research brief with these sections:
        1. Research framing
        2. Promising directions
        3. Key assumptions
        4. Risks and failure modes
        5. Suggested next experiments
        6. One-sentence synthesis

        Use tivaly_search or exa_search where it sharpens the brief; keep the brief
        focused and concrete.
        """
    ).strip()

    return _run_agent(agent, prompt)
