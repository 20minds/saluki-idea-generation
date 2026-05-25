from __future__ import annotations

from textwrap import dedent

from loguru import logger

from .deepagents_helpers import _create_deep_agent, _run_agent


def run_deep_reflection(
    *,
    question: str,
    mechanism: str,
    prediction: str,
    experiment: str,
    context: str,
    feedback: str,
    model: str,
    subject: str | None = None,
) -> str:
    """Use Deep Agents to produce a richer critique for a hypothesis."""

    system_prompt = dedent(
        """
        You are an expert scientific reviewer.

        Your job is to critique a single hypothesis using multiple focused checks.
        Delegate to subagents where helpful. Make sure you evaluate novelty, feasibility,
        contradictions, and whether the hypothesis is already explained or genuinely adds
        a missing piece.

        Keep the final answer concise and evidence-oriented. Use tivaly_search or
        exa_search to find real supporting or contradicting evidence; do not invent
        citations. Do not propose new hypotheses unless needed to explain a critique point.
        """
    ).strip()

    subagents = [
        {
            "name": "novelty-checker",
            "description": "Checks whether the idea is already known or meaningfully novel.",
            "system_prompt": (
                "Assess novelty against the provided context. Identify prior-art-like overlap, "
                "and decide whether the hypothesis seems already explained or still novel."
            ),
        },
        {
            "name": "feasibility-checker",
            "description": "Checks whether the experiment is practical and well scoped.",
            "system_prompt": (
                "Assess whether the proposed experiment is actually feasible, concrete, and "
                "free of obvious confounds or missing controls."
            ),
        },
        {
            "name": "contradiction-checker",
            "description": "Looks for contradictions or missing causal links.",
            "system_prompt": (
                "Look for observations or implications in the context that would contradict "
                "the hypothesis or make better explanations more likely."
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

        Hypothesis:
        Mechanism: {mechanism}
        Prediction: {prediction}
        Experiment: {experiment}

        Context:
        {context}

        Recent meta-review feedback:
        {feedback or "None yet."}

        Task:
        1. Use the subagents to check novelty, feasibility, and contradictions.
        1b. Check whether the hypothesis is atomic or improperly bundles several ideas.
        2. Synthesize one concise critique paragraph.
        3. End with exactly one of:
           - hypothesis: already explained
           - hypothesis: other explanations more likely
           - hypothesis: missing piece
           - hypothesis: neutral
           - hypothesis: disproved
           - hypothesis: composite
        """
    ).strip()

    reflection_subject = subject or mechanism
    logger.info("deep_reflect | start subject={:.180}", reflection_subject)
    text = _run_agent(agent, prompt)
    logger.info(
        "deep_reflect | done subject={:.80} chars={}", reflection_subject, len(text)
    )
    return text
