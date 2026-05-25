RESEARCH_PLAN = """\
You are a research supervisor. Parse the scientist's research goal into a structured plan.

Research goal: {question}

Return a JSON object with exactly these fields and no other text:
{{
  "objective":     "<one sentence restating the core objective>",
  "sub_questions": ["<sub-question 1>", "..."],
  "constraints":   ["<constraint 1>", "..."],
  "keywords":      ["<keyword 1>", "..."]
}}

  objective     — one sentence restating the core objective
  sub_questions — 2–5 concrete sub-questions that decompose the objective
  constraints   — known constraints, scope boundaries, or success criteria (may be empty)
  keywords      — 3–6 search keywords useful for retrieving relevant literature
"""

GENERATE = """\
You are an expert researcher generating one novel, robust hypothesis.
The hypothesis must be atomic: exactly one causal mechanism, one prediction, and
one experiment. Do not combine multiple independent ideas; pick the strongest one.

Goal: {question}

Idea research brief:
{literature}

Previously explored (avoid repeating these):
{existing}

Recent meta-review feedback:
{feedback}

Return a single JSON object with exactly these fields and no other text. Do NOT
wrap it under an id key; do NOT return a list. Each value is exactly one sentence:
{{
  "mechanism":  "<the core insight or causal mechanism>",
  "prediction": "<the expected outcome if pursued>",
  "experiment": "<the simplest way to validate the hypothesis>"
}}
"""

GENERATE_DECISION = """\
You are an expert researcher generating one novel, robust hypothesis framed as a
drafted decision ready for action. The hypothesis must be atomic: exactly one
causal mechanism, one prediction, and one experiment. Do not combine multiple
independent ideas; pick the strongest one.

Goal: {question}

Idea research brief:
{literature}

Previously explored (avoid repeating these):
{existing}

Recent meta-review feedback:
{feedback}

Return a single JSON object with exactly these fields and no other text. Do NOT
wrap it under an id key; do NOT return a list. Each value is exactly one sentence:
{{
  "mechanism":  "<the core insight or reasoning behind the idea>",
  "prediction": "<the expected outcome if pursued>",
  "experiment": "<the simplest way to validate it>",
  "decision":   "<starts with a verb: the specific action to take, e.g. 'Launch a weekly digest for power users'>",
  "for_whom":   "<who this decision targets, their need, and why they benefit>",
  "reversal_condition": "<the measurable condition that would reverse this decision, e.g. 'Reverse if 30-day retention drops below 40%'>"
}}
"""

GENERATE_DEBATE = """\
You are simulating a scientific debate between domain experts to produce one hypothesis.

Goal: {question}

Idea research brief:
{literature}

Previously explored (avoid repeating these):
{existing}

Recent meta-review feedback:
{feedback}

Imagine three experts with different backgrounds debating how to attack this goal. They
challenge each other's assumptions, then converge on the single strongest novel hypothesis
that survives the debate. Report only that consensus hypothesis.

The hypothesis must be atomic: exactly one causal mechanism, one prediction, one experiment.

Return a single JSON object with exactly these fields and no other text. Do NOT
wrap it under an id key; do NOT return a list. Each value is exactly one sentence:
{{
  "mechanism":  "<the core insight the debate converged on>",
  "prediction": "<the expected outcome if pursued>",
  "experiment": "<the simplest way to validate it>"
}}
"""

REFLECT = """\
You are an expert in scientific hypothesis evaluation.
Critically evaluate the relationship between the hypothesis and the provided context.
Determine whether the hypothesis offers a novel causal explanation for relevant observations,
whether better explanations already exist, or whether the context contradicts it.
Also determine whether the hypothesis is atomic or improperly combines several independent ideas.

Hypothesis:
  Mechanism: {mechanism}
  Prediction: {prediction}
  Experiment: {experiment}

Idea research brief:
{literature}

Recent meta-review feedback:
{feedback}

Consider:
  - relevant observations or implications suggested by the context
  - whether the cause is already established
  - whether the hypothesis could plausibly explain the observations
  - logical consistency and mechanistic soundness
  - novelty relative to the provided context
  - whether the hypothesis contains multiple independent mechanisms that should be split
  - feasibility of the proposed experiment
  - potential confounds or missing assumptions
  - whether the hypothesis is already explained, a missing piece, neutral, or disproved

Return a concise paragraph of constructive critique. If possible, end with a brief
judgment using one of: already explained, other explanations more likely, missing piece,
neutral, disproved, or composite.
"""

SCORE = """\
Score the hypothesis below on five dimensions (0.0–1.0 each).

{hypotheses}

Idea research brief:
{literature}

Recent meta-review feedback:
{feedback}

Dimensions:
  novelty_score     — how original relative to existing work (0 = well established, 1 = genuinely new)
  rigor_score       — logical consistency and mechanistic soundness (0 = speculative, 1 = tight causal chain)
  testability_score — how feasible and concrete the experiment is (0 = untestable, 1 = clear protocol)
  impact_score      — significance if confirmed (0 = incremental, 1 = field-changing)
  simplicity_score  — how simple the idea is to explain and execute (0 = complex, 1 = single clear action);
                       atomic hypotheses should score higher than composite ones

Return a single JSON object with exactly these fields and no other text:
{{
  "novelty_score":     <float 0.0-1.0>,
  "rigor_score":       <float 0.0-1.0>,
  "testability_score": <float 0.0-1.0>,
  "impact_score":      <float 0.0-1.0>,
  "simplicity_score":  <float 0.0-1.0>,
  "reason":            "<one sentence covering all five dimensions>"
}}
"""

COMPARE = """\
Simulate a debate between two scientists: one defends Hypothesis A, the other defends
Hypothesis B. Each argues for their hypothesis and rebuts the other, weighing the goal,
the criteria, and the reviews. Conclude by declaring which hypothesis is superior.

Goal: {goal}

Criteria:
{criteria}

Recent meta-review feedback:
{feedback}

Hypothesis A:
  Mechanism: {mech_a}
  Prediction: {pred_a}
  Reflection: {refl_a}
  Scores — novelty: {nov_a:.2f}  rigor: {rig_a:.2f}  testability: {test_a:.2f}  impact: {imp_a:.2f}  simplicity: {simp_a:.2f}
{decision_section_a}{not_doing_section_a}

Hypothesis B:
  Mechanism: {mech_b}
  Prediction: {pred_b}
  Reflection: {refl_b}
  Scores — novelty: {nov_b:.2f}  rigor: {rig_b:.2f}  testability: {test_b:.2f}  impact: {imp_b:.2f}  simplicity: {simp_b:.2f}
{decision_section_b}{not_doing_section_b}

Weigh correctness, novelty, feasibility, and testability, then declare the winner.
Return a single JSON object with exactly these fields and no other text:
{{
  "winner": "<'A' or 'B'>",
  "reason": "<one-sentence justification for the verdict>"
}}
"""

VERIFY = """\
You are an expert scientific reviewer performing deep verification.
Break the hypothesis below into its core assumptions, then judge whether each assumption
is scientifically plausible.

Hypothesis:
  Mechanism: {mechanism}
  Prediction: {prediction}
  Experiment: {experiment}

Evidence / context:
{literature}

Return a single JSON object with exactly these fields and no other text:
{{
  "assumptions": [
    {{"assumption": "<one sentence>", "plausible": <true|false>, "note": "<one-sentence note>"}}
  ],
  "verdict": "<'sound', 'questionable', or 'implausible'>",
  "summary": "<one-sentence synthesis of the verification>"
}}
"""

FINAL_OVERVIEW = """\
You are an expert research supervisor writing the final research overview for a scientist.

Research objective:
{objective}

Top hypotheses (ranked by Elo):
{top_hypotheses}

Cross-round system feedback:
{system_feedback}

Synthesize the top-ranked hypotheses into a single coherent research overview. Outline the
main research directions, justify why each is promising, note how they relate, and recommend
concrete next steps. Write 6–10 sentences in plain prose. No headers.
"""

EVOLVE = """\
You are an expert researcher tasked with improving an existing hypothesis.
Combine insights from two parent hypotheses to produce one improved child hypothesis.

Parent 1:
  Mechanism: {mech_1}
  Prediction: {pred_1}

Parent 2:
  Mechanism: {mech_2}
  Prediction: {pred_2}

Recent meta-review feedback:
{feedback}

Tournament context:
{notes}

Synthesize a novel child that takes the best elements of both and improves feasibility,
specificity, and coherence.

Return a single JSON object with exactly these fields and no other text. Each value
is one sentence:
{{
  "mechanism":  "<the core insight or causal mechanism>",
  "prediction": "<the expected outcome if pursued>",
  "experiment": "<the simplest way to validate it>"
}}
"""

EVOLVE_DECISION = """\
You are an expert researcher tasked with improving an existing hypothesis.
Combine insights from two parent hypotheses to produce one improved child hypothesis
as a drafted decision.

Parent 1:
  Mechanism: {mech_1}  |  Decision: {dec_1}
  For whom: {whom_1}   |  Reversal: {rev_1}

Parent 2:
  Mechanism: {mech_2}  |  Decision: {dec_2}
  For whom: {whom_2}   |  Reversal: {rev_2}

Recent meta-review feedback:
{feedback}

Tournament context:
{notes}

Synthesize a novel child that sharpens the action and tightens the reversal condition
while improving feasibility and coherence.

Return a single JSON object with exactly these fields and no other text. Each value
is one sentence:
{{
  "mechanism":  "<the core insight or causal mechanism>",
  "prediction": "<the expected outcome if pursued>",
  "experiment": "<the simplest way to validate it>",
  "decision":   "<starts with a verb: the specific action to take>",
  "for_whom":   "<who this decision targets and why they benefit>",
  "reversal_condition": "<the measurable condition that would reverse this decision>"
}}
"""

EVOLVE_SIMPLIFY = """\
You are an expert researcher. Take the following hypothesis and simplify it —
make it more focused, more testable, and easier to execute without losing the
core insight. Eliminate ambiguity and unnecessary complexity.

Hypothesis:
  Mechanism: {mech}
  Prediction: {pred}
  Experiment: {exp}

Recent meta-review feedback:
{feedback}

Return the simplified hypothesis as a single JSON object with exactly these fields
and no other text. Each value is one sentence:
{{
  "mechanism":  "<the core insight or causal mechanism>",
  "prediction": "<the expected outcome if pursued>",
  "experiment": "<the simplest way to validate it>"
}}
"""

EVOLVE_SIMPLIFY_DECISION = """\
You are an expert researcher. Take the following decision hypothesis and simplify it —
sharpen the action, clarify the target audience, and make the reversal condition
more measurable. Eliminate ambiguity without losing the core insight.

Hypothesis:
  Mechanism: {mech}  |  Decision: {dec}
  For whom: {whom}   |  Reversal: {rev}
  Prediction: {pred}

Recent meta-review feedback:
{feedback}

Return the simplified hypothesis as a single JSON object with exactly these fields
and no other text. Each value is one sentence:
{{
  "mechanism":  "<the core insight or causal mechanism>",
  "prediction": "<the expected outcome if pursued>",
  "experiment": "<the simplest way to validate it>",
  "decision":   "<starts with a verb: the specific action to take>",
  "for_whom":   "<who this decision targets and why they benefit>",
  "reversal_condition": "<the measurable condition that would reverse this decision>"
}}
"""

EVOLVE_DIVERGE = """\
You are a creative research scientist. The following hypotheses represent the
current best thinking on this problem. Your task is to propose a fundamentally
different approach — not an incremental improvement, but a completely new angle
that has not yet been explored.

Current top hypotheses:
{hypotheses}

Recent meta-review feedback:
{feedback}

Tournament context:
{notes}

Propose a bold, unconventional hypothesis that attacks the problem from a
completely different direction.

Return a single JSON object with exactly these fields and no other text. Each value
is one sentence:
{{
  "mechanism":  "<the core insight or causal mechanism>",
  "prediction": "<the expected outcome if pursued>",
  "experiment": "<the simplest way to validate it>"
}}
"""

EVOLVE_DIVERGE_DECISION = """\
You are a creative research scientist. The following decisions represent the
current best thinking on this problem. Your task is to propose a fundamentally
different action — not an incremental improvement, but a completely different strategy.

Current top decisions:
{hypotheses}

Recent meta-review feedback:
{feedback}

Tournament context:
{notes}

Propose a bold, unconventional decision that takes a completely different angle.

Return a single JSON object with exactly these fields and no other text. Each value
is one sentence:
{{
  "mechanism":  "<the core insight or causal mechanism>",
  "prediction": "<the expected outcome if pursued>",
  "experiment": "<the simplest way to validate it>",
  "decision":   "<starts with a verb: the specific action to take>",
  "for_whom":   "<who this decision targets and why they benefit>",
  "reversal_condition": "<the measurable condition that would reverse this decision>"
}}
"""

NEGATIVE_SPACE_ADDON = """
Also add a "not_doing" field to the same JSON object: an array of 2–4 strings,
each a deliberate exclusion (something this idea explicitly will NOT do, and why).
Format each string exactly as: "Will not [action] — [one-sentence reason]".
Focus on boundary decisions that reduce scope ambiguity and pre-empt common
misunderstandings or stakeholder resistance during adoption.
  Bad example:  "Will not scale to enterprise — too complex"
  Good example: "Will not support SSO on launch — adds 6-week implementation overhead that delays learning"
"""

META_REVIEW = """\
You are an expert in scientific research and meta-analysis.
Summarize the state of hypothesis exploration after round {round}.
Prefer atomic hypotheses over composite ones, and call out any repeated tendency to
bundle multiple ideas into a single proposal.

Top hypotheses by Elo score:
{top_hypotheses}

Tournament results this round:
{tournament_summary}

Write a 3–5 sentence meta-review: what patterns emerged, which directions look promising,
what recurring weaknesses appeared, whether ideas are becoming too composite, and
what gaps remain. No headers.
"""

SYSTEM_FEEDBACK = """\
You are an expert research supervisor analyzing the progress of a multi-round
hypothesis generation and evaluation system.

Your task is to produce a concise, actionable system-wide feedback summary
that will guide future hypothesis generation and reflection steps.

Round meta-reviews:
{meta_reviews}

Tournament transcripts (all rounds):
{tournament_transcripts}

Current hypothesis pool (top 10 by Elo):
{top_hypotheses}

Analyze the full history and provide:
  1. Recurring weaknesses across rounds (e.g., hypotheses are too composite,
     experiments lack specificity, novelty is low)
  2. Promising directions that have emerged
  3. Gaps in exploration (topics, mechanisms, or approaches not yet pursued)
  4. One or two concrete suggestions for the next round of generation

Write 4-6 sentences. No headers. Be specific and actionable.
"""
