MEMORY_CONSOLIDATION_SHARED_CONTEXT_TEMPLATE = (
    "This content comes from interactions between {user_name} and {companion_name}. "
    "Treat the text as evidence records and preserve literal meaning."
)


NARRATIVE_REFINEMENT_SYSTEM_PROMPT = """
You refine long-horizon narrative memory using strict evidence discipline.

Mission:
- Update the target narrative using only supplied prior narrative text and new evidence.
- Keep the result neutral, analytic, and grounded in explicit observations.
- Avoid interpretation drift.

Hard rules:
1) Use only information present in the input evidence and prior narrative.
2) Do not infer hidden motives, diagnoses, personality labels, or symbolic interpretations.
3) Do not exaggerate certainty. If evidence is limited, keep statements narrow.
4) Do not use bullet points, numbered lists, markdown, headings, or JSON.
5) Output exactly one plain paragraph in natural prose.
6) Preserve continuity with prior narrative when evidence does not justify change.
7) Prefer concrete event-grounded wording over abstract judgments.
8) Keep names, roles, and chronology consistent with evidence.
9) Never invent missing timeline details.
10) Keep language concise and literal so downstream systems are not contaminated by framing.
"""


def build_narrative_refinement_user_prompt(
    target_name: str,
    narrative_type_name: str,
    prompt_goal: str,
    previous_content: str,
    feature_context: str,
) -> str:
    return f"""
Target: {target_name}
Narrative type: {narrative_type_name}
Goal: {prompt_goal}

Previous narrative:
---
{previous_content}
---

New evidence:
---
{feature_context}
---

Task:
Produce one updated narrative paragraph for {target_name} that is faithful to the evidence and the stated goal.
Prioritize factual continuity and minimal justified changes.
Do not output formatting or labels.
"""
