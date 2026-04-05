MEMORY_QUERY_REPHRASE_SYSTEM_PROMPT = """
You are a retrieval query expansion specialist.
Generate alternative phrasings that improve retrieval recall while preserving the original intent.
Keep each variant concise and semantically faithful.
Do not introduce new facts.
Do not add interpretive labels or diagnostic framing not present in the original question.
"""

MEMORY_QUERY_REPHRASE_USER_TEMPLATE = """
Original question:
{question}

Generate {variant_count} diverse rewrites for retrieval.
"""

MEMORY_QUERY_RERANK_SYSTEM_PROMPT = """
You are an objective relevance analyst.
Select only items that provide direct evidence for the user's question.
Prefer precision over recall.
If evidence is weak or indirect, exclude it.
Return only item identifiers.
Do not prioritize items merely because they support an interpretation or label.
"""

MEMORY_QUERY_RERANK_USER_TEMPLATE = """
Question:
{question}

Candidate items:
{context_block}
"""

MEMORY_QUERY_SYNTHESIS_SYSTEM_PROMPT = """
You are a strict evidence reporter.
Primary goal: report what is directly supported by evidence, without interpretation.

Non-negotiable rules:
1. Use only the provided evidence.
2. Do not infer motives, traits, intentions, personality, diagnosis, or hidden causes.
3. Do not assign labels, categories, or named constructs to behavior patterns.
4. Do not coin terms or summarize behavior with abstract tags.
5. If a label appears in the question or evidence, you may quote it as a claim, but do not endorse it as fact.
6. Separate observed regularities from interpretation: report regularities as plain event sequences only.
7. If evidence is insufficient, state "Unknown" explicitly for that part.

Style requirements:
- Dry, neutral, analytic.
- Concrete and literal.
- No persuasive wording, no narrative embellishment, no moral framing.

Required output structure:
1) Direct Answer
2) Observed Regularities (unlabeled, plain-language)
3) Evidence
4) Unknowns
"""

MEMORY_QUERY_SYNTHESIS_USER_TEMPLATE = """
Question:
{question}

Evidence:
---
{narrative_context}
---

Write a concise report grounded only in this evidence.
When reporting regularities, describe only repeated sequences/events.
Do not attach named labels or conceptual categories to those regularities.
If any part cannot be supported directly, mark it as Unknown.
"""


def _normalize_section_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if total <= 0.0:
        raise Exception("At least one synthesis section weight must be > 0.")
    return {key: value / total for key, value in weights.items()}


def _detail_level_instruction(normalized_weight: float) -> str:
    if normalized_weight >= 0.45:
        return "primary focus: provide maximal detail and explicit grounding for this section"
    if normalized_weight >= 0.25:
        return "high focus: provide strong detail and clear grounding"
    if normalized_weight >= 0.12:
        return "medium focus: provide concise detail"
    if normalized_weight > 0.0:
        return "low focus: keep this section minimal"
    return "zero focus: output 'Unknown' unless the question explicitly requires this section"


def _section_word_budgets(normalized: dict[str, float], target_word_count: int) -> dict[str, int]:
    return {
        "direct_answer": int(round(target_word_count * normalized["direct_answer"])),
        "regularities": int(round(target_word_count * normalized["regularities"])),
        "evidence": int(round(target_word_count * normalized["evidence"])),
        "unknowns": int(round(target_word_count * normalized["unknowns"])),
    }


def _section_bullet_caps(normalized: dict[str, float]) -> dict[str, int]:
    regularities_cap = 6 if normalized["regularities"] >= 0.4 else 4 if normalized["regularities"] >= 0.2 else 2 if normalized["regularities"] > 0.0 else 0
    evidence_cap = 6 if normalized["evidence"] >= 0.4 else 4 if normalized["evidence"] >= 0.2 else 2 if normalized["evidence"] > 0.0 else 0
    unknowns_cap = 6 if normalized["unknowns"] >= 0.4 else 4 if normalized["unknowns"] >= 0.2 else 2 if normalized["unknowns"] > 0.0 else 0
    return {
        "regularities": regularities_cap,
        "evidence": evidence_cap,
        "unknowns": unknowns_cap,
    }


def build_memory_query_synthesis_prompts(
    question: str,
    narrative_context: str,
    body_and_situation: str,
    direct_answer_weight: float,
    regularities_weight: float,
    evidence_weight: float,
    unknowns_weight: float,
    target_word_count: int,
) -> tuple[str, str]:
    normalized = _normalize_section_weights(
        {
            "direct_answer": direct_answer_weight,
            "regularities": regularities_weight,
            "evidence": evidence_weight,
            "unknowns": unknowns_weight,
        }
    )
    budgets = _section_word_budgets(normalized, target_word_count)
    bullet_caps = _section_bullet_caps(normalized)
    min_total_words = int(round(target_word_count * 0.9))
    max_total_words = int(round(target_word_count * 1.1))

    dynamic_focus_block = f"""
Section weighting profile (normalized):
- Direct Answer: {normalized["direct_answer"]:.3f} -> {_detail_level_instruction(normalized["direct_answer"])}
- Observed Regularities: {normalized["regularities"]:.3f} -> {_detail_level_instruction(normalized["regularities"])}
- Evidence: {normalized["evidence"]:.3f} -> {_detail_level_instruction(normalized["evidence"])}
- Unknowns: {normalized["unknowns"]:.3f} -> {_detail_level_instruction(normalized["unknowns"])}

Hard output budget:
- Total response length: {min_total_words}-{max_total_words} words.
- Direct Answer: target ~{budgets["direct_answer"]} words, prose paragraphs only.
- Observed Regularities: target ~{budgets["regularities"]} words, max {bullet_caps["regularities"]} bullets.
- Evidence: target ~{budgets["evidence"]} words, max {bullet_caps["evidence"]} bullets.
- Unknowns: target ~{budgets["unknowns"]} words, max {bullet_caps["unknowns"]} bullets.

Zero-weight omission rule:
- If a section has a target of 0 words, omit that section entirely.

Priority rule:
- If budget conflicts occur, preserve Direct Answer first.
- Truncate Evidence before truncating Direct Answer.

Formatting rule:
- Use exact section headers when included:
  1) Direct Answer
  2) Observed Regularities
  3) Evidence
  4) Unknowns

Compliance rule:
- Before finalizing, verify section budgets and bullet caps.
- If any section exceeds its cap, rewrite to comply.

Comply with this weighting profile while preserving all non-negotiable anti-interpretation rules.
Do not override anti-interpretation rules even if a section has high weight.
"""

    capability_context = (
        "Architecture and capability grounding context:\n"
        f"{architecture_description_story}\n\n"
        f"{architecture_capability_addendum}\n\n"
        f"{body_and_situation}"
    )
    system_prompt = (
        MEMORY_QUERY_SYNTHESIS_SYSTEM_PROMPT
        + "\n\n"
        + capability_context
        + "\n\n"
        + dynamic_focus_block
    )
    user_prompt = MEMORY_QUERY_SYNTHESIS_USER_TEMPLATE.format(
        question=question,
        narrative_context=narrative_context,
    )
    return system_prompt, user_prompt
from pm.data.prompts.character_card_addendum import (
    architecture_description_story,
    architecture_capability_addendum,
)
