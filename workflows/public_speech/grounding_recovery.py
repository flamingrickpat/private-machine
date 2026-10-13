"""One bounded uncertainty response after rejected public grounding.

The rejected candidate remains unchanged. This handler receives original
reports instead of repeating unsupported prose or older voice examples.
"""
from __future__ import annotations

from workflows.public_speech.claim_review import review_public_claims
from workflows.public_speech.public_context import Presentation, PublicContext, render_public_context


async def recover_grounding(complete, context: PublicContext) -> dict:
    """Generate and review one scoped response. Never repeat a failed recovery.

    The caller retains the original rejection and labels recovered publication.
    Protocol errors propagate to the caller's explicit failure response.
    """
    current = [item for item in context.evidence if item["knoxel_id"] == context.user_frontier
               and item.get("evidence_role") == "user_report"]
    if not current:
        raise ValueError("Grounding recovery requires the current original user report.")
    presentation = context.presentation if context.presentation.mode == "withhold" else Presentation()
    source_use = context.capabilities.get("evidence_limit", {}).get("source_use", "event")
    reports = current if source_use == "general" else [
        item for item in context.evidence if item.get("evidence_role") in
        {"user_report", "runtime_event", "world_observation"}]
    scoped = context.model_copy(deep=True, update={
        "evidence": reports,
        "memories": [], "goals": [], "workspace": "", "selected_action": "",
        "presentation": presentation, "action_schema_id": None,
    })
    candidate = await complete([
        {"role": "system", "content": "Write one bounded public uncertainty reply to the current user report. "
         "Use one short paragraph of two to four sentences. "
         "Use the character only for voice and accepted affect only for mood. The evidence defines what is known. "
         "State what the original reports establish and what cannot be established. Ask one useful clarification if needed. "
         "Answer the current question only. Earlier reports on a nearby topic do not answer a new advice question. "
         "Describe a user's secondhand statement as their report, not independent verification of its contents. "
         "Use older reports for history, retained names, preferences, or plans; preserve dates and uncertainty. "
         "For historical recall, name the date and say the user reported, followed by a verbatim complete source sentence. "
         "Do not change its pronouns, actor, object, or uncertainty. Do not reconstruct events or explain their causes. "
         "The dated original report is the supporting conversation. When it exists, do not claim that a missing thread ID "
         "or more details prevent identifying the conversation. Ask for clarification only if needed to answer the request. "
         "When a dated report answers a historical recall request, stop after identifying that report. "
         "Do not append a question based on an adjacent report or turn a tentative source statement into a definite premise. "
         "Do not add technical explanations, treatments, or unverified current plant/room/world conditions. "
         "Make selections and plans in text. Do not claim to perform physical actions such as serving a drink. "
         "The evidence is a partial memory snapshot. Do not claim that memory is absent or that records disappeared. "
         "Identify one supporting original report, not the only record in memory. Never assert an exhaustive record count. "
         "Describe what the user reported. Do not say all I have or only report; the supplied excerpt is not an inventory. "
         "Retain unsure as unsure; do not assign an inferred attitude or independently confirm a secondhand speaker's action. "
         "Do not repeat unsupported prose. Keep private thoughts private and honor withholding. "
         "Return only natural companion speech, without a narrator or an implementation report."},
        {"role": "user", "content": render_public_context(scoped) +
         "\nAnswer only the current report. Do not add an unrelated historical observation or assert that one "
         "report is the only evidence or only record. Preserve every attribution and uncertainty qualifier."},
    ], temperature=0.2, chat_template_kwargs={"enable_thinking": False})
    review = await review_public_claims(complete, candidate, scoped)
    return {"schema_version": "pm.public-grounding-recovery.v1", "candidate": candidate,
            "accepted": review.accepted, "claim_review": review.model_dump(mode="json"),
            "context": scoped.model_dump(mode="json")}


__all__ = ["recover_grounding"]
