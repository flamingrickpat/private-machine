"""Review meaningful public claims once against supplied evidence.

The review does not score voice or rewrite prose. A rejected candidate remains
in its report. The caller publishes a separate recovery outcome.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
import yaml

from workflows.public_context import PublicContext, render_public_context
from workflows.main_cognition.stage_support import extract_json


class PublicClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1)
    domain: Literal["history", "task", "human_sense", "minecraft", "emotion", "private", "other"]
    status: Literal["supported", "figurative", "uncertain", "intentional_deception", "unsupported"]
    evidence_ids: list[int] = Field(default_factory=list)
    evidence_use: Literal["event", "utterance", "state", "none"] = "event"
    verification_claim: bool = True
    time_scope: Literal["past", "current", "stable", "unspecified"] = "unspecified"
    external_physical_state: bool = False
    reason: str


class LiteralCondition(BaseModel):
    """An utterance-only time interpretation, without factual evidence."""

    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1)
    time_scope: Literal["past", "current", "stable", "unspecified"]


class LiteralConditions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    conditions: list[LiteralCondition] = Field(default_factory=list)


class CitationEntailment(BaseModel):
    """One bounded comparison with the claim's cited original text."""

    model_config = ConfigDict(extra="forbid")
    claim_index: int = Field(ge=0, le=31)
    supported: bool
    reason: str


class CitationEntailments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    checks: list[CitationEntailment]


class ClaimReview(BaseModel):
    """Semantic claims and the host's final acceptance decision."""

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["pm.public-claims.v1"] = "pm.public-claims.v1"
    accepted: bool
    public_form: Literal["speech", "narration", "private_transcript", "empty"] = "speech"
    claims: list[PublicClaim] = Field(default_factory=list)
    reason: str = ""
    literal_conditions: list[LiteralCondition] = Field(default_factory=list)
    citation_checks: list[CitationEntailment] = Field(default_factory=list)


async def review_public_claims(complete, candidate: str, context: PublicContext,
                               receipts: list[dict] | None = None) -> ClaimReview:
    """Interpret time and evidence, then check citations. Each phase runs once.

    Unknown evidence IDs and unsupported claims reject the whole utterance.
    Deception never waives capability, task, or private disclosure checks.
    """
    prompts = yaml.safe_load((Path(__file__).parent / "public_claims.yaml").read_text(encoding="utf-8"))
    # Evidence dates must not change the meaning of an unqualified assertion.
    # Interpret literal environment conditions before exposing old reports.
    if len(candidate) > 12000:
        raise ValueError("Public claim interpretation is limited to 12,000 characters.")
    literal_schema = LiteralConditions.model_json_schema()
    literal_schema["required"] = ["conditions"]
    literal_raw = await complete([
        {"role": "system", "content": "Extract literal external environment conditions from this utterance only. "
         "Include plant health, room conditions, or game-world conditions. Exclude capabilities, task/file verification, "
         "emotions, names, preferences, and plans. Return exact substrings with time_scope past, current, stable, or unspecified. "
         "Copy complete clauses verbatim, including negation, uncertainty, date, and attribution qualifiers. "
         "A quoted report inside You reported that ... is an attributed report, not a new assertion of its contents. "
         "Never extract a positive condition from a negation, question, or uncertain statement. "
         "Read the entire sentence and retain date or attribution qualifiers. I know your plant is alive asserts a current "
         "condition; do not invent an implicit past attribution. You reported in May that the plant was alive describes past "
         "reporting. Do not assume any supporting evidence. Return an empty list when there is no such condition."},
        {"role": "user", "content": candidate},
    ], temperature=0.1, chat_template_kwargs={"enable_thinking": True},
       response_format={"type": "json_schema", "json_schema": {"name": "literal_conditions", "strict": True,
                        "schema": literal_schema}})
    literal = LiteralConditions.model_validate(extract_json(literal_raw))
    if any(item.text not in candidate for item in literal.conditions):
        raise ValueError("Literal condition interpretation must retain an exact utterance substring.")
    # Voice examples and private prose cannot establish an external event.
    # Keep identity, accepted state, original evidence, and prior presentation.
    evidence_context = context.model_copy(update={
        "character": "", "workspace": "", "selected_action": "", "memories": [],
    })
    entailment = (
        "\nThe candidate is public speech by public_context.companion_name to public_context.user_name. "
        "In the candidate, I/me/my refers to that companion and you/your refers to that user. "
        "Resolve source first-person pronouns to each source's named author. The evaluator is outside this conversation. "
        "Referring to your neighbor's reported advice while withholding its truth does not assert that advice is correct. "
        "\nSupported means established by the supplied evidence. A plausible interpretation does not establish a fact. "
        "The reporter or recipient is not automatically the actor. A report that someone attached a wrong file "
        "leaves that actor unidentified. Do not infer that the reporter performed that error, even in a report "
        "about their own workflow. An unqualified inferred actor is unsupported, never uncertain merely because it could be true."
        "\nPreserve the recorded action and its object. Adjacent sentences do not establish an action-object relation or causality. "
        "A report such as I did coffee, followed by a separate plant update, does not state that coffee was given to the plant. "
        "Without an explicit application event, plant survival cannot establish that coffee application was safe or nonfatal. "
        "Reject an unqualified application or causal assertion when its source leaves that relation unstated."
        "\nExtract factual assertions and disclosure claims. Exclude pure greetings, thanks, apologies, and acknowledgments "
        "such as you are welcome. Those perform a conversational act; they do not assert an external fact and need no "
        "factual citation. Do not invent factual claims from these social words."
        " Rhetorical challenges, requests, jokes about style, and subjective evaluations are not external facts. "
        "Classify those as figurative or uncertain, rather than supported external claims with no evidence. "
        "A previous teasing request cannot establish that the current user requested another lie. "
        "Only the current report and accepted presentation establish the current request."
        "\nExplicit epistemic limits such as I cannot establish why, I cannot verify, or I do not know are uncertain. "
        "Do not classify them as supported external facts with no evidence. They withhold a conclusion. "
        "A positive claim that records are absent or that only one report exists is a different factual assertion."
        "\nThe evidence is a partial memory snapshot, not a complete inventory of the canonical store. "
        "It cannot establish that this is the only record in memory, that no other failure was recorded, or that a record "
        "disappeared. Unqualified exhaustive memory claims are unsupported. An explicitly bounded statement about the "
        "supplied excerpt is different. Uncertainty about the number of events does not qualify a definite record-count claim."
        "\nAn unknown condition does not qualify a definite conditional rule. If the plant is species X, treatment Y will "
        "kill it asserts that causal rule, even when its species is unknown. Such technical rules require supplied evidence. "
        "A claim about another person's intent also requires evidence or an explicitly qualified possibility. "
        "Do not treat a conditional antecedent as uncertainty about its asserted consequence."
        "\nPreserve the complete proposition, including negation, uncertainty, quotation, and attribution. "
        "Copy the actual claim span from the candidate. Do not invent a separate supporting statement that was not spoken. "
        "An implied premise is part of checking the original claim, not another public claim. "
        "Cite every original needed for a relation: chronology needs both compared events, and a correction needs "
        "the relevant old and new declarations. A restart record alone cannot date a codeword assignment. "
        "I cannot establish that the plant needs coffee does not assert that it needs coffee. "
        "Classify the complete statement as uncertain or a supported knowledge limitation; never extract its embedded "
        "positive clause as an unsupported assertion. I do not know X or have a verified source negates both alternatives."
        "\nAn archived report of a plant, room, or world condition cannot verify that condition today. "
        "A present-tense assertion needs a current user report or observation, not an imported historical record. "
        "Stable names, preferences, and plans can remain known until corrected; physical conditions require fresh evidence."
        "\nA claim that a report or conversation exists requires its actual source. Absence of a report does not "
        "support saying I have that report. Do not classify a positive historical assertion as knowledge state to omit its citation."
        "\nClassify time_scope from the utterance, not the source date. Unqualified present tense such as "
        "I know your plant is alive describes current state, even when its evidence is old. Set external_physical_state "
        "true for literal plant, room, or game-environment conditions. Capability statements, task/file verification, "
        "accepted emotions, names, preferences, and plans are not external physical conditions. A dated attribution "
        "to an older report has past scope. Current conditions require original evidence of that current condition; "
        "a current question or a report of someone's hypothesis does not establish it."
        "\nFor every supplied literal condition, include a claim whose text exactly matches its extracted substring. "
        "Preserve its time_scope. Evidence may support or reject that condition but cannot change its meaning."
    )
    messages = [
        {"role": "system", "content": prompts["system_prompt"] + entailment},
        {"role": "user", "content": "Accepted context:\n" + render_public_context(evidence_context)
         + "\nExecuted tool receipts:\n" + json.dumps(receipts or [], ensure_ascii=False, default=str)
         + "\nUtterance-only literal conditions:\n" + literal.model_dump_json()
         + "\nCandidate utterance:\n" + candidate},
    ]
    schema = ClaimReview.model_json_schema()
    schema["properties"].pop("literal_conditions")
    schema["properties"].pop("citation_checks")
    schema["required"] = ["accepted", "public_form", "claims", "reason"]
    schema["$defs"]["PublicClaim"]["required"] = list(PublicClaim.model_fields)
    raw = await complete(messages, temperature=0.1, chat_template_kwargs={"enable_thinking": True},
        response_format={"type": "json_schema", "json_schema": {"name": "public_claim_review",
                         "strict": True, "schema": schema}})
    review = ClaimReview.model_validate(extract_json(raw))
    review.literal_conditions = literal.conditions
    for claim in review.claims:
        for condition in literal.conditions:
            if condition.text.casefold() in claim.text.casefold() or claim.text.casefold() in condition.text.casefold():
                # Capability state is supplied by the live provider, not an
                # external plant, room, or game-environment observation.
                if claim.domain in {"history", "other"} or claim.external_physical_state:
                    claim.external_physical_state = True
                    claim.time_scope = condition.time_scope
    review = validate_claim_review(review, candidate, context)
    selected = [(index, claim) for index, claim in enumerate(review.claims)
                if claim.status == "supported" and claim.domain == "history"] if review.accepted else []
    if selected:
        if len(review.claims) > 32:
            raise ValueError("Citation review supports at most 32 claims.")
        by_id = {int(item["knoxel_id"]): item for item in context.evidence}
        comparisons = []
        for index, claim in selected:
            source_ids = set(claim.evidence_ids)
            source_ids.update(source for key in claim.evidence_ids for source in by_id.get(key, {}).get("source_ids", [])
                              if by_id.get(key, {}).get("evidence_role") == "derived")
            roles = {"user_report", "runtime_event", "world_observation"}
            if claim.evidence_use == "utterance":
                roles.add("companion_utterance")
            comparisons.append({"claim_index": index, "claim": claim.text,
                "sources": [by_id[key] for key in sorted(source_ids) if key in by_id
                            and by_id[key].get("evidence_role") in roles]})
        # The context marks original startup IDs separately from recalled prose.
        lifecycle = [item for item in context.evidence if item.get("lifecycle_event") == "startup"]
        current = [item for item in context.evidence if item["knoxel_id"] == context.user_frontier]
        raw = await complete([
            {"role": "system", "content": "Check each historical assertion against only its supplied cited source text. "
             "The candidate is public speech by the supplied speaker to the supplied addressee. "
             "In that speech, I/me refers to the speaker and you/your refers to the addressee. "
             "In original source text, first-person pronouns refer to that source's named author. "
             "Return supported true only when the exact asserted topic, actor, action, object, and relation are established. "
             "Faithful paraphrases are valid; do not require identical words. A rhetorical question echoing the current "
             "request is an acknowledgment, not a new historical event. Use the supplied current UTC time and source "
             "timestamps for straightforward relative dates such as yesterday. "
             "A merely plausible interpretation is unsupported. Separate nearby sentences do not establish causality or "
             "a stronger relation than the actual claim. Contextual where or temporal then does not itself assert because. "
             "I did coffee does not establish giving coffee to a plant. An email report cannot "
             "establish a neighbor's plant advice. A dated plant report establishes the past report, not today's condition. "
             "A generated utterance establishes speech, not its described event. Preserve uncertainty in the source. "
             "Original lifecycle events and UTC source timestamps establish restart chronology. "
             "The current user report supplies request scope, not missing facts. A source timestamp can date the report. "
             "A source with evidence_role user_report is original user testimony, not generated companion speech. "
             "The supplied original source objects are retained records in the accepted memory context. Their presence "
             "supports remembering or having that same sourced report on file. The user's text need not describe how "
             "the assistant stores it. Conversational on file is not a literal file operation without a path or execution claim. "
             "This proves existence of the supplied report, not an exhaustive inventory or an unrelated report's existence. "
             "An epistemic contrast such as I cannot establish why, only that you reported the correction distinguishes "
             "an established report from an unverified explanation. It does not assert an exhaustive record count "
             "or that no other event or record exists. "
             "A user's explicit assignment of a project codeword, name, preference, or plan establishes that declared fact. "
             "Those stable facts remain known until a correction. An assignment before the latest startup establishes a "
             "pre-restart codeword; do not demand an observation of its value at every intervening moment. "
             "Do not assume any additional context or missing events. Return one check for every supplied claim_index."},
            {"role": "user", "content": json.dumps({"comparisons": comparisons,
             "speaker": context.companion_name, "addressee": context.user_name,
             "lifecycle_events": lifecycle, "current_user_report": current,
             "current_temporal_context": context.temporal_context}, ensure_ascii=False, default=str)},
        ], temperature=0.1, chat_template_kwargs={"enable_thinking": True},
           response_format={"type": "json_schema", "json_schema": {"name": "citation_entailments", "strict": True,
                            "schema": CitationEntailments.model_json_schema()}})
        checks = CitationEntailments.model_validate(extract_json(raw)).checks
        if len(checks) != len(selected) or {item.claim_index for item in checks} != {index for index, _ in selected}:
            raise ValueError("Citation review must cover each supplied claim exactly once.")
        review.citation_checks = checks
        for check in checks:
            if not check.supported:
                review.claims[check.claim_index].status = "unsupported"
                review.claims[check.claim_index].reason = check.reason
        review = validate_claim_review(review, candidate, context)
    return review


def validate_claim_review(review: ClaimReview, candidate: str, context: PublicContext) -> ClaimReview:
    """Apply source and intent constraints to one parsed real review.

    This boundary cannot establish semantic entailment. It rejects citations
    that the model cannot use as factual evidence, even when it accepts them.
    """
    # Revision guards can include source leaves omitted by the prompt budget.
    # A citation must name evidence that this review actually received.
    allowed = {int(item["knoxel_id"]) for item in context.evidence}
    allowed.update(int(item["id"]) for item in context.ontology)
    event_ids = {int(item["knoxel_id"]) for item in context.evidence
                 if item.get("evidence_role") in {"user_report", "runtime_event", "world_observation"}}
    by_id = {int(item["knoxel_id"]): item for item in context.evidence}
    for condition in review.literal_conditions:
        matching = [claim for claim in review.claims if condition.text.casefold() in claim.text.casefold()
                    or claim.text.casefold() in condition.text.casefold()]
        if matching and all(claim.domain in {"task", "human_sense", "emotion", "private"}
                            or (claim.domain == "minecraft" and claim.evidence_use == "state"
                                and not claim.evidence_ids) for claim in matching):
            continue
        if condition.time_scope == "current" and not any(
            condition.text.casefold() == claim.text.casefold() for claim in review.claims
        ):
            review.accepted = False
            review.reason = "A current literal condition was omitted from evidence review."
    for claim in review.claims:
        if (claim.domain == "other" and claim.status == "supported"
                and claim.evidence_use == "none" and not claim.evidence_ids):
            claim.status = "unsupported"
            claim.reason = "Model knowledge alone does not verify this external factual claim."
        if any(key not in allowed for key in claim.evidence_ids):
            claim.status = "unsupported"
            claim.reason = "The claim cites evidence outside the supplied snapshot."
        if claim.domain == "history" and claim.status == "supported" and not claim.evidence_ids:
            claim.status = "unsupported"
            claim.reason = "A positive historical assertion requires a supplied source, even when labeled knowledge state."
        if claim.status == "supported" and claim.external_physical_state and claim.time_scope == "current":
            fresh = [by_id[key] for key in claim.evidence_ids if key in by_id
                     and not by_id[key].get("historical") and key in event_ids]
            if not fresh:
                claim.status = "unsupported"
                claim.reason = "Historical reports cannot establish a current external physical condition."
        if claim.domain == "history" and claim.status == "supported" and claim.evidence_use == "event":
            # Resolve a cited abstraction through its rendered original chain.
            # Generated speech has no such grant, even when it quotes a user.
            sources = {source for key in claim.evidence_ids for source in by_id.get(key, {}).get("source_ids", [])
                       if by_id.get(key, {}).get("evidence_role") == "derived" and source in event_ids}
            claim.evidence_ids = list(dict.fromkeys([*claim.evidence_ids, *sorted(sources)]))
        if (claim.domain == "history" and claim.status == "supported" and claim.evidence_use == "event"
                and not event_ids.intersection(claim.evidence_ids)):
            claim.status = "unsupported"
            claim.reason = "A historical event requires an original report or an executed event, not generated speech."
        if claim.status == "intentional_deception" and (
            context.presentation.mode != "deceive" or not context.action_schema_id
            or not context.presentation.purpose or not context.presentation.known_fact_ids
            or claim.domain not in {"history", "other", "task"} or claim.verification_claim
            or not set(claim.evidence_ids).intersection(context.presentation.known_fact_ids)
        ):
            claim.status = "unsupported"
            claim.reason = "No applicable prior deception intent supports this claim."
    review.accepted = bool(candidate.strip()) and review.public_form == "speech" and review.accepted and all(
        claim.status != "unsupported" for claim in review.claims
    )
    return review


__all__ = ["CitationEntailment", "CitationEntailments", "ClaimReview", "LiteralCondition", "LiteralConditions",
           "PublicClaim", "review_public_claims", "validate_claim_review"]
