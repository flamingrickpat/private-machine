"""Coordinate connected private operations through the existing child kernels.

See :mod:`workflows.inner_thought.host` for scheduling and acceptance. This
projection supplies accepted graph identities and their original branch
payloads to the existing thought child. The coordinator invokes nomination,
shared attention, actual scoped recall when requested, thought, and boundary
integration, and the parked re-admission that shares that same attention decision.
PrivateInquiryOwner retains commit and GWT repair authority. This module does not
convert legacy episode history.

An opportunity first closes work from an older Critical epoch, so stale work
never holds the private slot. The closed question competes in the same
attention decision as a displaced offer. Admission opens a new episode; the
earlier nodes reach its thought stage as earlier private artifacts only.
When several displaced questions wait, :mod:`workflows.inner_thought.offer_salience`
chooses the one to offer from what the private mind currently holds.
"""
from __future__ import annotations

from typing import Literal
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import monotonic
import hashlib

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from pm import vclock
from pm.memory.snapshot import Snapshot
from pm.memory.dependencies import SourceDependency
from workflows.public_speech.public_context import EvidenceItem
from pm.records.mental_state import FullMentalState, compute_attention_bias
from pm.records.enums import FeatureType
from pm.records.memory_records import Feature
from workflows.cognitive_control.attention import CapturedPrivateFocus, validate_private_attention, validate_private_focus
from workflows.inner_thought.graph_host import (
    PrivateInquiryOwner, active_inquiry, displaced_inquiries, parked_inquiries,
)
from workflows.inner_thought.inquiry import (
    InquiryAlternativeMetadata, InquiryNodeMetadata, InquiryRecord, displaced_offer, inquiry_origin, parked_offer,
    read_displaced_inquiry,
)
from workflows.inner_thought.nominations import NominationBatchRecord, read_nomination_batch
from workflows.inner_thought.offer_salience import RetainedQuestion, attention_cues, select_retained_offer
from workflows.inner_thought.operations import operation_lines, validate_operation
from workflows.inner_thought.graph_outcomes import capture_graph_outcome
from workflows.inner_thought.pipeline import ModelProposalError, citable_ids, validate_integration
from workflows.inner_thought.schemas import (
    AttentionCandidate, DisplacedInquiryOffer, InquiryAlternative, InquiryGrant, InquiryOrigin,
    InquiryResumeDecision, InquiryState, InquiryStep, InnerRequest, IntegrationResult, NominationBatch,
    ParkedInquiryOffer, PrivateAttentionContext, RetainedOfferSelection, ThoughtAlternative,
    ThoughtAttentionDecision, ThoughtCandidate, ThoughtResult, TriggerChoice,
)

StageRunner = Callable[[str, dict[str, JsonValue]], dict[str, JsonValue]]
#: Embed texts in input order through the run's one embedding endpoint.
EmbedTexts = Callable[[Sequence[str]], list[list[float]]]


class PrivateGraphResult(BaseModel):
    """The parent's accepted graph result; it is never committed as a second thought."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    stage_kind: Literal["private_inquiry_result"] = "private_inquiry_result"
    summary: str
    receipt_id: int | None = None
    finding_id: int | None = None


class PrivateRecall(BaseModel):
    """Actual frozen-view memory matches and their original source guards."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    matches: tuple[EvidenceItem, ...]
    sources: tuple[SourceDependency, ...]
    trace: dict[str, JsonValue]


class InquiryNodeView(BaseModel):
    """An accepted node available for exact relations and comparison citations."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    record_id: int
    text: str
    operation: str
    reality: str
    parent_record_ids: tuple[int, ...]
    source_event_ids: tuple[int, ...]
    payload: InquiryStep | InquiryAlternative


class GraphThoughtRequest(BaseModel):
    """Frozen current operation input, separate from legacy linear requests."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    stage_kind: Literal["inquiry_operation"] = "inquiry_operation"
    request: InnerRequest
    state: InquiryState
    grant: InquiryGrant
    origin: InquiryOrigin
    nodes: tuple[InquiryNodeView, ...]
    earlier_artifacts: tuple[InquiryNodeView, ...] = Field(default=(),
        description="Nodes of the closed earlier episodes behind a displaced question, oldest first. "
                    "They are private history, never graph relation targets of this episode.")
    recall: PrivateRecall | None = None


def _node_view(view: Snapshot, node_id: int) -> InquiryNodeView:
    """Read one accepted node with its original payload and lineage."""
    node = view.get(node_id)
    if not isinstance(node, Feature) or node.lineage is None:
        raise ValueError("An accepted inquiry node requires its original lineage.")
    if node.feature_type == FeatureType.InnerMonologue:
        payload = InquiryNodeMetadata.model_validate(node.metadata).inquiry_step
    elif node.feature_type == FeatureType.Thought:
        payload = InquiryAlternativeMetadata.model_validate(node.metadata).inquiry_alternative
    else:
        raise ValueError("An inquiry node must be an executed operation or an independent branch.")
    return InquiryNodeView(record_id=node_id, text=node.content,
        operation=node.lineage.operation, reality=node.lineage.reality,
        parent_record_ids=tuple(node.lineage.parent_record_ids),
        source_event_ids=tuple(node.lineage.source_event_ids), payload=payload)


def earlier_artifacts(view: Snapshot, state: InquiryState) -> tuple[InquiryNodeView, ...]:
    """Read the nodes of every closed episode behind a displaced question, oldest first.

    A question can be displaced more than once, so this follows the origin
    chain to its first episode. An episode that began from a nomination or a
    deliberate action has no earlier artifacts.
    """
    episodes_newest_first: list[InquiryState] = []
    current = state
    while inquiry_origin(view, current).kind == "displaced_inquiry":
        current = read_displaced_inquiry(view, current.origin_record_id).receipt.state
        episodes_newest_first.append(current)
    artifacts: list[InquiryNodeView] = []
    for earlier in reversed(episodes_newest_first):
        for node_id in earlier.node_ids:
            artifacts.append(_node_view(view, node_id))
    return tuple(artifacts)


def graph_thought_request(view: Snapshot, request: InnerRequest, parent: InquiryRecord,
                          grant: InquiryGrant) -> GraphThoughtRequest:
    """Supply actual accepted nodes; never invent IDs for new alternatives."""
    state = parent.receipt.state
    origin = inquiry_origin(view, state)
    nodes = tuple(_node_view(view, node_id) for node_id in state.node_ids)
    return GraphThoughtRequest(request=request, state=state, grant=grant, origin=origin, nodes=nodes,
                               earlier_artifacts=earlier_artifacts(view, state))


def graph_thought_instruction(request: GraphThoughtRequest) -> str:
    """Explain the admitted operation and graph's epistemic acceptance rules."""
    operation = validate_operation(request.state.next_operation)
    accepted_nodes = list(request.state.node_ids)
    if accepted_nodes:
        relation_rule = "Source IDs belong in evidence_ids. They are not graph relation targets."
    else:
        # The model otherwise relates the new thought to the source record that
        # prompted it, which the graph rejects because a relation is an edge
        # between executed nodes. State the empty target set as its own rule.
        relation_rule = ("No graph node has been accepted yet, so this step has no relation target. "
                         "Return relations=[] and progress.compared_node_ids=[]. "
                         "The source records you cite belong in evidence_ids only.")
    if operation.mode == "generative":
        branch_rule = "It can create zero to three new alternative branches."
        # A generative step that surveys earlier nodes otherwise reports that
        # survey as a comparison, which only a discriminatory operation can claim.
        progress_rule = ("A generative operation cannot claim progress.kind='discrimination'. "
                         "Use progress.kind='new_evidence' only when this step inspected a new original source; "
                         "otherwise use progress.kind='none' with compared_node_ids=[].")
    else:
        branch_rule = ("It must return alternatives=[]; assess the supplied material "
                       "without generating new branch records.")
        progress_rule = ("It can claim progress.kind='discrimination' only when it compares at least two accepted "
                         "graph nodes, each named in compared_node_ids and in relations.")
    return (
        f"Perform {operation.name} ({operation.mode}) as {request.request.companion_name}'s private inquiry. "
        f"Purpose: {operation.description}\n"
        f"Accepted graph node IDs available for relations and comparisons: {accepted_nodes}.\n"
        + relation_rule + "\n"
        + f"This operation is {operation.mode}. " + branch_rule + "\n"
        + progress_rule + "\n"
        + _earlier_artifact_rule(request)
        + "Return one InquiryStep. Execute only the admitted operation. Cite only supplied source or accepted node IDs. "
        "A later step must relate explicitly to accepted earlier nodes. New alternatives have no committed IDs yet. "
        "Keep assumptions, evidence, plausibility, interest, desirability and action suitability separate. "
        "Neither emotional importance nor repeated thoughts supplies independent corroboration. "
        "Only generative operations create alternatives. After generation, continue with a discriminatory operation "
        "that compares all pending generated nodes before claiming a finding or completion. "
        "Use explicit relations to every compared node. Generation alone cannot certify its conclusion. "
        "Progress means acquired original evidence or a meaningful comparison; changed wording is not progress. "
        "A missing fact stays an information gap. Imagined content is never an observation. "
        "Do not change needs, contact anyone, execute external actions, or turn a proposal into an external fact. "
        "Continue only with a specific useful question and a delivered next operation. Otherwise state the outcome "
        "and remaining uncertainty. Only outcome='continue' names next_operation; all other outcomes use an empty string.\n"
        "Available operations:\n" + "\n".join(operation_lines())
    )


def _earlier_artifact_rule(request: GraphThoughtRequest) -> str:
    """State how a new episode may use the nodes of the episode it replaced.

    Without this rule the model reads its own earlier conclusions as support
    for the same conclusions now. The earlier thinking happened under an
    authority that Critical reconsideration replaced, and repeating a thought
    is never corroboration.
    """
    if not request.earlier_artifacts:
        return ""
    artifact_ids = [artifact.record_id for artifact in request.earlier_artifacts]
    return (
        "This inquiry restarts a question that accepted Critical reconsideration interrupted. "
        f"earlier_artifacts {artifact_ids} are your own earlier private thinking on it, made before that "
        "reconsideration. They are not outside evidence, not observations, and not corroboration: agreeing with "
        "an earlier artifact adds no support. Re-examine them against what is known now. You can cite an earlier "
        "artifact ID in evidence_ids as internal material. An earlier artifact is never a graph relation target "
        "and never belongs in compared_node_ids.\n")


def validate_graph_step(request: GraphThoughtRequest, step: InquiryStep) -> InquiryStep:
    """Reject stochastic semantic violations inside the child's repair boundary.

    The store owner still validates original source roots, current parents,
    and durable invariants. This check does not catch or disguise those bugs.
    """
    state = request.state
    if step.operation != state.next_operation:
        raise ModelProposalError("Execute the admitted next operation, without replacing it.")
    operation = validate_operation(step.operation)
    if step.next_operation:
        try:
            validate_operation(step.next_operation)
        except KeyError as error:
            raise ModelProposalError("The next operation must come from the delivered catalogue.") from error
    permitted = {source.record_id for source in request.grant.sources}
    evidence = set(step.evidence_ids)
    for alternative in step.alternatives:
        evidence.update(alternative.evidence_ids)
    if not evidence.issubset(permitted):
        raise ModelProposalError("Inquiry citations must name supplied sources or accepted nodes.")
    targets = {relation.target_record_id for relation in step.relations}
    if not state.node_ids:
        if targets:
            raise ModelProposalError(
                "No graph node exists yet, so this step cannot relate to one. Return relations=[] and "
                f"progress.compared_node_ids=[]. The cited source records {sorted(targets)} belong in evidence_ids.")
    elif not targets.issubset(state.node_ids) or (state.sequence > 0 and not targets):
        raise ModelProposalError(
            f"Relations must target one of the accepted graph nodes {sorted(state.node_ids)}. "
            "Original source IDs belong in evidence_ids. After the first step, retain at least one "
            "relation to an accepted graph node.")
    if operation.mode != "generative" and step.alternatives:
        raise ModelProposalError("Only a generative operation can create alternative branches.")
    compared = set(step.progress.compared_node_ids)
    if not compared.issubset(targets):
        raise ModelProposalError("Every compared node needs its explicit relation.")
    if step.progress.kind == "discrimination":
        # These are two different mistakes. One shared message told a model that
        # had already compared several nodes to compare two, and it never
        # learned that its operation mode was the actual violation.
        if operation.mode != "discriminatory":
            raise ModelProposalError(
                f"{operation.name} is a {operation.mode} operation, so it cannot claim discrimination progress. "
                "Set progress.kind='new_evidence' if this step inspected a new original source; otherwise set "
                "progress.kind='none' with compared_node_ids=[] and meaningful_difference=false. "
                "A later discriminatory operation makes the comparison.")
        if len(compared) < 2:
            raise ModelProposalError("Discrimination progress must compare at least two accepted nodes.")
    if step.progress.kind == "new_evidence" and (not step.progress.inspected_source_ids
            or not set(step.progress.inspected_source_ids).issubset(step.evidence_ids)):
        raise ModelProposalError("Evidence acquisition must cite its inspected original source IDs.")
    pending = set(state.pending_generation_ids)
    if operation.mode == "discriminatory":
        pending.difference_update(compared)
    if step.outcome in {"finding", "completed"} and (operation.mode == "generative" or pending):
        raise ModelProposalError("Compare all pending generated nodes before a finding or completion.")
    return step


@dataclass(frozen=True, slots=True)
class OpportunityAdmission:
    """The inquiry one attention opportunity admitted, or why it admitted none."""

    inquiry: InquiryRecord | None
    result: PrivateGraphResult | None


@dataclass(frozen=True, slots=True)
class WaitingDisplacedQuestion:
    """One displaced question that no episode took over, with its attention offer."""

    inquiry: InquiryRecord
    offer: DisplacedInquiryOffer


@dataclass(frozen=True, slots=True)
class DisplacedCandidate:
    """A displaced question, its new grant, its offer, and why it was the one offered."""

    inquiry: InquiryRecord
    grant: InquiryGrant
    offer: DisplacedInquiryOffer
    selection: RetainedOfferSelection


def displaced_candidate(owner: PrivateInquiryOwner, *, observer_entity_id: int, grant: InquiryGrant,
                        request: InnerRequest, mental_state: FullMentalState, selection_seed: str,
                        embed_texts: EmbedTexts | None) -> DisplacedCandidate | None:
    """Choose the one displaced question this opportunity offers to shared attention.

    One opportunity offers at most one displaced question, like parked work, so
    the attention prompt stays bounded. A single waiting question is offered
    without any measurement. Several waiting questions are rated against what
    the private mind currently holds: shared original sources, and embedding
    similarity to any current cue. See
    :func:`workflows.inner_thought.offer_salience.select_retained_offer`.

    Rating several questions makes one embedding request for the question and
    cue texts. ``embed_texts`` is the host's embedding endpoint; a missing one
    raises when a rating is needed. ``selection_seed`` is the opportunity's
    identity, so a restart offers the same question again.
    """
    view = owner.store.snapshot(capture=False)
    waiting: dict[int, WaitingDisplacedQuestion] = {}
    for displaced in displaced_inquiries(view, observer_entity_id=observer_entity_id):
        offer = displaced_offer(view, displaced)
        if offer is not None:
            waiting[displaced.record.id] = WaitingDisplacedQuestion(displaced, offer)
    if not waiting:
        return None

    _, salience_bias = compute_attention_bias(mental_state)
    if len(waiting) == 1:
        only_receipt_id = next(iter(waiting))
        selection = RetainedOfferSelection(offered_receipt_id=only_receipt_id, mode="only_candidate", seed="",
                                           salience_bias=salience_bias, ratings=())
    else:
        if embed_texts is None:
            raise RuntimeError("Rating several displaced questions requires the host's embedding endpoint.")
        questions: list[RetainedQuestion] = []
        for receipt_id, question in waiting.items():
            state = question.inquiry.receipt.state
            origin = inquiry_origin(view, state)
            questions.append(RetainedQuestion(receipt_id=receipt_id, text=origin.text + "\n" + origin.question,
                                              source_ids=state.root_source_ids))
        cues = attention_cues(request)
        # One request embeds the questions first and the cues after them.
        vectors = embed_texts([item.text for item in questions] + [cue.text for cue in cues])
        if len(vectors) != len(questions) + len(cues):
            raise ValueError("The embedding endpoint returned a different number of vectors than texts.")
        selection = select_retained_offer(questions, cues, question_vectors=vectors[:len(questions)],
            cue_vectors=vectors[len(questions):], salience_bias=salience_bias, seed=selection_seed)

    chosen = waiting[selection.offered_receipt_id]
    new_grant = owner.displaced_grant(chosen.inquiry, offered_grant=grant)
    return DisplacedCandidate(chosen.inquiry, new_grant, chosen.offer, selection)


@dataclass(frozen=True, slots=True)
class ParkedCandidate:
    """Retained parked work, the grant that can re-admit it, and its offer."""

    inquiry: InquiryRecord
    grant: InquiryGrant
    offer: ParkedInquiryOffer


def parked_candidate(owner: PrivateInquiryOwner, *, observer_entity_id: int,
                     grant: InquiryGrant) -> ParkedCandidate | None:
    """Read the parked work that this opportunity may offer to shared attention.

    The offer must be judged with the parked inquiry's own retained sources plus
    everything the current opportunity can cite, because an early re-admission
    needs evidence newly supplied beside that retained grant.
    """
    view = owner.store.snapshot(capture=False)
    current_ids = tuple(source.record_id for source in grant.sources)
    for parked in parked_inquiries(view, observer_entity_id=observer_entity_id):
        if parked.receipt.state.grant.critical_epoch != grant.critical_epoch:
            continue
        parked_grant = owner.execution_grant(parked, source_ids=current_ids)
        offer = parked_offer(view, parked, grant=parked_grant, now_utc=vclock.utcnow(), limits=owner.limits)
        if offer is not None:
            return ParkedCandidate(parked, parked_grant, offer)
    return None


def admit_opportunity(request: InnerRequest, *, owner: PrivateInquiryOwner, grant: InquiryGrant,
                      run_stage: StageRunner, nominations: NominationBatchRecord, batch_id: str,
                      pending_input_ids: tuple[int, ...], focus: CapturedPrivateFocus,
                      mental_state: FullMentalState, embed_texts: EmbedTexts | None = None) -> OpportunityAdmission:
    """Offer one nomination batch and any retained work to shared attention.

    One model call admits at most one inquiry for the private slot. Parked work
    competes in that same decision, and only on an elapsed parked deadline or
    original outside evidence newly supplied beside its retained grant, so
    elapsed parking alone never resumes it. A displaced question competes there
    too and needs no such basis. Every outcome stays durable: a nomination
    selection through ``stage_attention``, a parked re-admission through its
    resume receipt beside the opportunity's attention outcome, a displaced
    question through its new entry receipt beside that outcome, and any other
    decision through that attention outcome alone. A completed opportunity is
    never paid for twice. A deferral or dismissal leaves a displaced question
    offerable for a later opportunity. When several displaced questions wait,
    :func:`displaced_candidate` chooses the one that is offered, and every
    outcome retains that selection.
    """
    from workflows.inner_thought.stages.trigger.steps import AttentionStageRequest
    candidate = parked_candidate(owner, observer_entity_id=grant.observer_entity_id, grant=grant)
    view = owner.store.snapshot(capture=False)
    displaced_waiting = bool(displaced_inquiries(view, observer_entity_id=grant.observer_entity_id))
    if not nominations.nominations and candidate is None and not displaced_waiting:
        return OpportunityAdmission(None, PrivateGraphResult(summary=nominations.receipt.reason,
            receipt_id=nominations.record.id))
    # A retained attention outcome is a completed selection call. A later
    # opportunity receives a different captured-context identity.
    if any(record.metadata.get("thought_attention_batch") == batch_id
           for record in view.all_knoxels.values()):
        return OpportunityAdmission(None, PrivateGraphResult(
            summary="This nomination opportunity already has an attention outcome.",
            receipt_id=nominations.record.id))
    # Choose the displaced offer only now: rating several questions costs one
    # embedding request, which a completed opportunity must not pay again.
    displaced = displaced_candidate(owner, observer_entity_id=grant.observer_entity_id, grant=grant,
        request=request, mental_state=mental_state, selection_seed=batch_id + ":displaced-offer",
        embed_texts=embed_texts)
    displaced_selection = None if displaced is None else displaced.selection
    attention = PrivateAttentionContext(
        candidates=tuple(AttentionCandidate(nomination_id=item.record.id, nomination=item.receipt.nomination,
            root_source_ids=item.receipt.root_source_ids) for item in nominations.nominations),
        parked=() if candidate is None else (candidate.offer,),
        displaced=() if displaced is None else (displaced.offer,),
        pending_input_ids=pending_input_ids, accepted_focus=focus.focus,
        active_intention_ids=tuple(goal.knoxel_id for goal in request.active_goal_records),
        attention_summary=mental_state.state_cognition.model_dump_json())
    payload = AttentionStageRequest(request=request, attention=attention)
    decision = ThoughtAttentionDecision.model_validate(run_stage("trigger", payload.model_dump(mode="json")))
    validate_private_attention(attention, decision)
    if decision.parked_receipt_id is not None:
        assert candidate is not None  # Only an offered parked receipt can be named.
        resumed = owner.admit_parked_inquiry(candidate.inquiry,
            InquiryResumeDecision(operation=decision.admitted_operation, reason=decision.reason,
                trigger_source_ids=candidate.offer.available_evidence_ids),
            attention=decision, nomination_ids=nominations.receipt.nomination_ids, offered_grant=grant,
            grant=candidate.grant, batch_id=batch_id, now_utc=vclock.utcnow(), focus=focus,
            displaced_selection=displaced_selection)
        return OpportunityAdmission(resumed, None)
    if decision.displaced_receipt_id is not None:
        assert displaced is not None  # Only an offered displaced receipt can be named.
        restarted = owner.admit_displaced_inquiry(displaced.inquiry, attention=decision,
            nomination_ids=nominations.receipt.nomination_ids, offered_grant=grant, grant=displaced.grant,
            batch_id=batch_id, episode_id="inquiry:" + batch_id, now_utc=vclock.utcnow(), focus=focus,
            displaced_selection=displaced.selection)
        return OpportunityAdmission(restarted, None)
    if not nominations.nominations:
        outcome = owner.record_attention_outcome(batch_id=batch_id, nomination_ids=(), decision=decision,
            grant=grant, focus=focus, displaced_selection=displaced_selection)
        return OpportunityAdmission(None, PrivateGraphResult(summary=decision.reason, receipt_id=outcome.id))

    def validate_context() -> None:
        """Keep the supplied focus head and original expiry through selection acceptance."""
        validate_private_focus(owner.store.snapshot(capture=False), focus, now_utc=vclock.utcnow())

    selected = owner.accept_attention(nominations.receipt.nomination_ids, decision,
        commit_id="attention:" + batch_id, episode_id="inquiry:" + batch_id, now_utc=vclock.utcnow(),
        validate_context=validate_context, displaced_selection=displaced_selection)
    if selected.inquiry is None:
        return OpportunityAdmission(None, PrivateGraphResult(summary=decision.reason, receipt_id=selected.record.id))
    return OpportunityAdmission(selected.inquiry, None)


def run_graph_opportunity(request: InnerRequest, *, owner: PrivateInquiryOwner,
                          grant: InquiryGrant, run_stage: StageRunner,
                          mental_state: FullMentalState, control_generation: int,
                          pending_input_ids: tuple[int, ...], focus: CapturedPrivateFocus,
                          recall_memory: Callable[[str], PrivateRecall] | None = None,
                          embed_texts: EmbedTexts | None = None) -> PrivateGraphResult:
    """Retain nomination, selection, and one executed step as separate boundaries.

    Continuation uses the accepted graph rather than main acknowledgment of
    every internal node, whether the episode began from a private nomination or
    from an accepted deliberate action. The host retains control authority
    through its commit callback. Parked work re-enters only through the shared
    attention decision in :func:`admit_opportunity`; this entry never resumes it
    from elapsed parking or legacy episode metadata.

    Work admitted under an older Critical epoch is closed first, durably and
    without a model call. The slot is then free for this same opportunity, and
    the closed question competes in its attention decision. ``embed_texts`` is
    needed only when several displaced questions wait and one must be chosen.
    """
    owner.displace_superseded(observer_entity_id=grant.observer_entity_id, critical_epoch=grant.critical_epoch)
    view = owner.store.snapshot(capture=False)
    parent = active_inquiry(view, observer_entity_id=grant.observer_entity_id)
    if parent is not None:
        if parent.receipt.state.grant.critical_epoch != grant.critical_epoch:
            # Older work was just displaced, so this inquiry is from a newer
            # epoch than the opportunity. The host captures both under one lock.
            raise ValueError("An active inquiry holds a newer Critical epoch than the opportunity reading it.")
        if parent.receipt.operation_record_id is not None:
            owner.ensure_inner_turn(parent)
    if parent is None:
        if not request.trigger_candidates:
            return PrivateGraphResult(summary="No private opportunity requires nomination.")
        # Time and fresh invocation IDs do not make unchanged evidence a new
        # nomination opportunity. Accepted main guidance or changed sources do.
        material = grant.model_dump_json() + str(request.main_frame.tick)
        material += "".join(trigger.model_dump_json() for trigger in request.trigger_candidates)
        batch_id = hashlib.sha256(material.encode()).hexdigest()
        # The same qualifying opportunity also gates reconsideration of parked
        # work, so a parked question never causes an extra model call by itself.
        nominations = read_nomination_batch(view, batch_id)
        if nominations is None:
            result = NominationBatch.model_validate(run_stage("nominate", request.model_dump(mode="json")))
            nominations = owner.accept_nominations(result, grant=grant, batch_id=batch_id)
        admission = admit_opportunity(request, owner=owner, grant=grant, run_stage=run_stage,
            nominations=nominations, batch_id=batch_id, pending_input_ids=pending_input_ids,
            focus=focus, mental_state=mental_state, embed_texts=embed_texts)
        if admission.inquiry is None:
            assert admission.result is not None  # Every non-admission names its own outcome.
            return admission.result
        parent = admission.inquiry
    node_grant = owner.execution_grant(parent)
    dependencies = {source.record_id: source for source in grant.sources}
    dependencies.update({source.record_id: source for source in node_grant.sources})
    execution_grant = node_grant.model_copy(update={"sources": tuple(dependencies[key] for key in sorted(dependencies))})
    supplied = graph_thought_request(owner.store.snapshot(capture=False), request, parent, execution_grant)
    started = monotonic()
    if parent.receipt.state.next_operation == "memory_search":
        if recall_memory is None:
            raise RuntimeError("An admitted memory search requires the host's actual frozen-view retrieval.")
        recalled = recall_memory(supplied.origin.question)
        for source in recalled.sources:
            dependencies.setdefault(source.record_id, source)
        execution_grant = execution_grant.model_copy(update={
            "sources": tuple(dependencies[key] for key in sorted(dependencies))})
        supplied = supplied.model_copy(update={"grant": execution_grant, "recall": recalled})
        request = request.model_copy(deep=True)
        existing_ids = citable_ids(request)
        for match in recalled.matches:
            if match.knoxel_id not in existing_ids:
                request.candidates.append(ThoughtCandidate(knoxel_id=match.knoxel_id, text=match.text,
                    revision=match.revision, kind="recalled_memory"))
                existing_ids.add(match.knoxel_id)
        supplied = supplied.model_copy(update={"request": request})
    step = InquiryStep.model_validate(run_stage("thought", supplied.model_dump(mode="json")))
    validate_graph_step(supplied, step)
    outcome = None
    if step.outcome != "continue":
        # Integration sees existing accepted IDs and the proposed operation,
        # without inventing an identity for that not-yet-committed operation.
        integration_request = request.model_copy(deep=True)
        existing = citable_ids(integration_request)
        for node in supplied.nodes:
            if node.record_id not in existing:
                integration_request.candidates.append(ThoughtCandidate(knoxel_id=node.record_id,
                    text=node.text, revision=owner.store.revisions[node.record_id], kind="inquiry_node"))
        thought = ThoughtResult(thought=step.thought, evidence_ids=list(step.evidence_ids),
            evidence_summary="Original source IDs: " + ", ".join(map(str, step.evidence_ids)),
            interpretation=step.reason, uncertainty=1 - step.judgments.plausibility,
            reality=step.reality,
            alternatives=[ThoughtAlternative(text=item.text, disposition=item.disposition, reason=item.reason)
                          for item in step.alternatives], continuation="summarize")
        trigger = TriggerChoice(operation=step.operation, candidate_ids=list(step.evidence_ids), reason=step.reason)
        payload = integration_request.model_dump(mode="json")
        payload.update(thought=thought.model_dump(mode="json"), trigger=trigger.model_dump(mode="json"),
                       operation={"name": step.operation})
        integration = IntegrationResult.model_validate(run_stage("integrate", payload))
        validate_integration(integration, citable_ids(integration_request), integration_request)
        outcome = capture_graph_outcome(execution_grant, step, integration)
    active_seconds = monotonic() - started
    accepted = owner.accept_step(parent, step, grant=execution_grant,
        commit_id=f"inquiry-step:{parent.record.id}", active_seconds=active_seconds,
        now_utc=vclock.utcnow(), mental_state=mental_state, control_generation=control_generation,
        outcome=outcome)
    boundary = step.outcome != "continue"
    return PrivateGraphResult(summary=accepted.receipt.state.stop_reason or step.reason,
        receipt_id=accepted.record.id, finding_id=accepted.receipt.operation_record_id if boundary else None)
