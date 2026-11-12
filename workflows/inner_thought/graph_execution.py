"""Coordinate connected private operations through the existing child kernels.

See :mod:`workflows.inner_thought.host` for scheduling and acceptance. This
projection supplies accepted graph identities and their original branch
payloads to the existing thought child. The coordinator invokes nomination,
shared attention, actual scoped recall when requested, thought, and boundary
integration. PrivateInquiryOwner retains commit and GWT repair authority.
This module does not convert legacy episode history or admit parked work.
"""
from __future__ import annotations

from typing import Literal
from collections.abc import Callable
from time import monotonic
import hashlib

from pydantic import BaseModel, ConfigDict, JsonValue

from pm import vclock
from pm.memory.snapshot import Snapshot
from pm.memory.dependencies import SourceDependency
from workflows.public_speech.public_context import EvidenceItem
from pm.records.mental_state import FullMentalState
from pm.records.enums import FeatureType
from pm.records.memory_records import Feature
from workflows.cognitive_control.attention import CapturedPrivateFocus, validate_private_attention, validate_private_focus
from workflows.inner_thought.graph_host import PrivateInquiryOwner, active_inquiry
from workflows.inner_thought.inquiry import InquiryAlternativeMetadata, InquiryNodeMetadata, InquiryRecord
from workflows.inner_thought.nominations import read_nomination, read_nomination_batch
from workflows.inner_thought.operations import operation_lines, validate_operation
from workflows.inner_thought.graph_outcomes import capture_graph_outcome
from workflows.inner_thought.pipeline import ModelProposalError, citable_ids, validate_integration
from workflows.inner_thought.schemas import (
    AttentionCandidate, InnerRequest, InquiryAlternative, InquiryGrant, InquiryState, InquiryStep, NominationBatch,
    IntegrationResult, PrivateAttentionContext, ThoughtAlternative, ThoughtAttentionDecision,
    ThoughtCandidate, ThoughtNomination, ThoughtResult, TriggerChoice,
)

StageRunner = Callable[[str, dict[str, JsonValue]], dict[str, JsonValue]]


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
    origin: ThoughtNomination
    nodes: tuple[InquiryNodeView, ...]
    recall: PrivateRecall | None = None


def graph_thought_request(view: Snapshot, request: InnerRequest, parent: InquiryRecord,
                          grant: InquiryGrant) -> GraphThoughtRequest:
    """Supply actual accepted nodes; never invent IDs for new alternatives."""
    state = parent.receipt.state
    origin = read_nomination(view, state.origin_record_id).receipt.nomination
    nodes: list[InquiryNodeView] = []
    for node_id in state.node_ids:
        node = view.get(node_id)
        if not isinstance(node, Feature) or node.lineage is None:
            raise ValueError("An accepted inquiry node requires its original lineage.")
        if node.feature_type == FeatureType.InnerMonologue:
            payload = InquiryNodeMetadata.model_validate(node.metadata).inquiry_step
        elif node.feature_type == FeatureType.Thought:
            payload = InquiryAlternativeMetadata.model_validate(node.metadata).inquiry_alternative
        else:
            raise ValueError("An inquiry node must be an executed operation or an independent branch.")
        nodes.append(InquiryNodeView(record_id=node_id, text=node.content,
            operation=node.lineage.operation, reality=node.lineage.reality,
            parent_record_ids=tuple(node.lineage.parent_record_ids),
            source_event_ids=tuple(node.lineage.source_event_ids), payload=payload))
    return GraphThoughtRequest(request=request, state=state, grant=grant, origin=origin, nodes=tuple(nodes))


def graph_thought_instruction(request: GraphThoughtRequest) -> str:
    """Explain the admitted operation and graph's epistemic acceptance rules."""
    operation = validate_operation(request.state.next_operation)
    return (
        f"Perform {operation.name} ({operation.mode}) as {request.request.companion_name}'s private inquiry. "
        f"Purpose: {operation.description}\n"
        f"Accepted graph node IDs available for relations and comparisons: {list(request.state.node_ids)}. "
        "Source IDs belong in evidence_ids. They are not graph relation targets. "
        "When the accepted graph node list is empty, relations and compared_node_ids must both be empty. "
        f"This operation is {operation.mode}. "
        + ("It can create zero to three new alternative branches.\n" if operation.mode == "generative" else
           "It must return alternatives=[]; assess the supplied material without generating new branch records.\n") +
        "Return one InquiryStep. Execute only the admitted operation. Cite only supplied source or accepted node IDs. "
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
    if not targets.issubset(state.node_ids) or (state.sequence > 0 and not targets):
        raise ModelProposalError(f"Relations must target accepted graph nodes {list(state.node_ids)}. "
            "Original source IDs belong in evidence_ids. With no graph nodes, return relations=[]. "
            "After the first step, retain at least one relation to an accepted graph node.")
    if operation.mode != "generative" and step.alternatives:
        raise ModelProposalError("Only a generative operation can create alternative branches.")
    compared = set(step.progress.compared_node_ids)
    if not compared.issubset(targets):
        raise ModelProposalError("Every compared node needs its explicit relation.")
    if step.progress.kind == "discrimination" and (operation.mode != "discriminatory" or len(compared) < 2):
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


def run_graph_opportunity(request: InnerRequest, *, owner: PrivateInquiryOwner,
                          grant: InquiryGrant, run_stage: StageRunner,
                          mental_state: FullMentalState, control_generation: int,
                          pending_input_ids: tuple[int, ...], focus: CapturedPrivateFocus,
                          recall_memory: Callable[[str], PrivateRecall] | None = None) -> PrivateGraphResult:
    """Retain nomination, selection, and one executed step as separate boundaries.

    Continuation uses the accepted graph rather than main acknowledgment of
    every internal node. The host retains control authority through its commit
    callback. Parked resumption and deliberate origins are separate admission
    work; this entry does not invent either from legacy episode metadata.
    """
    view = owner.store.snapshot(capture=False)
    parent = active_inquiry(view, observer_entity_id=grant.observer_entity_id)
    if parent is not None:
        if parent.receipt.state.grant.critical_epoch != grant.critical_epoch:
            return PrivateGraphResult(summary="This inquiry predates accepted Critical reconsideration and requires fresh admission.",
                                      receipt_id=parent.record.id)
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
        nominations = read_nomination_batch(view, batch_id)
        if nominations is None:
            result = NominationBatch.model_validate(run_stage("nominate", request.model_dump(mode="json")))
            nominations = owner.accept_nominations(result, grant=grant, batch_id=batch_id)
        if not nominations.nominations:
            return PrivateGraphResult(summary=nominations.receipt.reason, receipt_id=nominations.record.id)
        # A retained attention outcome is a completed selection call. A later
        # opportunity receives a different captured-context identity.
        view = owner.store.snapshot(capture=False)
        if any(record.metadata.get("thought_attention_batch") == batch_id
               for record in view.all_knoxels.values()):
            return PrivateGraphResult(summary="This nomination opportunity already has an attention outcome.",
                                      receipt_id=nominations.record.id)
        from workflows.inner_thought.stages.trigger.steps import AttentionStageRequest
        attention = PrivateAttentionContext(
            candidates=tuple(AttentionCandidate(nomination_id=item.record.id, nomination=item.receipt.nomination,
                root_source_ids=item.receipt.root_source_ids) for item in nominations.nominations),
            pending_input_ids=pending_input_ids, accepted_focus=focus.focus,
            active_intention_ids=tuple(goal.knoxel_id for goal in request.active_goal_records),
            attention_summary=mental_state.state_cognition.model_dump_json())
        payload = AttentionStageRequest(request=request, attention=attention)
        decision = ThoughtAttentionDecision.model_validate(run_stage("trigger", payload.model_dump(mode="json")))
        validate_private_attention(attention, decision)
        def validate_context() -> None:
            """Keep the supplied focus head and original expiry through selection acceptance."""
            validate_private_focus(owner.store.snapshot(capture=False), focus, now_utc=vclock.utcnow())
        selected = owner.accept_attention(nominations.receipt.nomination_ids, decision,
            commit_id="attention:" + batch_id, episode_id="inquiry:" + batch_id, now_utc=vclock.utcnow(),
            validate_context=validate_context)
        parent = selected.inquiry
        if parent is None:
            return PrivateGraphResult(summary=decision.reason, receipt_id=selected.record.id)
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
