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

An episode that an accepted deliberate action opened runs through the same
steps. Its thought stage also reads the sources the action named, and the
integration that ends it is asked for optional focus advice. The thought stage
is told that main cognition waits for the result and how many operations are
left, and the last operation must state an outcome.
"""
from __future__ import annotations

from typing import Literal
from dataclasses import dataclass
from time import monotonic
import hashlib
import pickle

from pydantic import BaseModel, ConfigDict, Field

from pm import vclock
from pm.memory.snapshot import Snapshot
from workflows.inner_thought.request_views import PrivateInspection
from pm.records.mental_state import FullMentalState, compute_attention_bias
from pm.records.enums import FeatureType
from pm.records.base import RecordScope
from pm.records.memory_records import Feature
from workflows.cognitive_control.attention import CapturedPrivateFocus, validate_private_attention
from workflows.inner_thought.graph_host import (
    PrivateInquiryOwner, accepted_source_ids, active_inquiry, displaced_inquiries, parked_inquiries,
)
from workflows.inner_thought.inquiry import (
    InquiryAlternativeMetadata, InquiryNodeMetadata, InquiryRecord, deliberate_admission, displaced_offer,
    inquiry_origin, parked_offer, read_displaced_inquiry,
)
from workflows.inner_thought.metacognition import (
    METACOGNITION_OPERATIONS, activity_rule, render_thought_activity, thought_activity,
)
from workflows.inner_thought.nominations import (
    NominationBatchRecord, extend_inquiry_grant, original_source_ids, read_nomination_batch,
)
from workflows.inner_thought.offer_salience import RetainedQuestion, attention_cues, select_retained_offer
from workflows.inner_thought.operations import operation_lines, validate_operation
from workflows.inner_thought.perspective import (
    PerspectiveRevisionRequest, capture_perspective_context, perspective_rule, render_perspective_context,
    revision_request,
)
from workflows.inner_thought.rendering import inspection as inspection_material
from workflows.reflection.rendering import private_material
from workflows.reflection.schemas import ReflectionContext
from workflows.inner_thought.graph_outcomes import capture_graph_outcome, graph_outcome_rules
from workflows.inner_thought.pipeline import (
    ModelProposalError, citable_ids, validate_graph_outcome_rules, validate_integration,
)
from workflows.inner_thought.schemas import (
    AttentionCandidate, DisplacedInquiryOffer, FocusAdviceRequest, InquiryAlternative, InquiryGrant, InquiryOrigin,
    InquiryResumeDecision, InquiryState, InquiryStep, InnerRequest,
    ParkedInquiryOffer, PrivateAttentionContext, RetainedOfferSelection, ThoughtAlternative,
    ThoughtCandidate, ThoughtResult, TriggerChoice,
)

from workflows.inner_thought.stage_runner import PrivateStageRunner as StageRunner
from workflows.inner_thought.memory_access import PrivateMemoryAccess, PrivateRecall


class PrivateGraphResult(BaseModel):
    """The parent's accepted graph result; it is never committed as a second thought."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    stage_kind: Literal["private_inquiry_result"] = "private_inquiry_result"
    summary: str
    receipt_id: int | None = None
    finding_id: int | None = None


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


class DeliberateBudget(BaseModel):
    """Where a deliberate episode stands within its operation bound."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    operation_number: int = Field(ge=1, description="One-based number of the operation that is about to run.")
    operation_limit: int = Field(ge=1, description="Most operations the episode can execute before it is parked without an outcome.")
    stalled_operations: int = Field(ge=0, description="Consecutive accepted operations so far without new original evidence or a meaningful comparison.")
    stalled_limit: int = Field(ge=1, description="Consecutive operations without progress at which the episode is parked without an outcome.")


class InquiryPromptGrant(BaseModel):
    """Readable citation authority; persistence fingerprints stay with the host."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    observer_entity_id: int = Field(description="Companion whose captured perspective governs this inquiry.")
    scope: RecordScope = Field(description="Reality and frame within which supplied evidence may be interpreted.")
    permitted_evidence_ids: tuple[int, ...] = Field(description="Supplied record IDs permitted in evidence citations.")


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
    reflection_evidence: str | None = Field(default=None,
        description="Bounded reflection material the host rendered from experiment and episode evidence it "
                    "selected under its captured scope. Prompt text, not a second evidence grant.")
    inspection_evidence: str | None = Field(default=None,
        description="Bounded implementation evidence the host rendered from the inspector's own projection. "
                    "Episode identifiers in it are not citation grants.")
    deliberate_budget: DeliberateBudget | None = Field(default=None,
        description="Set only for deliberate mental work that main cognition selected. The serialized request "
                    "leaves the field out otherwise, so the prompt of a spontaneous inquiry is unchanged.")
    perspective_evidence: str | None = Field(default=None,
        description="Rendered perspective instruction for one theory-of-mind step: the accepted view of the "
                    "subject and the epistemic rules, built by perspective.perspective_rule. Absent for every "
                    "step that is not a perspective inquiry, so other prompts are byte-identical.")
    activity_evidence: str | None = Field(default=None,
        description="Rendered recorded activity for one noticing step, built by metacognition.activity_rule. "
                    "Facts about the companion's own recent thinking; absent for every other step.")
    original_source_ids: tuple[int, ...] = Field(
        description="Granted sources that count as acquired original evidence, by the store owner's rule "
                    "(nominations.original_source_ids). Read only by validate_graph_step and never echoed "
                    "into the prompt.")


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


#: Operations whose step reads recorded experiment and episode evidence.
REFLECTION_OPERATIONS: frozenset[str] = frozenset({"reflection_review"})
#: Operations whose step reads the inspector's implementation evidence.
INSPECTION_OPERATIONS: frozenset[str] = frozenset({"capability_inspection", "episode_inspection"})


def graph_thought_request(view: Snapshot, request: InnerRequest, parent: InquiryRecord,
                          grant: InquiryGrant, *, reflection: ReflectionContext | None = None,
                          inspection: PrivateInspection | None = None) -> GraphThoughtRequest:
    """Supply actual accepted nodes and the evidence the admitted step reads.

    Never invent IDs for new alternatives. Only an operation that reads
    reflection or inspection evidence gets its rendered text, so the prompt of
    every other step stays exactly as it was. The text is rendered here from
    the typed evidence the host read under its captured scope, and the raw
    collections stay out of the serialized request.
    """
    state = parent.receipt.state
    origin = inquiry_origin(view, state)
    nodes = tuple(_node_view(view, node_id) for node_id in state.node_ids)
    operation = state.next_operation
    reflection_evidence = (str(private_material(reflection))
                           if operation in REFLECTION_OPERATIONS else None)
    inspection_evidence = (str(inspection_material(inspection))
                           if operation in INSPECTION_OPERATIONS else None)
    return GraphThoughtRequest(
        request=request,
        state=state,
        grant=grant,
        origin=origin,
        nodes=nodes,
        earlier_artifacts=earlier_artifacts(view, state),
        reflection_evidence=reflection_evidence,
        inspection_evidence=inspection_evidence,
        original_source_ids=original_source_ids(view, grant),
    )


def with_origin_evidence(view: Snapshot, request: InnerRequest, origin: InquiryOrigin) -> InnerRequest:
    """Return the request with the deliberate action's named sources as readable candidates.

    A deliberate episode cites the sources main cognition named with the
    instruction. The private request of a later timer tick need not contain
    them, and the model would then see their IDs without their text. Sources
    the request already supplies are left as they are.
    """
    supplied_ids = citable_ids(request)
    extended = request.model_copy(deep=True)
    for source_id in origin.evidence_ids:
        if source_id in supplied_ids:
            continue
        record = view.get(source_id)
        extended.candidates.append(ThoughtCandidate(knoxel_id=source_id, text=record.content,
            revision=view.revisions[source_id], kind="deliberate_evidence",
            reason="Named with the deliberate instruction."))
        supplied_ids.add(source_id)
    return extended


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
        if len(accepted_nodes) < 2:
            progress_rule += (
                " Fewer than two accepted nodes are available, so discrimination progress is unavailable. "
                "Original expectations and observations are evidence_ids, never compared_node_ids. "
                "You may still review that evidence and state an outcome without claiming discrimination. "
                "Use progress.kind='none', compared_node_ids=[], meaningful_difference=false unless "
                "this step actually inspected new original evidence. Keep the required relation to the "
                "existing node separate from progress.")
    return (
        f"Perform {operation.name} ({operation.mode}) as {request.request.companion_name}'s private inquiry. "
        f"Purpose: {operation.description}\n"
        f"Accepted graph node IDs available for relations and comparisons: {accepted_nodes}.\n"
        + relation_rule + "\n"
        + f"This operation is {operation.mode}. " + branch_rule + "\n"
        + progress_rule + "\n"
        + _earlier_artifact_rule(request)
        + _reflection_rule(request)
        + _perspective_rule(request)
        + _activity_rule(request)
        + _deliberate_rule(request)
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


def _perspective_rule(request: GraphThoughtRequest) -> str:
    """Return the rendered perspective instruction, or nothing for other steps.

    The rules and the accepted view are assembled at the build site, where the
    subject name is known; only the finished block travels in the request. An
    absent field means this step is not a perspective inquiry, so its prompt
    keeps its original bytes.
    """
    if not request.perspective_evidence:
        return ""
    return request.perspective_evidence + "\n"


def _activity_rule(request: GraphThoughtRequest) -> str:
    """Return the rendered activity block of a noticing step, or nothing otherwise."""
    if not request.activity_evidence:
        return ""
    return request.activity_evidence + "\n"


def _reflection_rule(request: GraphThoughtRequest) -> str:
    """State what the recorded reflection and inspection evidence is.

    Without this material a step whose admitted operation is
    ``reflection_review`` reads a recorded prediction's identifier and no
    experiment at all. The text is rendered by the host under the same captured
    scope as the grant; experiment and episode identifiers are not record IDs,
    so the model must not treat them as one.
    """
    blocks: list[str] = []
    if request.reflection_evidence:
        blocks.append(
            "Recorded reflection evidence: experiments with their recorded prediction, retained episodes, and "
            "the sources they name. A source ID this material lists is an allowed citation. An experiment_id "
            "and an episode_id identify that evidence; neither is a record ID:\n" + request.reflection_evidence)
    if request.inspection_evidence:
        blocks.append(
            "Selected implementation evidence. A capability state is observed evidence about this runtime, "
            "not a memory or an external event, and an inspection episode identifier is not a record ID:\n"
            + request.inspection_evidence)
    if not blocks:
        return ""
    return "\n".join(blocks) + "\n"


def _deliberate_rule(request: GraphThoughtRequest) -> str:
    """Tell deliberate mental work that its result is awaited and how much room is left.

    Without this rule the model keeps choosing another operation until the
    episode is parked at its bound, and main cognition receives no conclusion
    for the work it selected. The last operation therefore has to state where
    the thinking stands, even when that is an open gap.
    """
    budget = request.deliberate_budget
    if budget is None:
        return ""
    rule = (
        f"This inquiry is deliberate mental work that {request.request.companion_name} chose to do. "
        "Main cognition waits for its result. "
        f"This is operation {budget.operation_number} of at most {budget.operation_limit}. "
        "Reach an outcome within them instead of planning further operations you cannot run. ")
    rule += (
        "Progress means new original evidence or a meaningful comparison of accepted graph nodes. "
        f"{budget.stalled_operations} of {budget.stalled_limit} consecutive operations without progress are used; "
        f"at {budget.stalled_limit} the work is parked without a result. ")
    if budget.operation_number >= budget.operation_limit:
        rule += (
            "This is the last operation, so outcome='continue' is not available. State where the thinking "
            "stands: outcome='finding' or 'completed' when an accepted comparison supports it, otherwise "
            "outcome='information_gap' with the missing evidence, or outcome='pause'. An honest open gap is a "
            "valid result. ")
    elif budget.stalled_operations + 1 >= budget.stalled_limit:
        rule += (
            "This operation must make such progress or state an outcome. outcome='continue' together with "
            "progress.kind='none' is not available, because it would park the work without a result. ")
    return rule + "\n"


def validate_graph_step(request: GraphThoughtRequest, step: InquiryStep) -> InquiryStep:
    """Reject stochastic semantic violations inside the child's repair boundary.

    The store owner still validates original source roots, current parents,
    and durable invariants. This check does not catch or disguise those bugs.
    """
    state = request.state
    if step.operation != state.next_operation:
        raise ModelProposalError("Execute the admitted next operation, without replacing it.")
    budget = request.deliberate_budget
    if budget is not None and budget.operation_number >= budget.operation_limit and step.outcome == "continue":
        raise ModelProposalError(
            "This is the last operation of deliberate mental work, so it cannot continue. State an outcome: "
            "'finding' or 'completed' when an accepted comparison supports it, otherwise 'information_gap' with "
            "the missing evidence, or 'pause'. Leave next_operation empty.")
    if (budget is not None and budget.stalled_operations + 1 >= budget.stalled_limit
            and step.outcome == "continue" and step.progress.kind == "none"):
        raise ModelProposalError(
            "This operation makes no progress and would park the deliberate mental work without a result. "
            "State an outcome instead: 'information_gap' with the missing evidence, or 'pause', and leave "
            "next_operation empty. A meaningful comparison of at least two accepted nodes is the other way on, "
            "when this operation is discriminatory.")
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
    if step.progress.kind == "new_evidence":
        # The store owner counts only an original record as acquired evidence.
        # An earlier thought or a summary restates evidence; it is not new.
        derived = sorted(set(step.progress.inspected_source_ids) - set(request.original_source_ids))
        if derived:
            raise ModelProposalError(
                f"progress.inspected_source_ids names {derived}, which are earlier thoughts, summaries, or other "
                "internal derivations, not original sources, so inspecting them is not new evidence. The original "
                f"source IDs of this step are {sorted(request.original_source_ids)}. Name only those, or set "
                "progress.kind='none' with inspected_source_ids=[] when this step inspected no original source.")
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
                        memory_access: PrivateMemoryAccess | None) -> DisplacedCandidate | None:
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
        if memory_access is None or memory_access.embeddings is None:
            raise RuntimeError("Rating several displaced questions requires the host's embedding endpoint.")
        questions: list[RetainedQuestion] = []
        for receipt_id, question in waiting.items():
            state = question.inquiry.receipt.state
            origin = inquiry_origin(view, state)
            questions.append(RetainedQuestion(receipt_id=receipt_id, text=origin.text + "\n" + origin.question,
                                              source_ids=state.root_source_ids))
        cues = attention_cues(request)
        # One request embeds the questions first and the cues after them.
        vectors = memory_access.embed_texts([item.text for item in questions] + [cue.text for cue in cues])
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
                      mental_state: FullMentalState, memory_access: PrivateMemoryAccess | None = None) -> OpportunityAdmission:
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
        memory_access=memory_access)
    displaced_selection = None if displaced is None else displaced.selection
    attention = PrivateAttentionContext(
        candidates=tuple(AttentionCandidate(nomination_id=item.record.id, nomination=item.receipt.nomination,
            root_source_ids=item.receipt.root_source_ids) for item in nominations.nominations),
        parked=() if candidate is None else (candidate.offer,),
        displaced=() if displaced is None else (displaced.offer,),
        pending_input_ids=pending_input_ids, accepted_focus=focus.focus,
        active_intention_ids=tuple(goal.knoxel_id for goal in request.active_goal_records),
        attention_summary=(f"Interlocus {mental_state.state_cognition.interlocus:.3f}; "
            f"mental aperture {mental_state.state_cognition.mental_aperture:.3f}; "
            f"ego strength {mental_state.state_cognition.ego_strength:.3f}; "
            f"willpower {mental_state.state_cognition.willpower:.3f}."))
    payload = AttentionStageRequest(request=request, attention=attention)
    decision = run_stage.attention(payload)
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

    selected = owner.accept_attention(nominations.receipt.nomination_ids, decision,
        commit_id="attention:" + batch_id, episode_id="inquiry:" + batch_id, now_utc=vclock.utcnow(),
        focus=focus, displaced_selection=displaced_selection)
    if selected.inquiry is None:
        return OpportunityAdmission(None, PrivateGraphResult(summary=decision.reason, receipt_id=selected.record.id))
    return OpportunityAdmission(selected.inquiry, None)


def run_graph_opportunity(request: InnerRequest, *, owner: PrivateInquiryOwner,
                          grant: InquiryGrant, run_stage: StageRunner,
                          reflection: ReflectionContext | None = None,
                          inspection: PrivateInspection | None = None,
                          mental_state: FullMentalState, control_generation: int,
                          pending_input_ids: tuple[int, ...], focus: CapturedPrivateFocus,
                          memory_access: PrivateMemoryAccess | None = None) -> PrivateGraphResult:
    """Retain nomination, selection, and one executed step as separate boundaries.

    Continuation uses the accepted graph rather than main acknowledgment of
    every internal node, whether the episode began from a private nomination or
    from an accepted deliberate action. The host retains control authority
    through its commit callback. Parked work re-enters only through the shared
    attention decision in :func:`admit_opportunity`; this entry never resumes it
    from elapsed parking or legacy episode metadata.

    Work admitted under an older Critical epoch is closed first, durably and
    without a model call, and so is work whose retained grant names a changed
    source. The slot is then free for this same opportunity, and the closed
    question competes in its attention decision. ``embed_texts`` is
    needed only when several displaced questions wait and one must be chosen.

    ``reflection`` and ``inspection`` are the scoped evidence the host read.
    Only an operation that reads them receives their rendered text, and only
    a reflection step extends its citation grant with the reflection sources.
    """
    owner.displace_superseded(observer_entity_id=grant.observer_entity_id, critical_epoch=grant.critical_epoch)
    # A changed goal, expectation, or source of a retained grant would refuse
    # every later step at commit. Close such work before any model call.
    owner.close_outdated(observer_entity_id=grant.observer_entity_id)
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
        material = pickle.dumps((grant, request.main_frame.tick, tuple(request.trigger_candidates)), protocol=5)
        batch_id = hashlib.sha256(material).hexdigest()
        # The same qualifying opportunity also gates reconsideration of parked
        # work, so a parked question never causes an extra model call by itself.
        nominations = read_nomination_batch(view, batch_id)
        if nominations is None:
            result = run_stage.nominate(request)
            nominations = owner.accept_nominations(result, grant=grant, batch_id=batch_id)
        admission = admit_opportunity(request, owner=owner, grant=grant, run_stage=run_stage,
            nominations=nominations, batch_id=batch_id, pending_input_ids=pending_input_ids,
            focus=focus, mental_state=mental_state, memory_access=memory_access)
        if admission.inquiry is None:
            assert admission.result is not None  # Every non-admission names its own outcome.
            return admission.result
        parent = admission.inquiry
    node_grant = owner.execution_grant(parent)
    dependencies = {source.record_id: source for source in grant.sources}
    dependencies.update({source.record_id: source for source in node_grant.sources})
    execution_grant = InquiryGrant(
        observer_entity_id=node_grant.observer_entity_id,
        scope=node_grant.scope,
        sources=tuple((dependencies[key] for key in sorted(dependencies))),
        critical_epoch=node_grant.critical_epoch,
    )
    step_view = owner.store.snapshot(capture=False)
    if parent.receipt.state.next_operation in REFLECTION_OPERATIONS:
        # A reflection step reads experiment and episode evidence, and citing it
        # needs the captured scope of those records. Every other step keeps its
        # original grant, so its prompt is unchanged.
        reflection_ids = tuple(item.knoxel_id for item in (reflection.records if reflection is not None else []))
        if reflection_ids:
            execution_grant = extend_inquiry_grant(step_view, execution_grant, reflection_ids)
    supplied = graph_thought_request(step_view, request, parent, execution_grant,
                                     reflection=reflection, inspection=inspection)
    deliberate = supplied.origin.kind == "deliberate_action"
    #: Optional revision question of a perspective episode, kept until the
    #: outcome, where the integrate stage is the only caller that may answer it.
    revision: PerspectiveRevisionRequest | None = None
    revision_texts: dict[int, str] = {}
    if deliberate:
        request = with_origin_evidence(step_view, request, supplied.origin)
        deliberate_state = parent.receipt.state
        budget = DeliberateBudget(operation_number=deliberate_state.sequence + 1,
            operation_limit=owner.limits.episode_steps, stalled_operations=deliberate_state.stalled_steps,
            stalled_limit=owner.limits.stalled_steps)
        supplied = GraphThoughtRequest(
            stage_kind=supplied.stage_kind,
            request=request,
            state=supplied.state,
            grant=supplied.grant,
            origin=supplied.origin,
            nodes=supplied.nodes,
            earlier_artifacts=supplied.earlier_artifacts,
            recall=supplied.recall,
            reflection_evidence=supplied.reflection_evidence,
            inspection_evidence=supplied.inspection_evidence,
            deliberate_budget=budget,
            perspective_evidence=supplied.perspective_evidence,
            activity_evidence=supplied.activity_evidence,
            original_source_ids=supplied.original_source_ids,
        )
        admission = deliberate_admission(step_view, parent.receipt.state)
        if admission is not None and admission.subject_entity_id is not None:
            # One theory-of-mind step reads the accepted model of its subject
            # and the original messages behind it. The sources become citable
            # candidates and extend the step grant, so citing them is legal;
            # the finished rule block travels in the request, and no other
            # step's prompt changes.
            perspective = capture_perspective_context(step_view,
                subject_entity_id=admission.subject_entity_id, grant=execution_grant,
                accepted_ids=accepted_source_ids(owner.store), as_of_utc=vclock.utcnow())
            if perspective.evidence_source_ids:
                execution_grant = extend_inquiry_grant(step_view, execution_grant,
                                                       perspective.evidence_source_ids)
                request = request.model_copy(deep=True)
                supplied_ids = citable_ids(request)
                for source_id in perspective.evidence_source_ids:
                    if source_id in supplied_ids:
                        continue
                    record = step_view.get(source_id)
                    request.candidates.append(ThoughtCandidate(knoxel_id=source_id, text=record.content,
                        revision=step_view.revisions[source_id], kind="entity_evidence",
                        reason="Original message behind the subject's accepted claim."))
                    supplied_ids.add(source_id)
            revision = revision_request(step_view, perspective, execution_grant)
            revision_texts = {source_id: step_view.get(source_id).content
                              for source_id in revision.citable_source_ids}
            perspective_text = render_perspective_context(perspective)
            supplied = GraphThoughtRequest(
                stage_kind=supplied.stage_kind,
                request=request,
                state=supplied.state,
                grant=execution_grant,
                origin=supplied.origin,
                nodes=supplied.nodes,
                earlier_artifacts=supplied.earlier_artifacts,
                recall=supplied.recall,
                reflection_evidence=supplied.reflection_evidence,
                inspection_evidence=supplied.inspection_evidence,
                deliberate_budget=supplied.deliberate_budget,
                perspective_evidence=perspective_rule(perspective_text, companion_name=request.companion_name, subject_name=perspective.subject_name),
                activity_evidence=supplied.activity_evidence,
                original_source_ids=supplied.original_source_ids,
            )
    if parent.receipt.state.next_operation in METACOGNITION_OPERATIONS:
        # A noticing step reads the recorded facts about the companion's own
        # recent thinking. The facts are built without a model call; the rule
        # block travels in the request, and no other step's prompt changes.
        activity = thought_activity(request.recent_thoughts)
        supplied = GraphThoughtRequest(
            stage_kind=supplied.stage_kind,
            request=supplied.request,
            state=supplied.state,
            grant=supplied.grant,
            origin=supplied.origin,
            nodes=supplied.nodes,
            earlier_artifacts=supplied.earlier_artifacts,
            recall=supplied.recall,
            reflection_evidence=supplied.reflection_evidence,
            inspection_evidence=supplied.inspection_evidence,
            deliberate_budget=supplied.deliberate_budget,
            perspective_evidence=supplied.perspective_evidence,
            activity_evidence=activity_rule(render_thought_activity(activity)),
            original_source_ids=supplied.original_source_ids,
        )
    else:
        activity = None
    started = monotonic()
    if parent.receipt.state.next_operation == "memory_search":
        if memory_access is None:
            raise RuntimeError("An admitted memory search requires the host's actual frozen-view retrieval.")
        recalled = memory_access.recall(supplied.origin.question)
        recalled_ids = tuple(source.record_id for source in recalled.sources)
        execution_grant = extend_inquiry_grant(memory_access.view, execution_grant, recalled_ids)
        supplied = GraphThoughtRequest(
            stage_kind=supplied.stage_kind,
            request=supplied.request,
            state=supplied.state,
            grant=execution_grant,
            origin=supplied.origin,
            nodes=supplied.nodes,
            earlier_artifacts=supplied.earlier_artifacts,
            recall=recalled,
            reflection_evidence=supplied.reflection_evidence,
            inspection_evidence=supplied.inspection_evidence,
            deliberate_budget=supplied.deliberate_budget,
            perspective_evidence=supplied.perspective_evidence,
            activity_evidence=supplied.activity_evidence,
            original_source_ids=supplied.original_source_ids,
        )
        request = request.model_copy(deep=True)
        existing_ids = citable_ids(request)
        for match in recalled.matches:
            if match.knoxel_id not in existing_ids:
                request.candidates.append(ThoughtCandidate(knoxel_id=match.knoxel_id, text=match.text,
                    revision=match.revision, kind="recalled_memory"))
                existing_ids.add(match.knoxel_id)
        supplied = GraphThoughtRequest(
            stage_kind=supplied.stage_kind,
            request=request,
            state=supplied.state,
            grant=supplied.grant,
            origin=supplied.origin,
            nodes=supplied.nodes,
            earlier_artifacts=supplied.earlier_artifacts,
            recall=supplied.recall,
            reflection_evidence=supplied.reflection_evidence,
            inspection_evidence=supplied.inspection_evidence,
            deliberate_budget=supplied.deliberate_budget,
            perspective_evidence=supplied.perspective_evidence,
            activity_evidence=supplied.activity_evidence,
            original_source_ids=supplied.original_source_ids,
        )
    # The grant may have grown above (perspective, recall). The in-loop check
    # must judge original evidence against the grant the store owner receives.
    supplied = GraphThoughtRequest(
        stage_kind=supplied.stage_kind,
        request=supplied.request,
        state=supplied.state,
        grant=supplied.grant,
        origin=supplied.origin,
        nodes=supplied.nodes,
        earlier_artifacts=supplied.earlier_artifacts,
        recall=supplied.recall,
        reflection_evidence=supplied.reflection_evidence,
        inspection_evidence=supplied.inspection_evidence,
        deliberate_budget=supplied.deliberate_budget,
        perspective_evidence=supplied.perspective_evidence,
        activity_evidence=supplied.activity_evidence,
        original_source_ids=original_source_ids(step_view, execution_grant),
    )
    # The rendered evidence fields carry the prompt material. The raw typed
    # collections behind them stay out of the serialized request, so no store
    # record reaches the prompt without its renderer.
    step = run_stage.inquiry(supplied)
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
        from workflows.inner_thought.stages.integrate.steps import IntegrationEnvelope
        from workflows.inner_thought.request_views import OperationSuggestion
        integration_ids = citable_ids(integration_request)
        focus_advice = (FocusAdviceRequest(instruction=supplied.origin.text,
            citable_ids=tuple(sorted(integration_ids))) if deliberate else None)
        rules = graph_outcome_rules(step_view, integration_request, integration_ids, grant=execution_grant)
        integration = run_stage.integrate(IntegrationEnvelope(request=integration_request,
            thought=thought, trigger=trigger, operation=OperationSuggestion(name=step.operation),
            focus_advice=focus_advice, perspective_revision=revision, graph_outcome_rules=rules))
        validate_integration(integration, integration_ids, integration_request, focus_requested=deliberate,
                             perspective=revision, source_texts=revision_texts if revision is not None else None)
        validate_graph_outcome_rules(integration, rules, integration_ids)
        outcome = capture_graph_outcome(execution_grant, step, integration)
    active_seconds = monotonic() - started
    accepted = owner.accept_step(parent, step, grant=execution_grant,
        commit_id=f"inquiry-step:{parent.record.id}", active_seconds=active_seconds,
        now_utc=vclock.utcnow(), mental_state=mental_state, control_generation=control_generation,
        outcome=outcome, activity=activity)
    boundary = step.outcome != "continue" or step.operation in METACOGNITION_OPERATIONS
    return PrivateGraphResult(summary=accepted.receipt.state.stop_reason or step.reason,
        receipt_id=accepted.record.id, finding_id=accepted.receipt.operation_record_id if boundary else None)
