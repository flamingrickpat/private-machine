"""Accept temporary focus advice through ordinary attention processing."""

from pm import vclock
from pm.memory.snapshot import GhostConflict, RecordView, Snapshot
from pm.memory.store import MemoryStore
from pm.memory.write_batch import WriteBatch
from pm.records.lineage import RecordLineage
from pm.records.memory_records import Feature

from pm.records.copying import pickle_copy
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pydantic import BaseModel, ConfigDict

from pm.constants import Sampler
from pm.context_render.turns import Turn
from workflows.cognitive_control.schemas import FocusRejection, FocusState
from pm.llm.structured import complete_json
from pm.llm.generation import GenerationSettings
from pm.llm.role_model import role_model
from pm_workflows.python_role import RoleContext
from pm.context_render.prompt import RendererCapture, RendererInput
from workflows.cognitive_control.rendering import FocusEvidence, FocusRenderInput, render_focus
from workflows.main_cognition.schemas import FocusProposalRef
from workflows.inner_thought.operations import validate_operation
from workflows.inner_thought.schemas import PrivateAttentionContext, ThoughtAttentionDecision


class FocusDecision(BaseModel):
    """An appraisal-aware acceptance or rejection of one mental proposal."""

    accept: bool
    reason: str


class PrivateFocusChanged(GhostConflict):
    """The frozen private selection no longer uses effective accepted focus."""


class _PrivateFocusMetadata(BaseModel):
    """Read the advice owner's required fields from heterogeneous metadata."""

    model_config = ConfigDict(extra="ignore")
    focus: FocusState
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class CapturedPrivateFocus:
    """Effective focus and its exact frozen source, without a live store read.

    An expired latest record supplies no effective advice or future deadline.
    Its source identity still records which family head the selector observed.
    """

    focus: FocusState | None
    source_id: int | None
    expires_at: datetime | None


def captured_private_focus(view: Snapshot, now_utc: datetime, *,
                           maximum_strength: float | None = None) -> CapturedPrivateFocus:
    """Read private selection advice only from its original frozen snapshot.

    The caller adds the returned source to its governing grant. Family identity
    and expiry need a separate check because advice appends independent records.
    """
    if now_utc.tzinfo is not None:
        raise ValueError("Private focus capture requires naive UTC world time.")
    record = next((item for item in reversed(list(view.all_knoxels.values()))
                   if item.metadata.get("control_kind") == "attention_focus"), None)
    if record is None:
        return CapturedPrivateFocus(None, None, None)
    metadata = _PrivateFocusMetadata.model_validate(record.metadata)
    if metadata.expires_at.tzinfo is not None:
        raise ValueError("Accepted focus expiry requires naive UTC world time.")
    if now_utc >= metadata.expires_at:
        return CapturedPrivateFocus(None, record.id, None)
    focus = metadata.focus.model_copy(deep=True)
    focus.fresh = False
    focus.control_record_id = record.id
    if maximum_strength is not None and focus.accept:
        focus.strength = min(focus.strength, maximum_strength)
    return CapturedPrivateFocus(focus, record.id, metadata.expires_at)


def validate_private_focus(view: Snapshot, captured: CapturedPrivateFocus, *, now_utc: datetime) -> None:
    """Reject replaced or expired guidance before private attention commits.

    The host holds the store lock through this check and selection acceptance.
    Original source guards separately retain the consumed record's meaning.
    """
    current = captured_private_focus(view, now_utc)
    if current.source_id != captured.source_id:
        raise PrivateFocusChanged("Accepted private focus changed after selection context was frozen.")
    if captured.expires_at is not None and now_utc >= captured.expires_at:
        raise PrivateFocusChanged("Accepted private focus expired during selection.")


def validate_private_attention(context: PrivateAttentionContext,
                               decision: ThoughtAttentionDecision) -> ThoughtAttentionDecision:
    """Admit one supplied nomination, one offered parked inquiry, or one displaced question, or abstain.

    Nomination scores express separate judgments and confer no selection
    authority. Retained parked work competes for the same private slot and wins
    only on this same decision; elapsed parking alone never resumes it. A
    question displaced by Critical reconsideration competes there too and
    enters as a new episode. The host supplies current shared attention
    guidance and keeps Critical, wait, source, and private-slot guards through
    acceptance.
    """
    offered_ids = [candidate.nomination_id for candidate in context.candidates]
    if len(set(offered_ids)) != len(offered_ids):
        raise ValueError("Private attention requires unique committed nomination identities.")
    offered_parked = [offer.receipt_id for offer in context.parked]
    if decision.nomination_id is not None and decision.nomination_id not in offered_ids:
        raise ValueError("Private attention names a nomination outside the supplied batch.")
    if decision.parked_receipt_id is not None and decision.parked_receipt_id not in offered_parked:
        raise ValueError("Private attention names parked work outside the supplied offers.")
    offered_displaced = [offer.receipt_id for offer in context.displaced]
    if decision.displaced_receipt_id is not None and decision.displaced_receipt_id not in offered_displaced:
        raise ValueError("Private attention names a displaced question outside the supplied offers.")
    if decision.disposition == "select":
        if context.blocked_reason:
            raise ValueError(f"Private attention cannot allocate the slot: {context.blocked_reason}")
        # A selection must already have named one identity, and the three checks
        # above require that identity to come from what was actually offered.
        if decision.parked_receipt_id is not None or decision.displaced_receipt_id is not None:
            # Reject a name outside the delivered catalogue before the paid thought call.
            validate_operation(decision.admitted_operation)
    return decision


#: The instruction of the focus check. It comes after the shared story prefix of main cognition.
FOCUS_INSTRUCTION = (
    "The character's own slower, deliberate thinking proposes a change of focus for this moment. It is the "
    "character's own suggestion, not a command from anyone in the story:\n{proposal}\n\n"
    "What the mind holds right now:\n{evidence}\n\n"
    "Your task: act as the focus check of the character's mind. Decide whether the character's mind takes this "
    "deliberate focus proposal now, given the feelings, the competing impulses, the goals, and the new events. "
    "The mind can refuse it. A focus change cannot erase a feeling, a memory, or a pressing concern; it only "
    "shifts attention for a while.\n"
    'Answer with only this JSON object and nothing else: {{"accept": true, "reason": "one sentence"}}')


def resolve_focus(context: RoleContext, active: FocusState | None, proposals: list[FocusProposalRef],
                  evidence: FocusEvidence, prefix: Sequence[Turn] = ()) -> FocusState | None:
    """Return effective advice after appraisal and codelet contextualization.

    ``active`` is the stored advice that is still in effect, or None.
    ``proposals`` are the fresh proposals that this cycle's stimuli carry.
    ``evidence`` is the cycle context that the model reads beside each
    proposal. ``prefix`` is the shared story prefix of main cognition: the
    focus check is a story-framed call, and its instruction comes last. The
    result is None when no advice and no rejection exist.

    The model decides whether fresh advice fits. Its answer cannot widen the
    proposal's source grant or raise its strength. Existing advice expires
    through the host's real-time filter and requires no repeated model call.

    A fresh rejection never hides older advice that is still in effect. The
    result keeps the accepted advice in its fields and lists every fresh
    rejection under ``rejections``, so both receipts remain visible.
    Each proposal costs one model call through ``complete_json``.
    """
    effective = pickle_copy(active) if active is not None else None
    # Keep recent rejection receipts bounded across accepted advice revisions.
    rejections: list[FocusRejection] = list(effective.rejections)[-4:] if effective is not None else []
    for item in proposals:
        proposal = item.focus
        session = role_model(context.runtime)
        settings = session.render_settings()
        input = FocusRenderInput(tuple(prefix), FOCUS_INSTRUCTION, proposal, evidence)
        with RendererCapture(session.capture_settings("cognitive-focus"), RendererInput("cognitive focus", input, settings)) as capture:
            rendered = render_focus(input, settings)
            capture.save_output(rendered)
        result = complete_json(session, rendered.turns, FocusDecision,
                               settings=GenerationSettings(temperature=Sampler.TempInstruct, enable_thinking=False))
        if result.accept:
            effective = FocusState(instruction=proposal.instruction, promote_ids=list(proposal.promote_ids),
                reduce_ids=list(proposal.reduce_ids), strength=proposal.strength, seconds=proposal.seconds,
                topic_hints=list(proposal.topic_hints), reason=proposal.reason, proposal_id=item.record_id,
                accept=True, decision_reason=result.reason, fresh=True)
        else:
            rejection = FocusRejection(proposal_id=item.record_id, decision_reason=result.reason,
                                       instruction=proposal.instruction)
            rejections.append(rejection)
            if effective is None:
                # No advice is in effect, so the rejection is the whole result.
                effective = FocusState(proposal_id=rejection.proposal_id, accept=rejection.accept,
                    decision_reason=rejection.decision_reason, fresh=rejection.fresh, instruction=rejection.instruction)
    if effective is not None:
        effective.rejections = rejections[-4:]
    return effective


def active_focus(store: MemoryStore, now: datetime | None = None, *,
                 maximum_strength: float | None = None) -> FocusState | None:
    """Read the latest accepted advice, or None. Expired advice has no effect.

    The advice record stores a ``FocusState`` as ``metadata["focus"]``. The
    result is not fresh and names the advice record as
    ``control_record_id``. A lower operator cap applies to stored advice too.
    Clamping here means an old proposal cannot keep an influence the current
    configuration forbids.
    """
    now = now or vclock.utcnow()
    with store.lock:
        record = next((item for item in reversed(list(store.all_knoxels.values()))
                       if item.metadata.get("control_kind") == "attention_focus"), None)
        if record is None or now >= datetime.fromisoformat(record.metadata["expires_at"]):
            return None
        focus = FocusState.model_validate(record.metadata["focus"])
        if maximum_strength is not None and focus.accept:
            focus.strength = min(focus.strength, float(maximum_strength))
        captured = focus.model_copy(deep=True)
        captured.fresh = False
        captured.control_record_id = record.id
        return captured


def focus_versions(view: RecordView) -> dict[int, int]:
    """Return the revision set of the advice owner from a frozen view."""
    return {record.id: view.revisions[record.id] for record in view.all_knoxels.values()
            if record.metadata.get("control_kind") == "attention_focus"}


@dataclass(frozen=True, slots=True)
class FocusStagingSettings:
    experiment_id: int | None = None
    lineage: RecordLineage | None = None


def stage_focus(batch: WriteBatch, view: RecordView, *, instruction: str, focus: FocusState, companion_name: str,
                expires_at: str, settings: FocusStagingSettings = FocusStagingSettings()) -> Feature:
    """Stage a new advice version and reject concurrent advice at commit.

    The caller supplies an absolute UTC expiry. Reversal preserves that expiry
    so it cannot extend old advice. The caller owns cancellation and commit.
    """
    from pm.records.enums import FeatureType, SourceKind
    from pm.records.base import Provenance
    batch.expect_metadata_version("control_kind", "attention_focus", focus_versions(view))
    return batch.create(Feature(content=instruction, feature_type=FeatureType.StateUpdate,
        override_source=companion_name, causal=True, lineage=settings.lineage,
        provenance=Provenance(source_kind=SourceKind.Self, source_ref=companion_name),
        metadata={"memory_kind": "executed_event", "executed_at": vclock.utcnow().isoformat(),
                  "control_kind": "attention_focus", "focus": focus.model_dump(mode="json"), "expires_at": expires_at,
                  "reflection_experiment_id": settings.experiment_id}))
