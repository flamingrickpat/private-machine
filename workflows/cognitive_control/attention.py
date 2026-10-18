"""Accept temporary focus advice through ordinary attention processing."""
from datetime import datetime
import json
from pydantic import BaseModel

from pm.constants import Sampler
from workflows.cognitive_control.schemas import FocusRejection, FocusState
from workflows.main_cognition.model_json import complete_json
from workflows.main_cognition.schemas import FocusProposalRef


class FocusDecision(BaseModel):
    """An appraisal-aware acceptance or rejection of one mental proposal."""

    accept: bool
    reason: str


def resolve_focus(context, active: FocusState | None, proposals: list[FocusProposalRef],
                  evidence: BaseModel) -> FocusState | None:
    """Return effective advice after appraisal and codelet contextualization.

    ``active`` is the stored advice that is still in effect, or None.
    ``proposals`` are the fresh proposals that this cycle's stimuli carry.
    ``evidence`` is the cycle context that the model reads beside each
    proposal. The result is None when no advice and no rejection exist.

    The model decides whether fresh advice fits. Its answer cannot widen the
    proposal's source grant or raise its strength. Existing advice expires
    through the host's real-time filter and requires no repeated model call.

    A fresh rejection never hides older advice that is still in effect. The
    result keeps the accepted advice in its fields and lists every fresh
    rejection under ``rejections``, so both receipts remain visible.
    Each proposal costs one model call through ``complete_json``.
    """
    effective = active.model_copy(deep=True) if active is not None else None
    # Keep recent rejection receipts bounded across accepted advice revisions.
    rejections: list[FocusRejection] = list(effective.rejections)[-4:] if effective is not None else []
    for item in proposals:
        proposal = item.focus
        result = complete_json(context, [
            {"role": "system", "content": "Decide whether to accept this deliberate focus proposal for this cycle. "
             "It is your own suggestion, not a user command. Consider the current appraisal, competing codelets, "
             "goals, and new evidence. You can reject it. It cannot erase an emotion, memory, or important concern. "
             "Accepting advice changes a temporary attention bias. Return JSON with accept and reason."},
            {"role": "user", "content": json.dumps({"proposal": proposal.model_dump(),
                **evidence.model_dump(mode="json")}, ensure_ascii=False)},
        ], FocusDecision, temperature=Sampler.TempInstruct, enable_thinking=False)
        if result.accept:
            effective = FocusState(**proposal.model_dump(), proposal_id=item.record_id,
                                   accept=True, decision_reason=result.reason, fresh=True)
        else:
            rejection = FocusRejection(proposal_id=item.record_id, decision_reason=result.reason,
                                       instruction=proposal.instruction)
            rejections.append(rejection)
            if effective is None:
                # No advice is in effect, so the rejection is the whole result.
                effective = FocusState(**rejection.model_dump())
    if effective is not None:
        effective.rejections = rejections[-4:]
    return effective


def active_focus(store, now=None, *, maximum_strength: float | None = None) -> dict:
    """Read the latest accepted advice. Expired advice has no effect.

    A lower operator cap applies to stored advice too. Clamping here means an
    old proposal cannot keep an influence the current configuration forbids.
    """
    now = now or datetime.utcnow()
    with store.lock:
        record = next((item for item in reversed(list(store.all_knoxels.values()))
                       if item.metadata.get("control_kind") == "attention_focus"), None)
        if record is None or now >= datetime.fromisoformat(record.metadata["expires_at"]):
            return {}
        focus = dict(record.metadata["focus"])
        if maximum_strength is not None and focus.get("accept"):
            focus["strength"] = min(float(focus.get("strength", 0.0)), float(maximum_strength))
        return {**focus, "fresh": False, "control_record_id": record.id}


def focus_versions(view) -> dict[int, int]:
    """Return the revision set of the advice owner from a frozen view."""
    return {record.id: view.revisions[record.id] for record in view.all_knoxels.values()
            if record.metadata.get("control_kind") == "attention_focus"}


def stage_focus(batch, view, *, instruction: str, focus: dict, companion_name: str,
                expires_at: str, experiment_id: int | None = None, lineage=None):
    """Stage a new advice version and reject concurrent advice at commit.

    The caller supplies an absolute UTC expiry. Reversal preserves that expiry
    so it cannot extend old advice. The caller owns cancellation and commit.
    """
    from pm.records.enums import FeatureType, SourceKind
    from pm.records.memory_records import new_feature

    batch.expect_metadata_version("control_kind", "attention_focus", focus_versions(view))
    return new_feature(batch._ghost, instruction, FeatureType.StateUpdate,
        source=companion_name, source_kind=SourceKind.Self, changes=batch, lineage=lineage,
        metadata={"control_kind": "attention_focus", "focus": focus, "expires_at": expires_at,
                  "reflection_experiment_id": experiment_id})
