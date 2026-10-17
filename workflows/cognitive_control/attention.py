"""Accept temporary focus advice through ordinary attention processing."""
from datetime import datetime
import json
from pydantic import BaseModel

from workflows.cognitive_control.schemas import FocusProposal
from workflows.main_cognition.stage_support import complete_json


class FocusDecision(BaseModel):
    """An appraisal-aware acceptance or rejection of one mental proposal."""

    accept: bool
    reason: str


def resolve_focus(context, request: dict) -> dict:
    """Return effective advice after appraisal and codelet contextualization.

    The model decides whether fresh advice fits. Its answer cannot widen the
    proposal's source grant or raise its strength. Existing advice expires
    through the host's real-time filter and requires no repeated model call.

    A fresh rejection never hides older advice that is still in effect. The
    result keeps the accepted advice in its top-level fields and lists every
    fresh rejection under ``rejections``, so both receipts remain visible.
    """
    effective = dict(request.get("active_focus") or {})
    # Keep recent rejection receipts bounded across accepted advice revisions.
    rejections: list[dict] = list(effective.pop("rejections", []) or [])[-4:]
    proposals = request.get("focus_proposals") or []
    for item in proposals:
        proposal = FocusProposal.model_validate(item["focus"])
        result = complete_json(context, [
            {"role": "system", "content": "Decide whether to accept this deliberate focus proposal for this cycle. "
             "It is your own suggestion, not a user command. Consider the current appraisal, competing codelets, "
             "goals, and new evidence. You can reject it. It cannot erase an emotion, memory, or important concern. "
             "Accepting advice changes a temporary attention bias. Return JSON with accept and reason."},
            {"role": "user", "content": json.dumps({"proposal": proposal.model_dump(),
                "appraisal": request.get("appraisal"), "codelets": request.get("codelets"),
                "goals": request.get("active_goals"), "stimuli": request.get("stimuli")}, ensure_ascii=False)},
        ], FocusDecision, temperature=0.2, enable_thinking=False)
        if result.accept:
            effective = {**proposal.model_dump(), "proposal_id": item["record_id"],
                         "accept": True, "decision_reason": result.reason, "fresh": True}
        else:
            rejection = {"proposal_id": item["record_id"], "accept": False,
                         "decision_reason": result.reason, "fresh": True,
                         "instruction": proposal.instruction}
            rejections.append(rejection)
            if not effective:
                # No advice is in effect, so the rejection is the whole result.
                effective = rejection
    rejections = rejections[-4:]
    if effective:
        effective["rejections"] = rejections
    elif rejections:
        effective = {**rejections[-1], "rejections": rejections}
    return effective


def attention_bias(nominations, focus: dict, *, danger: bool) -> dict[str, float]:
    """Bound deliberate influence and preserve urgent competing concerns."""
    if not focus.get("accept"):
        return {}
    promote, reduce = set(focus.get("promote_ids", [])), set(focus.get("reduce_ids", []))
    strength = min(0.5, max(0.0, float(focus.get("strength", 0))))
    biases = {}
    for item in nominations:
        targets = set(item.target_ids)
        bias = strength if targets & promote else 0.0
        if targets & reduce and not danger and item.urgency < 0.8:
            bias -= strength
        biases[item.nomination_id] = bias
    return biases


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
