"""Turn accepted cycle proposals into committed MemoryStore state.

Main cognition and the future inner loop share these helpers. They keep the
accepted state path in one place:

- an appraisal writes one appraisal Feature that carries the appraisal state,
  the delta, and its evidence. The appraise phase already computed the EMA
  mental state of the cycle;
- a goal change creates, completes, or reprioritises an ``Intention`` knoxel
  that has a stable identity and a lifecycle status.

The functions operate on one open ``WriteBatch``. They never commit, send
bus messages, or call a model. The caller commits after it adds the GWT turn
and the cycle state.
"""
from __future__ import annotations

import re

from pm.records.memory_records import Feature
from pm.records.base import Provenance
from pm.records.enums import FeatureSubtype, FeatureType, InterlocusType, IntentionType, SourceKind
from pm.records.cognitive_records import BeliefRecord, BeliefShiftRecord, Intention
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # ``public_context`` imports this module, and the schemas import
    # ``public_context``. The types are for annotations only.
    from workflows.main_cognition.schemas import AppraisalAffect, AppraisalResult, GoalChange


def _clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)


def normalize_goal_key(text: str) -> str:
    """Return a stable identity for goal text.

    The key ignores case and repeated whitespace, so light rewording of the
    same goal resolves to the same active intention.
    """
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def goal_identity(change: GoalChange) -> str:
    """Return the stable key for a goal change, preferring an explicit key."""
    explicit = change.goal_key.strip().lower()
    return explicit or normalize_goal_key(change.goal)


def intention_key(intention: Intention) -> str:
    """Return the identity of an existing intention record.

    Intentions that the goals commit created carry ``goal_key`` in their
    metadata. Older intentions resolve by their normalized text.
    """
    explicit = str(intention.metadata["goal_key"]).strip().lower() if "goal_key" in intention.metadata else ""
    return explicit or normalize_goal_key(intention.content)


def active_intentions(record_source) -> list[Intention]:
    """Return active Intention records from a MemoryStore view or a MemoryStore.

    ``record_source`` may be a batch view or a MemoryStore. Only intentions with an
    ``active`` flag and an ``active`` status steer current work.
    """
    result = []
    for knoxel in record_source.all_knoxels.values():
        if not isinstance(knoxel, Intention):
            continue
        if not knoxel.active or knoxel.status != "active":
            continue
        result.append(knoxel)
    result.sort(key=lambda item: (-float(item.incentive_salience), item.id))
    return result


def active_goal_texts(record_source) -> list[str]:
    """Return the content of each active goal, highest salience first."""
    return [str(item.content) for item in active_intentions(record_source)]


def find_intention(record_source, key: str):
    """Return the active intention with this identity, or None."""
    for intention in active_intentions(record_source):
        if intention_key(intention) == key:
            return intention
    return None


def apply_appraisal(
    batch,
    appraisal: AppraisalResult,
    affect: AppraisalAffect,
    *,
    evidence_ids,
    companion_name: str,
    memory_trace: dict | None = None,
) -> Feature:
    """Stage the appraisal ``Feature`` of one cycle and return it.

    ``affect`` comes from the appraise phase. The Feature stores the
    appraisal state in ``mental_state_appraisal_vec`` and the delta in
    ``mental_state_delta_vec``. The Feature is causal, so the next cycle reads
    its delta in the EMA history. The relationship partner sits in the entity
    id slot of the appraisal vector. The metadata keeps the feeling in words
    and the regulation mode. The caller commits the batch.
    """
    feeling = appraisal.feeling
    reasoning = appraisal.reasoning
    content = feeling or "appraisal"
    if reasoning:
        content = f"{content}: {reasoning}"
    return batch.create(Feature(
        content=content,
        feature_type=FeatureType.StateUpdate,
        feature_subtype=FeatureSubtype.Appraisal,
        causal=True,
        override_source=companion_name,
        interlocus=InterlocusType.PrivateReportable,
        provenance=Provenance(source_kind=SourceKind.Self, source_ref="main_appraisal"),
        mental_state_appraisal_vec=affect.appraisal_state.to_list(),
        mental_state_delta_vec=list(affect.delta_vec),
        metadata={
            "memory_kind": "internal_state",
            "memory_trace": dict(memory_trace or {}),
            "origin": "appraisal",
            "evidence_ids": [int(item) for item in (evidence_ids or [])],
            "appraisal": dict(appraisal.appraisal),
            "emotion_delta": dict(appraisal.emotion_delta),
            "need_delta": dict(appraisal.need_delta),
            "expectation_error": appraisal.expectation_error,
            "relationship_entity_id": affect.entity_id,
            "affect_summary": affect.felt,
            "regulation": affect.regulation,
        },
    ))


def _stage_intention(batch, *, goal, key, status, active, change, evidence_ids, appraisal_id=None,
                     origin="main_appraisal"):
    """Stage one intention with its identity and lifecycle metadata."""
    metadata = {
        "goal_key": key,
        "origin": origin,
        "reason": change.reason,
        "evidence_ids": [int(item) for item in (evidence_ids or [])],
        "appraisal_id": appraisal_id,
        "choice_history": [{"change": str(change.change), "reason": str(change.reason),
                            "evidence_ids": list(evidence_ids or []), "appraisal_id": appraisal_id}],
    }
    if status != "active":
        metadata["proposal"] = status
    return batch.create(Intention(
        content=goal,
        entity_id=0,
        affective_valence=0.0,
        incentive_salience=_clamp01(float(change.priority or 0.5)),
        intention_type=IntentionType.WorldState,
        status=status,
        active=active,
        metadata=metadata,
    ))


def apply_goal_changes(batch, view, changes: list[GoalChange], *, evidence_ids=None, appraisal_id: int | None = None,
                       origin="main_appraisal") -> list[dict]:
    """Stage goal lifecycle changes and return one record per change.

    Each record has ``change``, ``goal``, ``key``, and ``result``. Results are
    ``added``, ``completed``, ``reprioritized``, ``deferred``, ``refused``,
    ``retained``, or ``missing``. A missing target never invents a completion,
    but a refusal or deferral of a new request is recorded as durable evidence.
    Changed intentions retain the accepted appraisal ID and current reason.
    Existing intention history remains available after refusal or deferral.
    Each record also carries ``intention_id`` and ``created`` so the caller can
    map staged ids to committed ids after one commit.
    """
    records: list[dict] = []
    for change in changes:
        kind = change.change.strip().lower()
        goal = change.goal.strip()
        key = goal_identity(change)
        # Refresh the view so an intention added earlier in this batch is visible.
        existing = find_intention(batch.view(), key) if key else None
        result = "missing"
        created_id: int | None = None
        if kind == "add":
            if existing is None and goal:
                created = _stage_intention(
                    batch, goal=goal, key=key, status="active",
                    active=True, change=change, evidence_ids=evidence_ids, appraisal_id=appraisal_id,
                    origin=origin,
                )
                created_id = int(created.id)
                result = "added"
            elif existing is not None:
                result = "retained"
        elif kind == "complete":
            if existing is not None:
                replaced = existing
                replaced.active = False
                replaced.status = "completed"
                replaced.fulfilment = 1.0
                batch.replace(replaced)
                result = "completed"
        elif kind == "reprioritize":
            if existing is not None:
                replaced = existing
                replaced.incentive_salience = _clamp01(float(change.priority))
                batch.replace(replaced)
                result = "reprioritized"
        elif kind == "defer":
            if existing is not None:
                replaced = existing
                replaced.active = False
                replaced.status = "deferred"
                batch.replace(replaced)
                result = "deferred"
            elif goal:
                created = _stage_intention(
                    batch, goal=goal, key=key, status="deferred",
                    active=False, change=change, evidence_ids=evidence_ids, appraisal_id=appraisal_id,
                    origin=origin,
                )
                created_id = int(created.id)
                result = "deferred"
        elif kind == "refuse":
            if existing is not None:
                replaced = existing
                replaced.active = False
                replaced.status = "refused"
                batch.replace(replaced)
                result = "refused"
            elif goal:
                created = _stage_intention(
                    batch, goal=goal, key=key, status="refused",
                    active=False, change=change, evidence_ids=evidence_ids, appraisal_id=appraisal_id,
                    origin=origin,
                )
                created_id = int(created.id)
                result = "refused"
        elif kind == "retain":
            result = "retained" if existing is not None else "missing"
        intention_id = created_id if created_id is not None else (int(existing.id) if existing is not None else None)
        if created_id is None and intention_id is not None and kind != "retain" and result != "missing":
            intention = batch.view().get_knoxel_by_id(intention_id)
            # Only intentions with earlier changes carry a choice history.
            history = list(intention.metadata["choice_history"]) if "choice_history" in intention.metadata else []
            history.append({"change": kind, "reason": change.reason,
                            "evidence_ids": list(evidence_ids or []), "appraisal_id": appraisal_id})
            intention.metadata.update(reason=history[-1]["reason"], evidence_ids=list(evidence_ids or []),
                                      appraisal_id=appraisal_id, choice_history=history)
            batch.replace(intention)
        records.append({
            "change": kind, "goal": goal, "key": key, "result": result,
            "intention_id": intention_id,
            "created": created_id is not None,
        })
    return records


def subjective_belief(record) -> bool:
    """Identify an active interpretation that reflection can revise.

    Attributed user reports keep their original text, even when their author
    expressed uncertainty. Their belief kind does not make them self-authored.
    """
    return (isinstance(record, BeliefRecord) and record.active
            and record.belief_kind in {"guess", "interpretation", "theory", "self_model", "user_model", "world_model"}
            and record.source != "user_message" and not record.source.startswith("user:")
            and not (record.provenance and record.provenance.source_kind == SourceKind.User))


def revise_reflected_belief(batch, previous: BeliefRecord, *, text: str,
                           evidence_ids: list[int], opposing_ids: list[int],
                           experiment_id: int, companion_name: str) -> BeliefRecord:
    """Supersede a subjective belief without changing its observed sources.

    The caller guards the inspected revision and owns commit and cancellation.
    Reversal calls this function again with the original text and evidence.
    An inactive belief cannot replace a newer accepted interpretation.
    """
    if not subjective_belief(previous):
        raise ValueError("Reflection can revise only an active subjective belief.")
    revised = batch.create(BeliefRecord(
        content=text, belief_key=previous.belief_key, belief_kind="interpretation",
        confidence=previous.confidence, certainty=previous.certainty,
        importance=previous.importance, valence=previous.valence,
        about_entity_ids=list(previous.about_entity_ids), scope=previous.scope,
        supporting_knoxel_ids=list(evidence_ids), contradicting_knoxel_ids=list(opposing_ids),
        source=f"self:{companion_name}",
        provenance=Provenance(source_kind=SourceKind.Self, source_ref="reflection"),
        metadata={**previous.metadata, "reality": "inferred", "revised_from": previous.id,
                  "reflection_experiment_id": experiment_id}))
    old = previous.model_copy(update={"active": False, "superseded_by_id": revised.id})
    batch.replace(old)
    batch.create(BeliefShiftRecord(content=f"Reflection revised {previous.belief_key}",
        target_belief_id=previous.id, proposed_belief_key=previous.belief_key,
        shift_type="supersede", reason="A versioned private experiment revised this interpretation.",
        causal_knoxel_ids=[*evidence_ids, revised.id], accepted=True))
    return revised


__all__ = [
    "subjective_belief",
    "revise_reflected_belief",
    "active_goal_texts",
    "active_intentions",
    "apply_appraisal",
    "apply_goal_changes",
    "find_intention",
    "goal_identity",
    "intention_key",
    "normalize_goal_key",
]
