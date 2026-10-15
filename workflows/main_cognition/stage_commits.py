"""Turn accepted stage proposals into committed MemoryStore state.

Main cognition and the future inner loop share these helpers. They keep the
accepted state path in one place:

- an appraisal updates the cycle's latent mental state and writes one appraisal
  Feature that carries the new state, the delta, and its evidence;
- a goal change creates, completes, or reprioritises an ``Intention`` knoxel
  that has a stable identity and a lifecycle status.

The functions operate on one open ``WriteBatch``. They never commit, send
bus messages, or call a model. The caller commits after it adds the attention
frame, GWT turn, and cycle receipt.
"""
from __future__ import annotations

import re

from workflows.main_cognition.affect_dynamics import AffectProjection, project_affect
from pm.records.memory_records import Feature
from pm.records.base import Provenance
from pm.records.enums import FeatureSubtype, FeatureType, InterlocusType, IntentionType, SourceKind
from pm.records.cognitive_records import BeliefRecord, BeliefShiftRecord, Intention
from pm.records.mental_state import FullMentalState


def _normalize_topic(text: str, limit: int = 48) -> str:
    """Return a stable topic key from free text.

    Case and repeated whitespace are ignored so nearby reformulations of one
    belief resolve to one identity. The limit keeps the key readable in
    metadata when a codelet supplied no explicit topic.
    """
    return re.sub(r"\s+", " ", str(text or "").strip().casefold())[:limit]


def _clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)


def normalize_goal_key(text: str) -> str:
    """Return a stable identity for goal text.

    The key ignores case and repeated whitespace, so light rewording of the
    same goal resolves to the same active intention.
    """
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def goal_identity(change) -> str:
    """Return the stable key for a goal change, preferring an explicit key."""
    explicit = str(getattr(change, "goal_key", "") or "").strip().lower()
    return explicit or normalize_goal_key(getattr(change, "goal", ""))


def intention_key(intention) -> str:
    """Return the identity of an existing intention knoxel."""
    metadata = getattr(intention, "metadata", None) or {}
    explicit = str(metadata.get("goal_key", "") or "").strip().lower()
    return explicit or normalize_goal_key(getattr(intention, "content", ""))


def active_intentions(record_source) -> list:
    """Return active Intention knoxels from a MemoryStore view or a MemoryStore.

    ``record_source`` may be a batch view or a MemoryStore. Only intentions with an
    ``active`` flag and an ``active`` status steer current work.
    """
    if hasattr(record_source, "all_knoxels"):
        records = record_source.all_knoxels
        if isinstance(records, dict):
            candidates = records.values()
        else:  # pragma: no cover - defensive for unusual owners
            candidates = records
    else:  # pragma: no cover - defensive for unusual owners
        candidates = []
    result = []
    for knoxel in candidates:
        if not isinstance(knoxel, Intention):
            continue
        if not getattr(knoxel, "active", False):
            continue
        if str(getattr(knoxel, "status", "active")) != "active":
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
    view,
    appraisal,
    *,
    previous_vec,
    evidence_ids,
    companion_name: str,
    elapsed_seconds: float = 0.0,
    personality=None,
    expectation_error: float = 0.0,
    projection: AffectProjection | None = None,
    memory_trace: dict | None = None,
) -> tuple[Feature, FullMentalState]:
    """Apply one appraisal to the mental state and stage an appraisal Feature.

    ``previous_vec`` is the latent vector of the last accepted cycle. The
    projection applies elapsed-time decay, personality gains, canonical
    appraisal axes, explicit emotion and need deltas, and an expectation error.
    The Feature records the new total state in ``mental_state_appraisal`` and
    the difference in ``mental_state_delta``. The metadata carries the causal
    trace and the behavior mapping. The caller reads ``state`` to store it on
    the cycle.

    ``projection`` may be supplied when the parent pipeline already computed
    the same step, so the accepted state and the stage input cannot diverge.
    """
    if projection is None:
        projection = project_affect(
            previous_vec, appraisal,
            elapsed_seconds=elapsed_seconds,
            personality=personality,
            expectation_error=expectation_error,
        )
    updated = projection.state
    delta_vec = projection.delta.to_list()
    feeling = str(getattr(appraisal, "feeling", "") or "")
    reasoning = str(getattr(appraisal, "reasoning", "") or "")
    content = feeling or "appraisal"
    if reasoning:
        content = f"{content}: {reasoning}"
    feature = batch.create(Feature(
        content=content,
        feature_type=FeatureType.StateUpdate,
        feature_subtype=FeatureSubtype.Appraisal,
        causal=True,
        override_source=companion_name,
        interlocus=InterlocusType.PrivateReportable,
        provenance=Provenance(source_kind=SourceKind.Self, source_ref="main_appraisal"),
        mental_state_appraisal_vec=updated.to_list(),
        mental_state_delta_vec=delta_vec,
        metadata={
            "memory_kind": "internal_state",
            "memory_trace": dict(memory_trace or {}),
            "origin": "appraisal",
            "evidence_ids": [int(item) for item in (evidence_ids or [])],
            "emotion_delta": dict(getattr(appraisal, "emotion_delta", None) or {}),
            "need_delta": dict(getattr(appraisal, "need_delta", None) or {}),
            "affect_trace": projection.trace,
            "affect_summary": projection.summary,
            "regulation": projection.regulation,
            "behavior": projection.behavior,
        },
    ))
    return feature, updated


def _stage_intention(batch, *, goal, key, status, active, change, evidence_ids, appraisal_id=None,
                     origin="main_appraisal"):
    """Stage one intention with its identity and lifecycle metadata."""
    metadata = {
        "goal_key": key,
        "origin": origin,
        "reason": str(getattr(change, "reason", "") or ""),
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
        incentive_salience=_clamp01(float(getattr(change, "priority", 0.5) or 0.5)),
        intention_type=IntentionType.WorldState,
        status=status,
        active=active,
        metadata=metadata,
    ))


def apply_goal_changes(batch, view, changes, *, evidence_ids=None, appraisal_id: int | None = None,
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
    for change in changes or []:
        kind = str(getattr(change, "change", "") or "").strip().lower()
        goal = str(getattr(change, "goal", "") or "").strip()
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
                replaced.incentive_salience = _clamp01(float(getattr(change, "priority", 0.5)))
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
            history = list(intention.metadata.get("choice_history", []))
            history.append({"change": kind, "reason": str(getattr(change, "reason", "") or ""),
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


BELIEF_REALITIES = frozenset({"inferred", "internal"})


def apply_belief_proposals(
    batch,
    nominations,
    *,
    companion_name: str,
    stale_ids: set[str] | None = None,
    catalogue_version: str = "",
) -> dict:
    """Stage revisable subjective beliefs for accepted belief proposals.

    The caller passes the cycle's parsed nominations; only selected, non-stale
    ``belief``-kind nominations with text are consumed. The proposal is an
    interpretation: it cites the observed records through
    ``supporting_knoxel_ids`` and never rewrites them. A belief proposal whose
    reality is not ``inferred`` or ``internal`` is rejected, so a subjective
    reading cannot enter storage as an observed fact.

    Duplicate-effect prevention: the belief key is the codelet plus its topic
    (or the normalized text). An identical repeat retains the existing record
    and adds no second belief and no confidence. A changed text under the
    same key supersedes the old record and keeps its history through
    ``superseded_by_id`` and a BeliefShiftRecord. Repeated generated
    interpretations of one source therefore do not count as independent
    evidence.

    Returns a receipt with ``accepted_ids``, ``retained_ids``, ``revised``, and
    ``rejected``. Proposals take effect only when the caller commits the batch.
    """
    stale = set(stale_ids or set())
    view = batch.view()
    accepted, retained, revised, rejected = [], [], [], []
    for nomination in nominations or []:
        if str(getattr(nomination, "attention_kind", "")) != "belief":
            continue
        if str(getattr(nomination, "status", "")) != "selected":
            continue
        if str(getattr(nomination, "nomination_id", "")) in stale:
            continue
        text = " ".join(str(getattr(nomination, "text", "") or "").split())
        if not text:
            continue
        reality = str(getattr(nomination, "reality", "") or "")
        if reality not in BELIEF_REALITIES:
            rejected.append({"nomination_id": getattr(nomination, "nomination_id", ""),
                             "reason": f"belief proposal reality {reality!r} is not a permitted interpretation"})
            continue
        codelet = str(getattr(nomination, "codelet", "") or "")
        topic = _normalize_topic(getattr(nomination, "topic", "") or text)
        key = f"codelet:{codelet}:{topic}"
        targets = [int(item) for item in getattr(nomination, "target_ids", []) or []]
        in_view = [item for item in dict.fromkeys(targets) if view.get_knoxel_by_id(item) is not None]
        dropped = [item for item in targets if item not in in_view]
        prior = [record for record in view.all_knoxels.values()
                 if isinstance(record, BeliefRecord) and record.active
                 and record.belief_key == key]
        prior = sorted(prior, key=lambda record: record.id)
        if any(record.content == text for record in prior):
            # Retained: the same accepted interpretation does not stack.
            existing = next(record for record in prior if record.content == text)
            known = set(existing.supporting_knoxel_ids)
            additions = [item for item in in_view if item not in known]
            if additions:
                existing.supporting_knoxel_ids = [*existing.supporting_knoxel_ids, *additions]
                batch.replace(existing)
            retained.append(existing.id)
            continue
        scope = getattr(view.get_knoxel_by_id(in_view[0]), "scope", None) if in_view else None
        metadata = {
            "memory_kind": "subjective_belief",
            "codelet": codelet,
            "nomination_id": str(getattr(nomination, "nomination_id", "")),
            "reality": reality,
            "catalogue_version": catalogue_version,
            "topic": topic,
            "evidence_ids": list(in_view),
            "dropped_target_ids": dropped,
            "revised_from": prior[0].id if prior else None,
        }
        belief = batch.create(BeliefRecord(
            content=text, belief_key=key, belief_kind="interpretation",
            confidence=0.45 if in_view else 0.3,
            supporting_knoxel_ids=in_view,
            source=f"self:{companion_name}",
            **({"scope": scope} if scope is not None else {}),
            provenance=Provenance(source_kind=SourceKind.Self, source_ref=f"codelet:{codelet}"),
            metadata=metadata,
        ))
        accepted.append(belief.id)
        for old in prior:
            old.active = False
            old.superseded_by_id = belief.id
            batch.replace(old)
            revised.append({"old_id": old.id, "new_id": belief.id})
            batch.create(BeliefShiftRecord(
                content=f"Codelet {codelet} revised {key}",
                target_belief_id=old.id, proposed_belief_key=key,
                shift_type="supersede",
                reason="A later accepted interpretation with the same topic replaced it. "
                       "The earlier belief and its evidence stay linked.",
                causal_knoxel_ids=[*in_view, belief.id], accepted=True,
                provenance=Provenance(source_kind=SourceKind.Self, source_ref=f"codelet:{codelet}"),
            ))
    return {"accepted_ids": accepted, "retained_ids": retained, "revised": revised,
            "rejected": rejected}


__all__ = [
    "active_goal_texts",
    "active_intentions",
    "apply_appraisal",
    "apply_belief_proposals",
    "apply_goal_changes",
    "find_intention",
    "goal_identity",
    "intention_key",
    "normalize_goal_key",
]
