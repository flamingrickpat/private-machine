"""Evidence-preserving experiences and reported beliefs in a cognition batch.

The module proposes memory through the existing MemoryStore batch. It never commits
separately, infers external facts from companion prose, or invents activity.

Every function stages records into an open
:class:`~pm.memory.write_batch.WriteBatch` and leaves the commit to the host.
A repeated commit returns the earlier receipt, so staging is idempotent per
batch.
"""
from __future__ import annotations

from typing import TypedDict

from pm.memory.snapshot import Snapshot
from pm.memory.write_batch import WriteBatch
from pm.records.memory_records import ActionSchemaRecord, Feature, MemoryClusterRecord
from pm.records.base import Provenance
from pm.records.enums import ClusterType, FeatureType, SourceKind
from pm.records.cognitive_records import BeliefRecord, BeliefShiftRecord
from workflows.main_cognition.schemas import ActionMetadata, BeliefUpdate, RecordMetadataView


class RejectedBeliefUpdate(TypedDict):
    """One proposed belief update that the batch refused to stage, with the reason."""

    source_id: int
    reason: str


class BeliefUpdateReceipt(TypedDict):
    """The outcome of the belief updates of one cycle.

    ``accepted_ids`` are the belief records that carry the accepted reports,
    in proposal order. ``rejected`` keeps every refused proposal with its
    cause, so a refused report is evidence rather than silence.
    """

    accepted_ids: list[int]
    rejected: list[RejectedBeliefUpdate]


class UtteranceMetadata(TypedDict, total=False):
    """The metadata of one stored utterance.

    ``memory_kind`` and ``factual_promotion`` are always present. The other
    keys exist only when a prior action proves a retained deceptive intent.
    """

    memory_kind: str
    factual_promotion: bool
    action_schema_id: int
    deception_intent_id: int
    deception_purpose: str
    known_fact_ids: list[int]


def consolidate_cycle(batch: WriteBatch, source_ids: list[int], appraisal: Feature,
                      affect_summary: str) -> list[int]:
    """Stage one episodic summary for real inputs and accepted affect.

    The cluster retains original evidence links and source timestamps. A
    repeated cognition commit reuses its receipt, so it adds no duplicate.
    """
    view = batch.view()
    groups = {}
    for source_id in source_ids:
        source = view.get(source_id)
        groups.setdefault((source.scope.reality_id, source.scope.frame), []).append(source)
    created = []
    for records in groups.values():
        ids = [record.id for record in records]
        lines = []
        for record in records:
            kind = RecordMetadataView.model_validate(record.metadata).memory_kind or "attributed_record"
            lines.append(f"[{record.id}; {kind}] {record.override_source if isinstance(record, Feature) else 'source'}: {record.content}")
        # Affect is an accepted internal event, even when its cause is imagined.
        lines.append(f"Accepted internal consequence [{appraisal.id}]: {affect_summary}")
        ids.append(appraisal.id)
        cluster = batch.create(MemoryClusterRecord(
            content="\n".join(lines), cluster_type=ClusterType.Topical, level=100,
            included_event_ids=",".join(map(str, ids)),
            min_tick_id=min(record.tick_id for record in records), max_tick_id=appraisal.tick_id,
            min_event_id=min(ids), max_event_id=max(ids), scope=records[0].scope,
            timestamp_world_begin=min(record.timestamp_world_begin for record in records),
            timestamp_world_end=appraisal.timestamp_world_end,
            provenance=Provenance(source_kind=SourceKind.System, source_ref=batch.commit_id),
            metadata={"memory_kind": "experience_summary", "source_ids": ids,
                      "affect_trace_id": appraisal.id},
        ))
        created.append(cluster.id)
    return created


def update_beliefs(batch: WriteBatch, updates: list[BeliefUpdate], *, user_name: str,
                   permitted_source_ids: set[int] | None = None) -> BeliefUpdateReceipt:
    """Stage exact user reports and correction links, or record a rejection.

    Quotes must occur verbatim in live user dialogue. Companion speech, seed
    background, predictions, and imagination cannot establish factual beliefs.
    Conflicts remain active until an explicit correction supersedes them.
    With a source grant, an unlisted ID is rejected before lookup.
    Proposals take effect only when the caller commits the batch.
    """
    accepted: list[int] = []
    rejected: list[RejectedBeliefUpdate] = []
    for update in updates:
        if permitted_source_ids is not None and update.source_id not in permitted_source_ids:
            rejected.append({"source_id": update.source_id, "reason": "source is outside the current stimulus grant"})
            continue
        view = batch.view()
        source = view.get_knoxel_by_id(update.source_id)
        if (not isinstance(source, Feature) or source.feature_type != FeatureType.Dialogue
                or source.override_source != user_name
                or RecordMetadataView.model_validate(source.metadata).memory_kind != "utterance"
                or source.provenance is None or source.provenance.source_kind != SourceKind.User
                or update.quote not in source.content):
            rejected.append({"source_id": update.source_id, "reason": "requires an exact live user utterance"})
            continue
        key = " ".join(update.belief_key.casefold().split())
        prior = [record for record in view.all_knoxels.values()
                 if isinstance(record, BeliefRecord) and record.active
                 and record.belief_key == key and record.scope == source.scope
                 and record.source == f"user:{user_name}"]
        duplicate = next((record for record in prior if record.content == update.quote), None)
        if duplicate:
            accepted.append(duplicate.id)
            continue
        belief = batch.create(BeliefRecord(
            content=update.quote, belief_key=key, belief_kind="reported_by_user",
            supporting_knoxel_ids=[source.id], source=f"user:{user_name}", scope=source.scope,
            provenance=source.provenance,
            metadata={"memory_kind": "reported_belief", "correction": update.correction},
        ))
        accepted.append(belief.id)
        for old in prior:
            old = old.copy_for_edit()
            old.contradicting_knoxel_ids.append(source.id)
            if update.correction:
                old.active = False
                old.superseded_by_id = belief.id
            batch.replace(old)
            batch.create(BeliefShiftRecord(
                content=f"User {'corrected' if update.correction else 'reported a competing claim for'} {key}",
                target_belief_id=old.id, proposed_belief_key=key,
                shift_type="supersede" if update.correction else "conflict",
                reason="Original user statements remain linked. No outside fact was independently verified.",
                causal_knoxel_ids=[source.id, belief.id], accepted=True, scope=source.scope,
                provenance=source.provenance,
            ))
    return BeliefUpdateReceipt(accepted_ids=accepted, rejected=rejected)


def utterance_metadata(view: Snapshot, action_id: int | None) -> UtteranceMetadata:
    """Retain a prior presentation intent without granting its content truth.

    Only an action already present in the read view can justify deception.
    Adding an intent after the utterance cannot alter this stored attribution.
    """
    result: UtteranceMetadata = {"memory_kind": "utterance", "factual_promotion": False}
    action = view.get_knoxel_by_id(action_id) if action_id is not None else None
    if isinstance(action, ActionSchemaRecord):
        result["action_schema_id"] = action.id
        # Actions from the main cycle store their presentation. Other actions
        # have none and are candid.
        presentation = ActionMetadata.model_validate(action.metadata).presentation
        if presentation.mode == "deceive" and presentation.purpose and presentation.known_fact_ids:
            known = list(presentation.known_fact_ids)
            if all(view.get_knoxel_by_id(key) is not None for key in known):
                result.update(deception_intent_id=action.id, deception_purpose=presentation.purpose, known_fact_ids=known)
    return result
