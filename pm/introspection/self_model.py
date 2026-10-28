"""Revisable self-model beliefs in the accepted memory store.

A capability belief records what the companion currently takes itself to be
able to do, with the evidence state and revision it read. A narrative belief
records an autobiographical interpretation that connects attributed episodes.
Both reuse ``BeliefRecord`` and ``BeliefShiftRecord``. An interpretation never
rewrites the observations it cites, and a source or configuration change marks
the dependent belief stale without erasing its history.

The module writes no memory by itself. The caller commits the batch.

Reading map: :func:`capability_belief_key` and :func:`narrative_belief_key`
name the stable lookup keys, :func:`record_capability_beliefs` and
:func:`record_narrative_belief` stage one belief each, and
:func:`mark_stale_beliefs` records a staleness change without erasing history.
"""
from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from typing import TypedDict

from pm.memory.snapshot import Snapshot
from pm.memory.write_batch import WriteBatch
from pm.records.base import Provenance
from pm.records.cognitive_records import BeliefRecord, BeliefShiftRecord
from pm.records.enums import SourceKind
from pm.introspection.capabilities import CapabilityOverlay

__all__ = [
    "CapabilityBeliefReceipt",
    "NarrativeBeliefReceipt",
    "capability_belief_key",
    "mark_stale_beliefs",
    "narrative_belief_key",
    "record_capability_beliefs",
    "record_narrative_belief",
]


class CapabilityBeliefReceipt(TypedDict):
    """What one capability-belief batch created, retained, and superseded.

    ``accepted_ids`` are the new belief records, ``retained_ids`` are the
    unchanged beliefs of the same evidence revision, and ``revised`` are the
    prior belief records that this batch superseded.
    """

    accepted_ids: list[int]
    retained_ids: list[int]
    revised: list[int]


class NarrativeBeliefReceipt(TypedDict):
    """What one narrative-belief batch created, retained, and superseded.

    ``dropped_ids`` names the cited record ids that the batch's view did not
    hold, so an interpretation never claims evidence it could not read.
    """

    accepted_ids: list[int]
    retained_ids: list[int]
    revised: list[int]
    dropped_ids: list[int]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def capability_belief_key(name: str) -> str:
    """Return the stable lookup key for one capability belief."""
    return f"self_model:capability:{name.strip().casefold()}"


def narrative_belief_key(topic: str) -> str:
    """Return the stable lookup key for one autobiographical interpretation."""
    return f"self_model:narrative:{topic.strip().casefold()}"


def _active(view: Snapshot, key: str) -> list[BeliefRecord]:
    """Return the active beliefs of one key in record order."""
    found = [record for record in view.all_knoxels.values()
             if isinstance(record, BeliefRecord) and record.active and record.belief_key == key]
    return sorted(found, key=lambda record: record.id)


def _supersede(batch: WriteBatch, prior: list[BeliefRecord], belief: BeliefRecord, *, reason: str,
               source_kind: SourceKind, source_ref: str) -> list[int]:
    """Deactivate the prior beliefs and record one shift for each."""
    revised: list[int] = []
    for old in prior:
        old = old.copy_for_edit()
        old.active = False
        old.superseded_by_id = belief.id
        batch.replace(old)
        batch.create(BeliefShiftRecord(
            content=f"Superseded {belief.belief_key}",
            target_belief_id=old.id, proposed_belief_key=belief.belief_key,
            shift_type="supersede", reason=reason, causal_knoxel_ids=[belief.id],
            accepted=True, provenance=Provenance(source_kind=source_kind, source_ref=source_ref),
        ))
        revised.append(old.id)
    return revised


def record_capability_beliefs(batch: WriteBatch, overlay: CapabilityOverlay, *, companion_name: str,
                              source_revision: str = "",
                              stale_names: Iterable[str] = ()) -> CapabilityBeliefReceipt:
    """Stage one capability belief per inspected capability.

    A repeated identical finding retains the existing record and adds no
    confidence. A changed finding supersedes the old record and keeps its
    history. A finding whose evidence revision changed is marked stale
    instead of being silently refreshed.
    """
    view = batch.view()
    accepted: list[int] = []
    retained: list[int] = []
    revised: list[int] = []
    stale_set = {name.strip().casefold() for name in stale_names}
    for name, finding in overlay.findings.items():
        key = capability_belief_key(name)
        content = f"{finding.summary} Current evidence state: {finding.strongest.value}."
        metadata = {
            "memory_kind": "capability_belief",
            "capability": name,
            "state": finding.strongest.value,
            "states": sorted(state for state, rows in finding.evidence.items() if rows),
            "evidence": finding.evidence,
            "evidence_revision": finding.revision,
            "source_revision": source_revision,
            "observed_at": finding.observed_at.isoformat(),
            "stale": name.casefold() in stale_set,
        }
        prior = _active(view, key)
        same = next((record for record in prior
                     if record.content == content
                     and record.metadata.get("evidence_revision") == finding.revision), None)
        if same is not None:
            # The identical finding on the same evidence revision does not stack.
            retained.append(same.id)
            continue
        belief = batch.create(BeliefRecord(
            content=content, belief_key=key, belief_kind="capability",
            confidence=0.6 if finding.strongest == "observed" else 0.45,
            source=f"self:{companion_name}",
            provenance=Provenance(source_kind=SourceKind.Self, source_ref="source_inspection"),
            metadata=metadata,
        ))
        accepted.append(belief.id)
        revised.extend(_supersede(batch, prior, belief,
                                  reason="A later inspected evidence state replaced the earlier belief. "
                                         "The earlier belief and its evidence stay linked.",
                                  source_kind=SourceKind.Self, source_ref="source_inspection"))
    return CapabilityBeliefReceipt(accepted_ids=accepted, retained_ids=retained, revised=revised)


def mark_stale_beliefs(batch: WriteBatch, names: Iterable[str], *, revision: str, reason: str) -> list[int]:
    """Mark dependent capability beliefs stale after a source change.

    The belief stays active and keeps its original evidence. The metadata
    records when and why it became stale, and one shift record preserves the
    transition. A later refresh replaces it through the normal supersede path.
    """
    view = batch.view()
    marked: list[int] = []
    for name in names:
        for record in _active(view, capability_belief_key(name)):
            record = record.copy_for_edit()
            record.metadata = {
                **record.metadata, "stale": True, "stale_revision": revision,
                "stale_since": _utcnow().isoformat(), "stale_reason": reason,
            }
            batch.replace(record)
            batch.create(BeliefShiftRecord(
                content=f"Marked {record.belief_key} stale",
                target_belief_id=record.id, proposed_belief_key=record.belief_key,
                shift_type="mark_stale", reason=reason, causal_knoxel_ids=[record.id],
                accepted=True, provenance=Provenance(source_kind=SourceKind.System,
                                                     source_ref="source_inspection"),
            ))
            marked.append(record.id)
    return marked


def record_narrative_belief(batch: WriteBatch, *, topic: str, text: str, evidence_ids: Iterable[int],
                            episode_ids: Iterable[int] = (), confidence: float = 0.4,
                            companion_name: str = "") -> NarrativeBeliefReceipt:
    """Stage one revisable autobiographical interpretation.

    The interpretation cites the episodes and records it read. It is an
    attributed reading, so ``belief_kind`` is ``interpretation`` and the
    provenance is the self. A later interpretation on the same topic
    supersedes this one while the prior version stays in history.
    """
    view = batch.view()
    in_view: list[int] = []
    for record_id in dict.fromkeys(int(item) for item in evidence_ids):
        if view.get_knoxel_by_id(record_id) is not None:
            in_view.append(record_id)
    dropped = [int(item) for item in evidence_ids if int(item) not in in_view]
    key = narrative_belief_key(topic)
    normalized = " ".join(str(text).split())
    if not normalized:
        raise ValueError("A narrative interpretation needs text.")
    prior = _active(view, key)
    if any(record.content == normalized for record in prior):
        existing = next(record for record in prior if record.content == normalized)
        return NarrativeBeliefReceipt(accepted_ids=[], retained_ids=[existing.id], revised=[],
                                      dropped_ids=dropped)
    belief = batch.create(BeliefRecord(
        content=normalized, belief_key=key, belief_kind="interpretation",
        confidence=confidence, supporting_knoxel_ids=in_view,
        source=f"self:{companion_name}",
        provenance=Provenance(source_kind=SourceKind.Self, source_ref="private_reflection"),
        metadata={"memory_kind": "autobiographical_interpretation", "topic": topic,
                  "episode_ids": [int(item) for item in episode_ids],
                  "evidence_ids": in_view, "dropped_evidence_ids": dropped,
                  "reality": "inferred"},
    ))
    revised = _supersede(batch, prior, belief,
                         reason="A later interpretation on the same topic replaced it. "
                                "The earlier episode references stay linked.",
                         source_kind=SourceKind.Self, source_ref="private_reflection")
    return NarrativeBeliefReceipt(accepted_ids=[belief.id], retained_ids=[], revised=revised,
                                  dropped_ids=dropped)
