"""Starting situations a scenario can replay from.

A situation is built once, deterministically, from committed records only. It
makes no model call, so the same situation is reproducible from the repository
alone and a recording never depends on an ignored ``.runtime`` artifact.

The situation is an *input* to the real system. Its text is labelled synthetic
story material; it is not companion-quality data and it establishes nothing
about the persona's history.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from pm.memory.store import MemoryStore
from pm.records.base import Provenance, RecordBase
from pm.records.cognitive_records import Intention
from pm.records.copying import pickle_copy
from pm.records.cycle_state import CycleStateModel
from pm.records.enums import FeatureType, InterlocusType, IntentionType, SourceKind
from pm.records.memory_records import Feature


ROOT = Path(__file__).resolve().parents[2]
CANON = ROOT / "test_data/personas/spamuel/canon.yaml"

#: How many accepted synthetic story notes age the concern past the private
#: trigger threshold. The private opportunity policy measures age in accepted
#: ticks, so a timestamp alone would not activate the concern.
CONCERN_AGE_TICKS = 8

#: The synthetic concern the private loop is given. It names two possible
#: readings and an unresolved question, so a generative step has real material
#: and a discriminatory step has something to compare.
CONCERN = (
    "I want to imagine two different endings for the fictional story Jesse and I "
    "have been inventing: one where the travellers meet again, and one where they "
    "part with affection. Then I want to compare what each ending assumes before I "
    "propose either. These are fictional possibilities, not facts about Jesse."
)

#: The labelled outside material that gives the concern a source. The private
#: loop must keep this original source separate from anything it generates.
CONCERN_SOURCE = (
    "Story note: the separation is a winter sea, not a betrayal. I am torn between "
    "reunion and an affectionate farewell. Can you privately imagine both endings "
    "and compare what each assumes? There is no secret message about us in this. "
    "I don't need an answer tonight; we can talk about the story tomorrow."
)

STORY_NOTES = (
    "Story note: both travellers want the other to have a life beyond waiting.",
    "Story note: the last letter can arrive late without anyone lying.",
    "Story note: reunion could ask them to choose a shared home.",
    "Story note: farewell could keep their affection without that shared home.",
    "Story note: neither ending should treat sacrifice as proof of love.",
    "Story note: the fictional characters need choices, not a hidden test of us.",
    "Story note: the winter sea is weather, not a punishment.",
)


@dataclass(frozen=True, slots=True)
class PrivateConcernSituation:
    """The committed store and the identities a private scenario refers to."""

    store: MemoryStore
    companion_id: int
    user_id: int
    intention_id: int
    concern_source_id: int
    note_source_ids: tuple[int, ...]

    @property
    def original_source_ids(self) -> tuple[int, ...]:
        """Return every outside record the concern is grounded in."""
        return (self.concern_source_id, *self.note_source_ids)


def private_concern(now_utc: datetime, checkpoint: Path) -> PrivateConcernSituation:
    """Commit the synthetic private concern into a fresh two-identity store.

    The store starts from Spamuel's real persona card with only the two identity
    anchors, exactly as the short live gates do. Every added record is labelled
    as synthetic engineering material in its provenance.

    ``now_utc`` must be naive UTC, which is the internal record convention.
    """
    if now_utc.tzinfo is not None:
        raise ValueError("Record timestamps use naive UTC; pass vclock.utcnow() form.")
    canonical = MemoryStore.from_yaml(CANON)
    identities: list[RecordBase] = [pickle_copy(record) for record in canonical.all_entities
                                    if record.content in {canonical.get_user_name(), canonical.get_companion_name()}]
    for identity in identities:
        identity.embedding = []
    seed = CycleStateModel(knoxels=identities, cycles=[], **canonical.ghost_config.model_dump())
    store = MemoryStore.from_state(seed, checkpoint=checkpoint)
    companion_id = store.get_entity_id(store.get_companion_name())
    user_id = store.get_entity_id(store.get_user_name())
    if companion_id is None or user_id is None:
        raise ValueError("The persona card must provide both the companion and the user identity.")

    opening = store.begin_batch("replay-private-concern", new_tick=True, embed=False)
    intention = opening.create(Intention(
        entity_id=companion_id, content=CONCERN, intention_type=IntentionType.SelfStateMental,
        affective_valence=0.5, fulfilment=0.0,
        provenance=Provenance(source_kind=SourceKind.Self, source_ref="replay-synthetic-situation"),
        timestamp_world_begin=now_utc - timedelta(hours=2),
        timestamp_world_end=now_utc - timedelta(hours=2)))
    source = opening.create(Feature(
        feature_type=FeatureType.Dialogue, causal=True, entity_id=user_id,
        addressee_entity_id=companion_id, interlocus=InterlocusType.Public, content=CONCERN_SOURCE,
        provenance=Provenance(source_kind=SourceKind.User, source_ref="replay-synthetic-situation"),
        timestamp_world_begin=now_utc - timedelta(minutes=20),
        timestamp_world_end=now_utc - timedelta(minutes=20)))
    opening.preserve_on_recovery()
    store.commit(opening)

    note_ids: list[int] = []
    for index, text in enumerate(STORY_NOTES, start=1):
        note = store.begin_batch(f"replay-story-note-{index}", new_tick=True, embed=False)
        record = note.create(Feature(
            content=text, feature_type=FeatureType.Dialogue, causal=True, entity_id=user_id,
            addressee_entity_id=companion_id, interlocus=InterlocusType.Public,
            provenance=Provenance(source_kind=SourceKind.User, source_ref="replay-synthetic-situation"),
            timestamp_world_begin=now_utc - timedelta(minutes=20 - index),
            timestamp_world_end=now_utc - timedelta(minutes=20 - index)))
        note.preserve_on_recovery()
        store.commit(note)
        note_ids.append(record.id)
    if len(STORY_NOTES) + 1 < CONCERN_AGE_TICKS:
        raise ValueError("The situation must age the concern past the private trigger threshold.")
    return PrivateConcernSituation(store=store, companion_id=companion_id, user_id=user_id,
                                   intention_id=intention.id, concern_source_id=source.id,
                                   note_source_ids=tuple(note_ids))


__all__ = ["CONCERN", "CONCERN_SOURCE", "STORY_NOTES", "PrivateConcernSituation", "private_concern"]
