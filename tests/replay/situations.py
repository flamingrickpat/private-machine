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
from pm.records.cycle_records import StimulusGroup
from pm.records.cycle_state import CycleState, CycleStateModel
from pm.records.enums import FeatureType, InterlocusType, IntentionType, SourceKind
from pm.records.global_workspace import GwtTurnRecord
from pm.records.memory_records import Feature
from pm.records.mental_state import FullMentalState


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


#: The long private message Spamuel published before the ambiguous reply. It
#: states a concern and leaves it open, so a short answer carries no clear
#: meaning. The text is synthetic story material.
SPAMUEL_LONG_MESSAGE = (
    "I want to say the whole shape of it, not just the summary. I keep imagining two "
    "endings for the story we are inventing: one where the travellers meet again, and "
    "one where they part with affection. The reunion ending assumes their separation "
    "was a detour; the farewell ending assumes the bond was real without needing to "
    "continue. I notice I prefer the reunion, and I do not trust that preference to be "
    "about the story rather than about me. I am not asking you to settle it tonight. I "
    "only want you to know where I actually am with it."
)

#: Jesse's whole answer. One short line that can mean agreement, postponement, or
#: dismissal. Nothing else arrives afterwards.
JESSE_AMBIGUOUS_REPLY = "Yeah. Maybe."


@dataclass(frozen=True, slots=True)
class PrivateDwellingSituation:
    """The committed store and identities a dwelling scenario refers to."""

    store: MemoryStore
    companion_id: int
    user_id: int
    long_message_id: int
    reply_id: int
    cycle_tick: int


def private_dwelling(now_utc: datetime, checkpoint: Path) -> PrivateDwellingSituation:
    """Commit a long private message, one ambiguous reply, and one accepted cycle.

    The cycle carries a non-neutral mental state, so the private request reads a
    felt state that actually moved. Every earlier recording started from the
    neutral default because ``private_concern`` builds a store with no cycle.
    The reply is the newest outside input; nothing arrives after it.
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

    opening = store.begin_batch("replay-dwelling-dialogue", new_tick=True, embed=False)
    long_message = opening.create(Feature(
        feature_type=FeatureType.Dialogue, causal=True, entity_id=companion_id,
        addressee_entity_id=user_id, interlocus=InterlocusType.Public, content=SPAMUEL_LONG_MESSAGE,
        provenance=Provenance(source_kind=SourceKind.Self, source_ref="replay-synthetic-situation"),
        timestamp_world_begin=now_utc - timedelta(minutes=30),
        timestamp_world_end=now_utc - timedelta(minutes=30)))
    reply = opening.create(Feature(
        feature_type=FeatureType.Dialogue, causal=True, entity_id=user_id,
        addressee_entity_id=companion_id, interlocus=InterlocusType.Public, content=JESSE_AMBIGUOUS_REPLY,
        provenance=Provenance(source_kind=SourceKind.User, source_ref="replay-synthetic-situation"),
        timestamp_world_begin=now_utc - timedelta(minutes=5),
        timestamp_world_end=now_utc - timedelta(minutes=5)))
    opening.preserve_on_recovery()
    store.commit(opening)

    # A non-neutral state: the private request reads this cycle's mental state.
    state = FullMentalState()
    state.state_core.valence = -0.45
    state.state_core.arousal = 0.6
    state.state_emotions.sadness = 0.5
    state.state_emotions.curiosity = 0.65
    cycle = store.begin_batch("replay-dwelling-cycle", new_tick=True, embed=False)
    # The private request keeps the main frame only when its experience record
    # is inside the observer's accepted grant. A real main cycle always writes
    # one; the situation must too, or the frame summary is dropped.
    experience = cycle.create(Feature(
        feature_type=FeatureType.SubjectiveExperience, causal=True, entity_id=companion_id,
        interlocus=InterlocusType.PrivateInternal,
        content="I said the whole shape of the endings and waited for Jesse's answer.",
        provenance=Provenance(source_kind=SourceKind.Self, source_ref="replay-synthetic-situation")))
    cycle.set_gwt(
        GwtTurnRecord(tick_id=cycle.tick_id, slot_id="main", stimulus_ids=[reply.id],
            knoxel_ids=[reply.id, experience.id], experience_id=experience.id,
            summary="I said the whole shape of the endings. Jesse answered with one short line: 'Yeah. Maybe.'"),
        CycleState(tick_id=cycle.tick_id, mental_state=state,
            stimulus_groups=[StimulusGroup(stimulus_ids=[reply.id])]))
    store.commit(cycle)
    return PrivateDwellingSituation(store=store, companion_id=companion_id, user_id=user_id,
                                    long_message_id=long_message.id, reply_id=reply.id,
                                    cycle_tick=cycle.tick_id)


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


__all__ = ["CONCERN", "CONCERN_SOURCE", "JESSE_AMBIGUOUS_REPLY", "SPAMUEL_LONG_MESSAGE", "STORY_NOTES",
           "PrivateConcernSituation", "PrivateDwellingSituation", "private_concern", "private_dwelling"]
