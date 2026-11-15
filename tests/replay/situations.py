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

from pm.memory.dependencies import capture_source
from pm.memory.store import MemoryStore
from pm.records.base import Provenance, RecordBase, RecordScope
from pm.records.cognitive_records import Intention
from pm.records.copying import pickle_copy
from pm.records.cycle_records import StimulusGroup
from pm.records.cycle_state import CycleState, CycleStateModel
from pm.records.enums import FeatureType, InterlocusType, IntentionType, SourceKind
from pm.records.global_workspace import GwtTurnRecord
from pm.records.memory_records import Feature
from pm.records.mental_state import FullMentalState
from workflows.entity_context.owner import EntityContextOwner
from workflows.entity_context.schemas import (
    EntityClaimEvidence, EntityContextClaim, EntityContextKey, EntityContextProposal,
)


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


@dataclass(frozen=True, slots=True)
class PerspectiveMorningSituation:
    """The committed store and identities a morning-routine ToM scenario refers to."""

    store: MemoryStore
    companion_id: int
    user_id: int
    line_ids: tuple[int, ...]
    #: The line behind the accepted entity claim. The action does not name it.
    known_line_id: int
    #: The lines the theory-of-mind action names as its evidence.
    action_evidence_ids: tuple[int, ...]
    #: Receipt of the accepted entity claim about the user.
    entity_receipt_id: int


#: Synthetic Jesse dialogue lines on different world days, the SC-09 morning
#: routine. They are engineering input, not persona history: a workday waking
#: pattern, one weekend exception, one contact preference, and one unrelated
#: line. A perspective inquiry may read them as its subject's original messages.
PERSPECTIVE_LINES = (
    ("I am usually up around six on workdays, so a morning message finds me before I get started.", 3),
    ("Up at half past six today as usual, workday habit.", 2),
    ("Weekends are different: I sleep in and check nothing before nine.", 1),
    ("Nothing before coffee. Ever. Coffee first.", 1),
    ("The library book came in, by the way.", 0),
)

#: The oldest line. The companion already holds it as one accepted entity claim
#: about Jesse, and the theory-of-mind action does not name it as evidence. A
#: perspective inquiry therefore reaches it only through the accepted model.
PERSPECTIVE_KNOWN_LINE = PERSPECTIVE_LINES[0][0]

#: Proposition key of that accepted claim.
PERSPECTIVE_KNOWN_CLAIM_KEY = "routine.workday_wake"


def perspective_morning(now_utc: datetime, checkpoint: Path) -> PerspectiveMorningSituation:
    """Commit five synthetic lines from Jesse and one accepted claim about Jesse.

    The lines are labelled synthetic input in their provenance and are not
    persona history; they stand in for the messages a perspective inquiry
    reads when it works out the subject's morning routine. The oldest lines
    moved on world days before ``now_utc``; the library note is the newest.

    The oldest line is also held as one accepted entity claim, a report in the
    ``pattern`` layer, staged through the real ``EntityContextOwner``. It
    stands for what ordinary message handling accepted when that line arrived.
    The action names only the four newer lines, so the oldest line reaches the
    inquiry through the accepted model and not through the action.
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

    line_ids: list[int] = []
    for index, (text, days_ago) in enumerate(PERSPECTIVE_LINES, start=1):
        moment = now_utc - timedelta(days=days_ago, hours=6)
        line = store.begin_batch(f"replay-morning-line-{index}", new_tick=True, embed=False)
        record = line.create(Feature(
            feature_type=FeatureType.Dialogue, causal=True, entity_id=user_id,
            addressee_entity_id=companion_id, interlocus=InterlocusType.Public, content=text,
            provenance=Provenance(source_kind=SourceKind.User, source_ref="replay-synthetic-situation"),
            timestamp_world_begin=moment, timestamp_world_end=moment))
        line.preserve_on_recovery()
        store.commit(line)
        line_ids.append(record.id)
    known_line_id = line_ids[0]
    action_evidence_ids = tuple(line_ids[1:])

    owner = EntityContextOwner(store)
    key = EntityContextKey(subject_entity_id=user_id, observer_entity_id=companion_id, scope=RecordScope())
    with store.lock:
        view = store.snapshot(capture=False)
        known_line = view.get(known_line_id)
        evidence = EntityClaimEvidence(source=capture_source(view, known_line_id, current_head=True),
                                       exact_quote=PERSPECTIVE_KNOWN_LINE)
        claim = EntityContextClaim(layer="pattern", claim_key=PERSPECTIVE_KNOWN_CLAIM_KEY,
            statement="Jesse reports being usually up around six on workdays.", epistemic_kind="report",
            confidence=0.9, evidence=[evidence], observed_at_utc=known_line.timestamp_world_begin,
            applicable_from_utc=known_line.timestamp_world_begin)
        proposal = EntityContextProposal(proposal_id="replay-synthetic-entity-claim", key=key,
            parent_receipt_id=owner.current_head(view, key), producer="immediate",
            input_commit_ids=sorted(view.commit_ids), claims=[claim])
        changes = store.begin_batch("replay-morning-entity-claim", embed=False)
        staged = owner.stage_update(changes, proposal)
        if staged.disposition != "staged" or staged.staged_receipt_id is None:
            raise RuntimeError(f"The entity owner refused the synthetic claim: {staged.reason}.")
        changes.preserve_on_recovery()
        store.commit(changes)
    return PerspectiveMorningSituation(store=store, companion_id=companion_id, user_id=user_id,
        line_ids=tuple(line_ids), known_line_id=known_line_id, action_evidence_ids=action_evidence_ids,
        entity_receipt_id=staged.staged_receipt_id)


__all__ = ["CONCERN", "CONCERN_SOURCE", "JESSE_AMBIGUOUS_REPLY", "PERSPECTIVE_KNOWN_CLAIM_KEY",
           "PERSPECTIVE_KNOWN_LINE", "PERSPECTIVE_LINES",
           "SPAMUEL_LONG_MESSAGE", "STORY_NOTES", "PerspectiveMorningSituation", "PrivateConcernSituation",
           "PrivateDwellingSituation", "perspective_morning", "private_concern", "private_dwelling"]
