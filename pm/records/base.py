"""
holds - shared knoxel identity, typing, timestamps, and owner storage hooks for memory objects.

archetype: information-holder
owns: RecordKind names including Structure and ModuleOutput, RecordBase identity/timestamp fields, trace tags, type/subtype helpers, and RecordOwner in-memory ownership.
not own: concrete cognitive payload fields, persistence backends, module scheduling, or external side effects.
fails when: pydantic rejects malformed base fields or a subclass name is missing from the RecordKind registry.
domain: every memory object shares a base identity, causal time range, trace tags, and optional owner-backed lookup surface before specialized knoxel models add domain payload.
invariant: RecordBase.type resolves through the public RecordKind registry, and owner-backed access assumes RecordOwner stores knoxels by stable ids.
"""

from datetime import datetime
import logging
from enum import StrEnum
from typing import Dict, Any
from typing import List
from typing import (
    Optional,
)

from pydantic import BaseModel, Field, PrivateAttr

from pm.records.enums import FrameEnum, RecordSubtypeBase, NoSubtype, SourceKind

logger = logging.getLogger(__name__)

#: Persisted record type names, keyed by the current class name.
#: Seeds, checkpoints, and model-visible files store the historical names.
#: A class rename must never change saved data, so new names map back here.
PERSISTED_TYPE_NAMES: dict[str, str] = {
    "ActionSchemaRecord": "ActionSchemaKnoxel",
    "AttentionFrameRecord": "AttentionFrameKnoxel",
    "AttentionRecord": "AttentionKnoxel",
    "BeliefRecord": "BeliefKnoxel",
    "BeliefShiftRecord": "BeliefShiftKnoxel",
    "BodyEpisodeRecord": "BodyEpisodeKnoxel",
    "CauseEffectRecord": "CauseEffectKnoxel",
    "CoversTicksEventsRecord": "CoversTicksEventsKnoxel",
    "DeclarativeFactRecord": "DeclarativeFactKnoxel",
    "ExecutableSkillRecord": "ExecutableSkillKnoxel",
    "MemoryClusterRecord": "MemoryClusterKnoxel",
    "OntologyBindingRecord": "OntologyBindingKnoxel",
    "OntologyTermRecord": "OntologyTermKnoxel",
    "QualeRecord": "QualeKnoxel",
    "ExpectationRecord": "ExpectationKnoxel",
    "RealityRecord": "RealityKnoxel",
    "RecordBase": "KnoxelBase",
    "SemanticRecord": "SemanticRecordKnoxel",
    "SemanticallyClassifiedRecord": "SemanticallyClassifiedKnoxel",
    "WorldObjectRecord": "WorldObjectKnoxel",
    "WorldPlaceRecord": "WorldPlaceKnoxel",
}


def persisted_type_name(cls: type) -> str:
    """Return the name that seeds, checkpoints, and prompts use for ``cls``."""
    return PERSISTED_TYPE_NAMES.get(cls.__name__, cls.__name__)


def get_causal_datetime() -> datetime:
    """
    Return real datetime now in UTC, or get from test infrastructure for dry runs.
    For when simulating past (not implemented yet)
    """
    return datetime.utcnow()


def get_real_datetime() -> datetime:
    """
    Real world time, for system.
    """
    return datetime.utcnow()

# Sentinel reality id meaning "the real world" until a bootstrap/converter
# materializes the actual RealityKnoxel and rewrites default scopes.
REAL_REALITY_ID: int = -1


class RecordScope(BaseModel):
    """
    Reality scoping for every knoxel (spec v2 §1.1).

    ``reality_id`` references a RealityRecord id (entity-ref pattern).
    ``frame`` separates in-world facts from facts about the reality, so
    knowledge from one reality/frame cannot poison another
    (minecraft house location vs server ip; orgblorg kills goblin vs
    "rick plays orgblorg every friday").
    """
    reality_id: int = Field(default=REAL_REALITY_ID, description="RealityKnoxel id; REAL_REALITY_ID sentinel = the real world.")
    frame: FrameEnum = Field(default=FrameEnum.InWorld)


class Provenance(BaseModel):
    """
    Who/what brought this knoxel into the ghost (spec v2 §1.2).
    """
    source_kind: SourceKind
    source_ref: Optional[str] = Field(default=None, description="e.g. task id, workflow run id, entity id")


class RecordKind(StrEnum):
    Entity = "Entity"
    RealityRecord = "RealityKnoxel"
    Stimulus = "Stimulus"
    Action = "Action"
    ActionSchemaRecord = "ActionSchemaKnoxel"
    Intention = "Intention"
    BeliefRecord = "BeliefKnoxel"
    BeliefShiftRecord = "BeliefShiftKnoxel"
    AttentionRecord = "AttentionKnoxel"
    AttentionFrameRecord = "AttentionFrameKnoxel"
    OntologyTermRecord = "OntologyTermKnoxel"
    OntologyBindingRecord = "OntologyBindingKnoxel"
    QualeRecord = "QualeKnoxel"
    ExpectationRecord = "ExpectationKnoxel"
    CoversTicksEventsRecord = "CoversTicksEventsKnoxel"
    Narrative = "Narrative"
    Feature = "Feature"
    GraphNode = "GraphNode"
    GraphEdge = "GraphEdge"
    ConceptNode = "ConceptNode"
    CodeletPercept = "CodeletPercept"
    MemoryClusterRecord = "MemoryClusterKnoxel"
    DeclarativeFactRecord = "DeclarativeFactKnoxel"
    CauseEffectRecord = "CauseEffectKnoxel"
    VirtualRecord = "VirtualKnoxel"
    Structure = "Structure"
    ModuleOutput = "ModuleOutput"
    ExecutableSkillRecord = "ExecutableSkillKnoxel"
    WorldPlaceRecord = "WorldPlaceKnoxel"
    WorldObjectRecord = "WorldObjectKnoxel"
    BodyEpisodeRecord = "BodyEpisodeKnoxel"
    SemanticRecord = "SemanticRecordKnoxel"


class RecordBase(BaseModel):
    _trace_tags: set[str] = PrivateAttr(default_factory=set)
    _owner: Optional["RecordOwner"] = PrivateAttr(default=None)

    id: int = -1
    tick_id: int = -1
    sub_tick_id: int = 0
    content: str
    embedding: List[float] | None = Field(default=[], repr=False)
    timestamp_creation: datetime = Field(default_factory=datetime.utcnow)
    timestamp_world_begin: datetime = Field(default_factory=get_causal_datetime)
    timestamp_world_end: datetime = Field(default_factory=get_causal_datetime)
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    scope: RecordScope = Field(default_factory=RecordScope, description="Reality + frame scoping; default real/in_world.")
    provenance: Optional[Provenance] = Field(default=None, description="Who/what created this knoxel.")

    def __hash__(self):
        return hash(self.id) + hash(self.timestamp_creation)

    @property
    def type(self) -> RecordKind:
        return RecordKind[self.__class__.__name__]

    def is_beyond_frontier(self, frontier_tick: int, frontier_subtick: int) -> bool:
        """
        Future-ness is computed, not stored (spec v2 §1.4).

        A run's frontier is the tick/sub_tick of the temporally latest
        knoxel already part of the current inbox/execution batch. Knoxels
        beyond it were generated detached from the current causal state
        (fast-path dialogue, async cycles); retrieval for that run excludes
        them and the next cycle rationalizes them post-hoc.
        """
        if self.tick_id > frontier_tick:
            return True
        return self.tick_id == frontier_tick and self.sub_tick_id > frontier_subtick

    @property
    def subtype(self) -> RecordSubtypeBase:
        return self.get_sub_type()

    def get_sub_type(self) -> RecordSubtypeBase:
        return NoSubtype.NoSubtype

    def get_story_element(self) -> str:
        return f"{persisted_type_name(type(self))}: {self.content}"

    def __str__(self):
        return f"{persisted_type_name(type(self))}: {self.content}"

    def to_json(self):
        return self.id

    def __getattribute__(self, name):
        # NOTE: last_accessed is intentionally NOT updated on content reads.
        # Dumping/loading must never mutate state; access tracking is explicit
        # (touch_last_accessed) and driven by retrieval/tooling layers later.
        return super().__getattribute__(name)

    def touch_last_accessed(self) -> None:
        """Explicitly mark this knoxel as accessed (call from retrieval/tools, never from dumps)."""
        object.__setattr__(self, "last_accessed", datetime.utcnow())

    def get_emotional_state_at_time(self) -> str:
        """
        Get the mental state vector at the time this knoxel causally came into being.
        Use an algorithmic approach to turn the vector into a descriptive text by
        - computing baseline and comparing to that
        - finding low and high values and explaning them with descriptive language (x felt absolutey abyssmal, x was really down, x felt somewhat bad, x felt neutral, x felt good, x experienced pure bliss)
        - creating a string with 32 - 256 tokens explaining the emotional state in detail
        With this, emotional state can be essentially vector searched like normal text to find memories with queries such as
            "memories where the ai companion felt betrayed and disappointed, after a normal situation. the sudden swithc highlights the emotional intensity"
        :return:
        """
        return ""

    def get_emotional_appraisal(self) -> str:
        return ""

    def get_emotional_delta(self) -> str:
        return ""

    @property
    def trace_tags(self) -> List[str]:
        return sorted(self._trace_tags)

    def add_trace_tag(self, tag: str) -> None:
        t = str(tag or "").strip()
        if not t:
            return
        self._trace_tags.add(t)


class RecordOwner:
    def __init__(self):
        self.all_knoxels: Dict[int, RecordBase] = {}

    @property
    def sorted_knoxels_raw(self):
        tmp = list(self.all_knoxels.values())
        tmp.sort(key=lambda x: x.timestamp_creation)
        return tmp

    @property
    def sorted_knoxels_causal(self):
        tmp = list(self.all_knoxels.values())
        tmp.sort(key=lambda x: x.timestamp_world_begin)
        return tmp

    @property
    def max_knoxel_id(self):
        return max([k.id for k in self.all_knoxels.values()])
