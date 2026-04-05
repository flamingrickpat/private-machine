from datetime import datetime
import logging
from enum import StrEnum
from typing import Dict, Any
from typing import List
from typing import (
    Optional,
)

from pydantic import BaseModel, Field, PrivateAttr

from pm.model.knoxel_enums import KnoxelSubtypeBase, NoSubtype
from pm.utils.datetime_utils import get_causal_datetime

logger = logging.getLogger(__name__)


class KnoxelType(StrEnum):
    Entity = "Entity"
    Stimulus = "Stimulus"
    Action = "Action"
    Intention = "Intention"
    CoversTicksEventsKnoxel = "CoversTicksEventsKnoxel"
    Narrative = "Narrative"
    Feature = "Feature"
    GraphNode = "GraphNode"
    GraphEdge = "GraphEdge"
    ConceptNode = "ConceptNode"
    PerceptCoalition = "PerceptCoalition"
    CodeletPercept = "CodeletPercept"
    MemoryClusterKnoxel = "MemoryClusterKnoxel"
    DeclarativeFactKnoxel = "DeclarativeFactKnoxel"
    CauseEffectKnoxel = "CauseEffectKnoxel"
    VirtualKnoxel = "VirtualKnoxel"


class KnoxelBase(BaseModel):
    _trace_tags: set[str] = PrivateAttr(default_factory=set)
    _owner: Optional["KnoxelHaver"] = PrivateAttr(default=None)

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

    @property
    def type(self) -> KnoxelType:
        return KnoxelType[self.__class__.__name__]

    @property
    def subtype(self) -> KnoxelSubtypeBase:
        return self.get_sub_type()

    def get_sub_type(self) -> KnoxelSubtypeBase:
        return NoSubtype.NoSubtype

    def get_story_element(self) -> str:
        return f"{self.__class__.__name__}: {self.content}"

    def __str__(self):
        return f"{self.__class__.__name__}: {self.content}"

    def to_json(self):
        return self.id

    def __getattribute__(self, name):
        if name == "content":
            object.__setattr__(self, "last_accessed", datetime.utcnow())
        return super().__getattribute__(name)

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


class KnoxelHaver:
    def __init__(self):
        self.all_knoxels: Dict[int, KnoxelBase] = {}

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
