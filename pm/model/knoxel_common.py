from typing import List
from typing import (
    Optional,
)

from pydantic import Field

from pm.consts import TIMESTAMP_FORMAT
from pm.model.knoxel_core import KnoxelBase
from pm.model.knoxel_enums import EntityClass, StimulusType, ActionType, ClusterType, NarrativeTypes, KnoxelSubtypeBase


class Entity(KnoxelBase):
    """
    Persistent identity anchor for people, agents, or named actors in the story world.

    Stores the canonical display name in ``content`` plus alternative surface forms in
    ``aliases``. Entity knoxels are used to keep references stable across dialogue,
    summaries, extracted facts, and graph memory updates.
    """
    aliases: List[str]
    entity_class: EntityClass

    def get_story_element(self) -> str:
        return f"Character: {self.content}"


class Stimulus(KnoxelBase):
    """
    Input trigger that started or shaped a cognitive cycle.

    Represents either external events (e.g., user message, world event) or internal
    triggers (e.g., recalled memory). Async fields capture insertion order when stimuli
    arrive off-cycle so later procedures can reconstruct processing chronology.
    """
    source: str = Field(default_factory=str)
    stimulus_type: StimulusType

    based_on_tick: int = Field(default=0)
    async_sub_tick: int = Field(default=0)
    async_tick_insert_begin: bool = Field(default=True)
    async_tick_source_order: int = Field(default=0)

    def get_story_element(self) -> str:
        return f"[{self.stimulus_type.value}] {self.source}: {self.content}"

    def get_sub_type(self) -> KnoxelSubtypeBase:
        return self.stimulus_type


class Action(KnoxelBase):
    """
    Executed output decision produced by the companion.

    Contains the action payload in ``content`` and the typed action channel in
    ``action_type``. ``generated_expectation_ids`` links follow-up Intention knoxels
    created from this action, enabling action -> expectation -> outcome tracing.
    """
    action_type: ActionType  # Using the enum
    generated_expectation_ids: List[int] = Field(default=[], description="IDs of Intention knoxels (expectations) created by this action.")

    def get_story_element(self) -> str:
        return f"[{self.action_type.value}] {self.content}"

    def get_sub_type(self) -> KnoxelSubtypeBase:
        return self.action_type


class Intention(KnoxelBase):
    """
    Active goal or expectation tracked across future ticks.

    Used for both internal drives (``internal=True``) and externally directed
    expectations (``internal=False``), including urgency, valence, salience, fulfillment,
    and lifecycle markers (status/timeout). Often created by actions and later evaluated
    against observed outcomes.
    """
    urgency: float = Field(default=0.5, ge=0.0, le=1.0)
    affective_valence: float = Field(..., ge=-1.0, le=1.0, description="Expected valence if fulfilled (for internal goals) or desired valence if met (for external expectations).")
    incentive_salience: float = Field(default=0.5, ge=0.0, le=1.0, description="How much it 'pulls' attention/action.")
    fulfilment: float = Field(default=0.0, ge=0.0, le=1.0, description="Current fulfillment status (0=not met, 1=fully met).")
    internal: bool = Field(..., description="True = internal goal/drive, False = external expectation of an event/response.")
    originating_action_id: Optional[int] = Field(default=None, description="ID of the Action knoxel that generated this expectation (if internal=False).")
    status: str = Field(default="active", description="active, completed, failed, pending")
    timeout: int = Field(default=100, description="Ticks until this intention expires")


class CoversTicksEventsKnoxel(KnoxelBase):
    """
    Base class for memory artifacts that summarize or derive from event spans.

    The min/max tick and event references define provenance boundaries, so downstream
    retrieval can map abstractions back to original causal features.
    """
    minimum_interlocus: int = Field(default=0, description="To control what type of features are involved. 0 means only public and system features are included. -1 or -2 would incorporate codelet percepts, inner thougths and more.")
    min_tick_id: Optional[int] = Field(default=None, description="Lowest original tick ID covered")
    max_tick_id: Optional[int] = Field(default=None, description="Highest original tick ID covered")
    min_event_id: Optional[int] = Field(default=None, description="Lowest original Event/Feature knoxel ID covered")
    max_event_id: Optional[int] = Field(default=None, description="Highest original Event/Feature knoxel ID covered")


class MemoryClusterKnoxel(CoversTicksEventsKnoxel):
    """
    Consolidated episodic memory block produced by memory consolidation.

    Topical clusters group semantically related events. Temporal clusters summarize
    bounded time ranges (time-of-day/day/week/month/year hierarchies). ``included_*``
    fields preserve traceability to source events or lower-level clusters, while
    ``content`` is the natural-language summary used for retrieval and context packing.
    """
    level: int = Field(..., description="Hierarchy level (e.g., 1=Year...6=TimeOfDay for Temporal, 100 for Topical)")
    cluster_type: ClusterType
    included_event_ids: Optional[str] = Field(default=None, description="Comma-separated knoxel IDs of events included (for Topical clusters)")
    included_cluster_ids: Optional[str] = Field(default=None, description="Comma-separated knoxel IDs of clusters included (for Temporal clusters)")
    token: int = Field(default=0, description="Token count, usually based on summary for Temporal clusters")
    facts_extracted: bool = Field(default=False, description="Flag for Topical clusters: has declarative memory been extracted?")
    temporal_key: Optional[str] = Field(default=None, description="Unique key for merging temporal clusters (e.g., '2023-10-26-MORNING')")
    emotion_description: Optional[str] = Field(default=None, description="todo")
    emotion_embedding: Optional[List[float]] = Field(default=None, description="todo")

    def get_sub_type(self) -> KnoxelSubtypeBase:
        return self.cluster_type

    def get_story_element(self) -> str:
        if self.cluster_type == ClusterType.Temporal and self.content:
            return f"Summary start: {self.timestamp_world_begin.strftime(TIMESTAMP_FORMAT)} \n{self.content}\nSummary end: {self.timestamp_world_end.strftime(TIMESTAMP_FORMAT)}\n"
        elif self.cluster_type == ClusterType.Topical:
            all_knoxels = self._owner.sorted_knoxels_causal
            if all_knoxels is None:
                raise Exception("Must provide knoxels!")

            event_ids = [int(val.strip()) for val in self.included_event_ids.split(",")]
            kmap = {k.id: k for k in all_knoxels}

            buffer = [f"Conversation start: {self.timestamp_world_begin.strftime(TIMESTAMP_FORMAT)}\n"]
            events = []
            for event_id in event_ids:
                event = kmap[event_id]
                events.append(event)

            events.sort(key=lambda e: e.timestamp_world_begin)
            for event in events:
                buffer.append(event.get_story_element())

            buffer.append(f"Conversation end: {self.timestamp_world_end.strftime(TIMESTAMP_FORMAT)}\n")

            topic = "\n".join(buffer)
            return topic
        return super().get_story_element()  # Fallback


class DeclarativeFactKnoxel(CoversTicksEventsKnoxel):
    """
    Structured fact extracted from a memory cluster summary.

    Generated by the declarative extraction + categorization pipeline and linked to the
    originating topical/temporal cluster through ``source_cluster_id``. These knoxels are
    designed for high-precision retrieval: stable facts, categories, importance, and
    time-dependence for ranking and filtering.
    """
    source_cluster_id: Optional[int] = Field(default=None, description="ID of the MemoryClusterKnoxel (Topical) it came from")
    reason: str = Field(description="Explain why you chose these categories, importance, and time_dependent.")
    category: List[str] = Field(description="Short list of the most relevant categories.")
    importance: float = Field(description="Overall importance from 0.0 to 1.0.")
    time_dependent: float = Field(description="0.0 = stable/always true; 1.0 = only true right now (ephemeral).")

    def get_story_element(self) -> str:
        return f"*(Fact: {self.content})*"


class CauseEffectKnoxel(CoversTicksEventsKnoxel):
    """
    Learned causal pattern distilled from interaction history.

    Encodes a situation, hypothesized cause, and observed effect in reusable form for
    prediction and appraisal codelets. Provenance ranges keep each rule grounded in the
    event window it was inferred from.
    """
    situation: str
    cause: str
    effect: str
    category: Optional[str] = Field(default=None, description="Semantic category")
    source_cluster_id: Optional[int] = Field(default=None)

    def get_story_element(self) -> str:
        return f"*(Observed Cause/Effect: {self.content})*"


class Narrative(KnoxelBase):
    """
    Evolving long-horizon self/other narrative for a specific psychological dimension.

    Produced by narrative refinement over relevant causal features and versioned by
    ``last_refined_with_tick``. Narrative knoxels are interpretive priors (identity,
    style, relationship framing), useful for broad context but less authoritative than
    direct facts or raw event windows.
    """
    narrative_type: NarrativeTypes
    target_name: str
    content: str
    last_refined_with_tick: Optional[int] = Field(default=None, description="The tick ID up to which features were considered for this narrative version.")

    def get_sub_type(self) -> KnoxelSubtypeBase:
        return self.narrative_type

    def get_story_element(self) -> str:
        return f"Narrative '{self.narrative_type.value}' for {self.target_name}: \n{self.content}\n"

