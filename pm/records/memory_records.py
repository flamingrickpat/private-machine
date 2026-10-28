
from pm import vclock
from typing import Any, List
from typing import (
    Optional,
)

from pydantic import AliasChoices, Field, PrivateAttr

from pm.records.base import RecordBase
from pm.records.enums import *
from pm.records.taxonomy import EpistemicStatus, MemoryTaxonomy, MemoryVolatility, RecordStatus, RecordType, TruthStatus
from pm.records.mental_state import FullMentalState, VectorModelReservedSize, create_empty_ms_vector
from datetime import datetime
from pm.records.base import Provenance
from pm.records.lineage import RecordLineage
from pm.records.enums import FeatureType, SourceKind

QUOTE_START = '"'
QUOTE_END = '"'
TIMESTAMP_FORMAT = "%A, %d.%m.%y %H:%M"


def _get_owner_entity_id(owner: Any, name: str) -> int | None:
    clean_name = str(name or "").strip()
    if not owner or not clean_name:
        return None

    get_entity_id = getattr(owner, "get_entity_id", None)
    if callable(get_entity_id):
        entity_id = get_entity_id(clean_name)
        if entity_id is not None:
            return int(entity_id)

    for knoxel in getattr(owner, "all_knoxels", {}).values():
        if not isinstance(knoxel, Entity):
            continue
        aliases = getattr(knoxel, "aliases", []) or []
        if clean_name == knoxel.content or clean_name in aliases:
            return int(knoxel.id)

    add_knoxel = getattr(owner, "add_knoxel", None)
    if callable(add_knoxel):
        entity = Entity(content=clean_name, aliases=[], entity_class=EntityClass.Agent)
        created = add_knoxel(entity, generate_embedding=False)
        return int(created.id)

    return None


def _get_owner_entity_name(owner: Any, entity_id: int) -> str | None:
    if not owner or entity_id is None or int(entity_id) <= 0:
        return None

    get_entity_name = getattr(owner, "get_entity_name", None)
    if callable(get_entity_name):
        name = get_entity_name(int(entity_id))
        if name:
            return str(name)

    knoxel = getattr(owner, "all_knoxels", {}).get(int(entity_id))
    if isinstance(knoxel, Entity):
        return knoxel.content
    return None


def _coerce_full_mental_state(value: Any) -> FullMentalState:
    if isinstance(value, FullMentalState):
        return value
    if isinstance(value, (list, tuple)):
        vector = [float(item or 0.0) for item in value]
        if len(vector) < VectorModelReservedSize:
            vector.extend([0.0] * (VectorModelReservedSize - len(vector)))
        return FullMentalState.init_from_list(vector[:VectorModelReservedSize])
    if isinstance(value, dict):
        return FullMentalState.model_validate(value)
    raise TypeError(f"Cannot convert {type(value).__name__} to FullMentalState")


class Entity(RecordBase):
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


class Stimulus(RecordBase):
    """
    Input trigger that started or shaped a cognitive cycle.

    Represents either external events (e.g., user message, world event) or internal
    triggers (e.g., recalled memory). Async fields capture insertion order when stimuli
    arrive off-cycle so later procedures can reconstruct processing chronology.
    """
    source: str
    entity_id: int = Field(default=0)
    stimulus_type: StimulusType

    based_on_tick: int = Field(default=0)
    async_sub_tick: int = Field(default=0)
    async_tick_insert_begin: bool = Field(default=True)
    async_tick_source_order: int = Field(default=0)

    def get_story_element(self) -> str:
        return f"[{self.stimulus_type.value}] {self.source}: {self.content}"

    def get_sub_type(self) -> RecordSubtypeBase:
        return self.stimulus_type


class Action(RecordBase):
    """
    Executed output decision produced by the companion.

    Contains the action payload in ``content`` and the typed action channel in
    ``action_type``. ``generated_expectation_ids`` links follow-up Intention knoxels
    created from this action, enabling action -> expectation -> outcome tracing.
    """
    action_type: ActionType  # Using the enum
    generated_expectation_ids: List[int] = Field(default_factory=list)

    def get_story_element(self) -> str:
        return f"[{self.action_type.value}] {self.content}"

    def get_sub_type(self) -> RecordSubtypeBase:
        return self.action_type


class RealityRecord(RecordBase):
    """
    A reality the companion can hold knowledge about: real life, an embodied
    game world (the minecraft body), a fantasy campaign, the system itself.

    Other knoxels reference realities via ``scope.reality_id`` (entity-ref
    pattern), so knowledge from one reality cannot poison another
    ("orgblorg kills goblin" vs "rick plays orgblorg every friday").
    ``content`` = reality name, ``metadata`` = free description/props.
    """
    kind: RealityKind = RealityKind.Real

    def get_story_element(self) -> str:
        return f"[Reality:{self.kind.value}] {self.content}"

    def get_sub_type(self) -> RecordSubtypeBase:
        return self.kind


class ActionSchemaRecord(RecordBase):
    """
    A plan spec selected by the personality simulation - never prose.

    Evolves the limited Action object into the workflow era: the person-sim
    decides WHAT to do and with which expectations; a separate skill
    (roleplay/assistant/single-shot MCP) verbalizes or executes it.
    ``content`` = human readable summary of the plan.
    """
    action_type: ActionType = Field(description="The action channel this schema serves.")
    status: ActionSchemaStatus = Field(default=ActionSchemaStatus.Proposed)
    intention_ids: List[int] = Field(default_factory=list, description="Goals/intentions this schema serves.")
    expectation: str = Field(default="", description="What the persona expects to happen when executed.")
    tool_usage: str = Field(default="", description="MCP/skill/workflow usage hint for the executor.")
    execution_mode: ActionSchemaExecutionMode = Field(default=ActionSchemaExecutionMode.ProseExternal)
    verbalized_by: ActionSchemaVerbalization = Field(default=ActionSchemaVerbalization.Roleplay)

    def get_story_element(self) -> str:
        return f"[ActionSchema:{self.action_type.value}:{self.status.value}] {self.content}"

    def get_sub_type(self) -> RecordSubtypeBase:
        return self.action_type


class CoversTicksEventsRecord(RecordBase):
    """
    Base class for memory artifacts that summarize or derive from event spans.

    The min/max tick and event references define provenance boundaries, so downstream
    retrieval can map abstractions back to original causal features.
    """
    minimum_interlocus: int = Field(default=0, description="To control what type of features are involved. 0 means only public and system features are included. "
                                                           "-1 or -2 would incorporate codelet percepts, inner thougths and more.")
    min_tick_id: Optional[int] = Field(default=None, description="Lowest original tick ID covered")
    max_tick_id: Optional[int] = Field(default=None, description="Highest original tick ID covered")
    min_event_id: Optional[int] = Field(default=None, description="Lowest original Event/Feature knoxel ID covered")
    max_event_id: Optional[int] = Field(default=None, description="Highest original Event/Feature knoxel ID covered")


class MemoryClusterRecord(CoversTicksEventsRecord):
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

    def get_sub_type(self) -> RecordSubtypeBase:
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


class SemanticallyClassifiedRecord(RecordBase):
    """
    Semantic classification layer for a single memory claim.

    Added by memory reconsolidation (batch semantic classifier) on top of the raw
    claim text. Orthogonal to the pm category palette (topical buckets) and to
    importance/time_dependent (ranking): this layer answers "who is the claim about,
    what kind of claim is it, how was it obtained, and what is its canonical form".

    ``essence`` is the terse canonical sentence; ``(subject_entity_id, essence)``
    is the identity key used for cross-source deduplication into canonical records.
    """
    _subject_name_cache: str = PrivateAttr(default="")

    subject_entity_id: int = Field(default=0, description="Entity this claim is about. 0 = unknown/unresolved.")
    record_type: RecordType = Field(default=RecordType.Unclassified, description="Claim kind (fact/preference/goal/...). Drives routing and volatility.")
    object_kind: str = Field(default="", description="Short category of what the claim is about (e.g. daily_routine, communication_style).")
    object_value: str = Field(default="", description="Short clean noun phrase: what the claim is about.")
    epistemic_status: EpistemicStatus = Field(default=EpistemicStatus.Observed, description="How the claim was obtained (trust dimension).")
    confidence: float | None = Field(default=None, description="Classifier confidence 0.0-1.0. None = not yet classified.")
    topic_card: str = Field(default="", description="Short lowercase topic-card name grouping related claims.")
    essence: str = Field(default="", description="One terse canonical sentence capturing the claim. Dedup identity together with subject_entity_id.")
    semantic_record_id: Optional[int] = Field(default=None, description="Back-pointer to the canonical SemanticRecordKnoxel this claim was merged into (self-id on the record itself).")

    @property
    def subject_name(self) -> str:
        return _get_owner_entity_name(self._owner, self.subject_entity_id) or self._subject_name_cache

    @subject_name.setter
    def subject_name(self, value: str) -> None:
        clean_name = str(value or "").strip()
        entity_id = _get_owner_entity_id(self._owner, clean_name)
        if entity_id is None:
            self._subject_name_cache = clean_name
            return
        self.subject_entity_id = entity_id
        self._subject_name_cache = ""

    @property
    def dedup_key(self) -> str:
        return MemoryTaxonomy.dedup_key(self.subject_entity_id, self.essence or self.content)

    @property
    def is_semantically_classified(self) -> bool:
        return self.record_type != RecordType.Unclassified and self.confidence is not None

    def get_sub_type(self) -> RecordSubtypeBase:
        return self.record_type


class DeclarativeFactRecord(CoversTicksEventsRecord, SemanticallyClassifiedRecord):
    """
    Structured fact extracted from a memory cluster summary.

    Generated by the declarative extraction + categorization pipeline and linked to the
    originating topical/temporal cluster through ``source_cluster_id``. These knoxels are
    designed for high-precision retrieval: stable facts, categories, importance, and
    time-dependence for ranking and filtering. The semantic classification layer
    (subject/record type/epistemic/essence) is added by the batch semantic classifier
    in memory reconsolidation; canonical cross-source identity lives in SemanticRecord.
    """
    source_cluster_id: Optional[int] = Field(default=None, description="ID of the MemoryClusterKnoxel (Topical) it came from")
    reason: str = Field(description="Explain why you chose these categories, importance, and time_dependent.")
    category: List[str] = Field(description="Short list of the most relevant categories.")
    importance: float = Field(description="Overall importance from 0.0 to 1.0.")
    time_dependent: float = Field(description="0.0 = stable/always true; 1.0 = only true right now (ephemeral).")

    def get_story_element(self) -> str:
        return f"*(Fact: {self.content})*"


class SemanticRecord(SemanticallyClassifiedRecord):
    """
    Canonical semantic claim: the deduplicated, cross-source identity of one fact.

    Consolidation may produce the same claim multiple times and from multiple
    sources (a DeclarativeFactRecord and one or more GraphEdges can state the
    identical thing). The canonical record is created once per
    ``(subject_entity_id, essence)`` identity and collects every source knoxel
    in ``source_knoxel_ids``. Sources get ``semantic_record_id`` back-pointers.

    Records are what retrieval-facing tools and the derived semantic tree should
    reference. Lifecycle relations (supersedes/contradicted_by/supports) link
    record knoxel ids.
    """
    record_id: str = Field(default="", description="Stable human-facing record id (mem_<ts>_<hash>_<slug>), assigned at creation.")
    subject_label: str = Field(default="", description="Persisted display name of the subject entity at creation time.")
    importance: float = Field(default=0.5, description="Canonical salience 0.0-1.0, aggregated from sources.")
    volatility: MemoryVolatility = Field(default=MemoryVolatility.Stable, description="Coarse expected persistence derived from time_dependent.")
    status: RecordStatus = Field(default=RecordStatus.Active)
    truth_status: TruthStatus = Field(default=TruthStatus.Truth, description="Epistemic truth lifecycle maintained by consolidation sleep passes.")
    superseded_by_ids: List[int] = Field(default_factory=list, description="Knoxel/record ids that supersede or falsify this knowledge (evidence chain).")
    source_knoxel_ids: List[int] = Field(default_factory=list, description="Knoxel ids (facts/edges) that state this claim.")
    source_texts: List[str] = Field(default_factory=list, description="Alternative phrasings of the claim from other sources (alias texts).")
    source_method: str = Field(default="consolidation", description="How the record was produced (consolidation | migration).")
    supersedes_record_ids: List[int] = Field(default_factory=list)
    contradicted_by_record_ids: List[int] = Field(default_factory=list)
    supports_record_ids: List[int] = Field(default_factory=list)
    related_record_ids: List[int] = Field(default_factory=list)

    @property
    def recurrence(self) -> int:
        return len(self.source_knoxel_ids)

    def get_story_element(self) -> str:
        return f"*(Record [{self.record_type.value}]: {self.content})*"


class CauseEffectRecord(CoversTicksEventsRecord):
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


class Narrative(RecordBase):
    """
    Evolving long-horizon self/other narrative for a specific psychological dimension.

    Produced by narrative refinement over relevant causal features and versioned by
    ``last_refined_with_tick``. Narrative knoxels are interpretive priors (identity,
    style, relationship framing), useful for broad context but less authoritative than
    direct facts or raw event windows.
    """
    _target_name_cache: str = PrivateAttr(default="")

    narrative_type: NarrativeTypes
    entity_id: int = Field(default=0, description="Entity ID if the feature is caused by entity, otherwise 0.")
    last_refined_with_tick: Optional[int] = Field(default=None, description="The tick ID up to which features were considered for this narrative version.")

    @property
    def target_name(self) -> str:
        return _get_owner_entity_name(self._owner, self.entity_id) or self._target_name_cache

    @target_name.setter
    def target_name(self, value: str) -> None:
        clean_name = str(value or "").strip()
        entity_id = _get_owner_entity_id(self._owner, clean_name)
        if entity_id is None:
            self._target_name_cache = clean_name
            return
        self.entity_id = entity_id
        self._target_name_cache = ""

    @property
    def source(self) -> str:
        return self.target_name

    @source.setter
    def source(self, value: str) -> None:
        self.target_name = value

    def get_sub_type(self) -> RecordSubtypeBase:
        return self.narrative_type

    def get_story_element(self) -> str:
        return f"Narrative '{self.narrative_type.value}' for {self.source}: \n{self.content}\n"


class Feature(RecordBase):
    """
    Atomic causal/mental event emitted by the LIDA-style cognitive pipeline.
    Feature knoxels are the primary high-resolution evidence stream: dialogue, thoughts,
    appraisals, expectations, action traces, and other simulation artifacts. Memory
    consolidation consumes these features to build topical/temporal clusters, facts,
    and narratives. ``causal`` marks whether an item actually entered the active
    workspace/history versus being a transient candidate.

    While mopst knoxels are data sources (memories), or objects that actually modify the state (attention shift, belief update, etc), the feature is the
    knoxel that makes the "raw" knoxels available to the conscious agent and other prompts.

    When the system decides to recall a memory, update attention, appraise an input, this will affect the system state.
    To additionally make the event and update itself visible to the prompt explicitely, a causal feature will be generated.
    """
    _source_cache: str = PrivateAttr(default="")

    feature_type: FeatureType
    feature_subtype: FeatureSubtype | None = Field(default=None)
    interlocus: InterlocusType = Field(default=InterlocusType.Undefined, description="Visibility to conscious agent, public and reportable are dialoge and thoughts, unreportable are third-person narrative that they might not be fully aware of.")
    causal: bool = Field(default=False, description="Did this feature ever enter workspace or stay 'virtual'?")
    entity_id: int = Field(default=0, description="Entity ID if the feature is caused by entity, otherwise 0.")
    override_source: str | None = Field(default=None, description="Override source for this feature, if there is no entity id (codelt or subsystem).")
    addressee_entity_id: int = Field(default=0, description="Entity ID of the listener of a dialogue line, otherwise 0. "
                                                            "A companion reply names the entity that it answers. 0 means the user or unknown.")
    mental_state_appraisal_vec: List[float] = Field(default_factory=create_empty_ms_vector, description="The state with the real appraisals and latent state (total, includes state at which feature was created).")
    mental_state_delta_vec: List[float] = Field(default_factory=create_empty_ms_vector, description="The state with no appraisals and the real delta from latent_state - previous_real_state (local delta, can be added up in EMA for 'state during'.")

    @property
    def session_id(self) -> Optional[str]:
        """
        Mechanical conversation-thread id (spec v2 §4). Metadata only -
        sessions are not first-class objects; the id groups dialogue when
        injected as memory.
        """
        value = self.metadata.get("session_id")
        return str(value) if value is not None else None

    def set_session_id(self, session_id: str) -> None:
        self.metadata["session_id"] = str(session_id)

    def __str__(self):
        return f"{self.__class__.__name__} ({self.feature_type}): {self.content}"

    @property
    def source(self) -> str:
        if self.override_source:
            return self.override_source
        return _get_owner_entity_name(self._owner, self.entity_id) or self._source_cache

    @source.setter
    def source(self, value: str) -> None:
        clean_name = str(value or "").strip()
        entity_id = _get_owner_entity_id(self._owner, clean_name)
        if entity_id is None:
            self._source_cache = clean_name
            return
        self.entity_id = entity_id
        self.override_source = None
        self._source_cache = ""

    @property
    def mental_state_appraisal(self) -> FullMentalState:
        return FullMentalState.init_from_list(self.mental_state_appraisal_vec)

    @mental_state_appraisal.setter
    def mental_state_appraisal(self, value: Any) -> None:
        self.mental_state_appraisal_vec = _coerce_full_mental_state(value).to_list()

    @property
    def mental_state_delta(self) -> FullMentalState:
        return FullMentalState.init_from_list(self.mental_state_delta_vec)

    @mental_state_delta.setter
    def mental_state_delta(self, value: Any) -> None:
        self.mental_state_delta_vec = _coerce_full_mental_state(value).to_list()

    def get_story_element(self) -> str:
        source = self.source
        story_map = {
            FeatureType.Dialogue: f'{source} says: {QUOTE_START}{self.content}{QUOTE_END}',
            FeatureType.InnerMonologue: f'{source} thinks: *{self.content}*',
            FeatureType.WorldEvent: f'{self.content}',
            FeatureType.StateUpdate: f'{self.subtype}: {self.content}',
        }
        return story_map.get(self.feature_type, self.content)

    def get_sub_type(self) -> RecordSubtypeBase:
        return self.feature_type


def new_feature(ghost, content, feature_type, *, source, interlocus=None, causal=True, changes=None,
             source_kind=None, metadata=None, lineage: RecordLineage | None = None, entity_id: int = 0):
    """Create an attributed feature, optionally inside the caller's write batch.

    Lineage preserves original episode links. With a batch, the caller owns
    the commit and cancellation boundary. Without one, the store commits it.
    ``entity_id`` names the speaker entity. Use it when the entity is new in
    the same batch, because the batch resolves names only from its snapshot.
    """
    kwargs = {
        "content": content,
        "feature_type": feature_type,
        "causal": causal,
        "override_source": source,
        "lineage": lineage,
        "entity_id": entity_id,
    }
    if interlocus is not None:
        kwargs["interlocus"] = interlocus
    if source_kind is not None:
        kwargs["provenance"] = Provenance(source_kind=source_kind, source_ref=str(source))
    elif feature_type == FeatureType.Dialogue:
        kind = SourceKind.User if source == ghost.get_user_name() else SourceKind.Self
        kwargs["provenance"] = Provenance(source_kind=kind, source_ref=str(source))
    kwargs["metadata"] = {
        "memory_kind": ("utterance" if feature_type == FeatureType.Dialogue else
                        "speculation" if feature_type in {FeatureType.Thought, FeatureType.InnerMonologue} else
                        "executed_event"),
        "executed_at": vclock.utcnow().isoformat(),
    }
    if metadata:
        kwargs["metadata"].update(metadata)
    create = ghost.store_knoxel if changes is None else changes.create
    return create(Feature(**kwargs))
