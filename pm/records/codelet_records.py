"""
holds - cognitive structures and module handoff artifacts as owner-backed knoxels.

archetype: information-holder
owns: affect metadata, inline structure relations, module-output provenance, and owner-backed accessors for referenced knoxels.
not own: module scheduling, destination routing, graph persistence, or external tool execution.
fails when: pydantic validation rejects malformed primitive metadata or relation fields.
domain: structures collect knoxel ids and local relationship hints while module outputs record debuggable handoffs between runtime modules.
invariant: serialized models keep references as ids and primitive metadata; live Python objects are resolved only through owner-backed properties.
"""

from typing import Dict, List, Union

from pydantic import BaseModel, Field, JsonValue

from pm.constants import CODELET_BASE_ACTIVATION, CODELET_COOLDOWN_TICKS
from pm.records.base import RecordBase
from pm.records.mental_state import FullMentalState, create_empty_ms_vector
from pm.records.workspace import ShardKey

class AffectMeta(BaseModel):
    """
    Lightweight affect annotation used when scoring knoxel relevance in a procedure.

    This is not a full persistent mental-state object; it is a local weighting tag for
    attention/selection steps where valence, salience, arousal, and dominance influence
    prioritization.
    """
    affective_valence: float = Field(default=0, le=1, ge=-1, description="Affect sign for {companion_name}. [-1..1]. -1 = aversive, 0 = neutral, 1 = positive.")
    incentive_salience: float = Field(default=0, le=1, ge=0, description="How salient is this Percept? [0..1]. 0 = irrelevant, 1 = decisive for {companion_name}.")
    arousal: float = Field(default=0, le=1, ge=-1, description="Activation/urgency proxy")
    dominance: float = Field(default=0, le=1, ge=-1, description="Control/agency proxy")


class StructureRelation(BaseModel):
    source_id: int
    relation: str
    target_id: int
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class Structure(RecordBase):
    """
    Bundle of knoxels sharing some common theme with metadata to allow competition for attention mechanisms.
    """
    active: bool = Field(description="Whether this structure is active or inactive (superceded, ignored)")
    internal_knoxel_ids: List[int] = Field(default_factory=list, description="list of knoxels that contribute (memories, features, beliefs, ...)")
    structure_ids: List[int] = Field(default_factory=list, description="Nested structure ids that contribute to this structure")
    relations: List[StructureRelation] = Field(default_factory=list, description="Inline relation triplets between referenced knoxels")
    affect: AffectMeta = Field(default_factory=AffectMeta, description=" Simple low-dim appraisal vector for runtime importance")
    predecessor_id: int | None = Field(default=None, description="Predecessor structure if there is one")
    primitive_metadata: dict[str, JsonValue] = Field(default_factory=dict, description="Freeform primitive metadata for runtime inspection")
    target_shard: ShardKey | None = Field(default=None, description="Shard that the structure concerns. Attention can choose another shard.")

    @property
    def knoxels(self) -> List[RecordBase]:
        return [x for x in self._owner.all_knoxels.values() if x.id in self.internal_knoxel_ids]

    @property
    def structures(self) -> List["Structure"]:
        return [
            structure
            for structure_id in self.structure_ids
            if isinstance(structure := self._owner.all_knoxels.get(structure_id), Structure)
        ]

    @property
    def predecessor(self) -> Union[None, "Structure"]:
        predecessor = self._owner.all_knoxels.get(self.predecessor_id, None)
        if isinstance(predecessor, Structure):
            return predecessor
        return None


class ModuleDestination(BaseModel):
    module: str
    port: str | None = None


class ModuleOutput(RecordBase):
    source_module: str
    destinations: List[ModuleDestination] = Field(default_factory=list)
    status: str = "completed"
    produced_structure_ids: List[int] = Field(default_factory=list)
    primary_structure_id: int | None = None
    control_knoxel_id: int | None = None
    predecessor_output_ids: List[int] = Field(default_factory=list)
    trace_data: dict[str, JsonValue] = Field(default_factory=dict)
    diagnostics: List[str] = Field(default_factory=list)

    @property
    def structures(self) -> List[Structure]:
        if self._owner is None:
            return []
        return [
            structure
            for structure_id in self.produced_structure_ids
            if isinstance(structure := self._owner.all_knoxels.get(structure_id), Structure)
        ]

    @property
    def primary_structure(self) -> Structure | None:
        if self._owner is None:
            return None
        primary = self._owner.all_knoxels.get(self.primary_structure_id, None)
        if isinstance(primary, Structure):
            return primary
        return None

    @property
    def control(self) -> RecordBase | None:
        if self._owner is None:
            return None
        return self._owner.all_knoxels.get(self.control_knoxel_id, None)

    @property
    def predecessor_outputs(self) -> List["ModuleOutput"]:
        if self._owner is None:
            return []
        return [
            output
            for output_id in self.predecessor_output_ids
            if isinstance(output := self._owner.all_knoxels.get(output_id), ModuleOutput)
        ]


class CodeletPercept(RecordBase):
    """
    Minimal header that every percept carries for selection and blending.
    """
    codelet_name: str
    explanation: str
    reasoning_trace: str
    affect: AffectMeta = Field(default_factory=AffectMeta, description="Simple low-dim appraisal vector for runtime importance")
    mental_state_appraisal_vec: List[float] = Field(default_factory=create_empty_ms_vector, description="The state with the real appraisals and latent state (total, includes state at which feature was created).")
    mental_state_delta_vec: List[float] = Field(default_factory=create_empty_ms_vector, description="The state with no appraisals and the real delta from latent_state - previous_real_state (local delta, can be added up in EMA for 'state during'.")
    internal_knoxel_ids: List[int] = Field(default_factory=list, description="list of knoxels that are based on this percept (new beliefs, attention shifts, goals, or just the features)")

    @property
    def knoxels(self) -> List[RecordBase]:
        return [x for x in self._owner.all_knoxels.values() if x.id in self.internal_knoxel_ids]

    @property
    def mental_state_appraisal(self) -> FullMentalState:
        return FullMentalState.init_from_list(self.mental_state_appraisal_vec)

    @mental_state_appraisal.setter
    def mental_state_appraisal(self, value: FullMentalState) -> None:
        self.mental_state_appraisal_vec = value.to_list()

    @property
    def mental_state_delta(self) -> FullMentalState:
        return FullMentalState.init_from_list(self.mental_state_delta_vec)

    @mental_state_delta.setter
    def mental_state_delta(self, value: FullMentalState) -> None:
        self.mental_state_delta_vec = value.to_list()


class CodeletRuntimeState(BaseModel):
    """The persisted runtime state of one registered codelet.

    Port of ``CodeletRuntime`` from ``private-machine-head/pm/codelets/codelet.py``.
    ``activation`` is the stored activation that the next activation score
    starts from. ``base_activation`` is the long-term base-level activation.
    It rises when a percept of the codelet enters a workspace shard.
    ``cooldown_ticks`` is the number of ticks after a run in which the codelet
    has a lower score. ``last_fired_tick`` is None until the first run.
    ``last_percept_ids`` holds the record ids of the percepts of the last run.
    ``precursor_ids`` holds the record ids of percepts from upstream pathway
    codelets (port of ``precursor_dict``). The CSM view of this codelet
    favors them. A run of the codelet clears them.
    """

    name: str = ""
    activation: float = 0.0
    base_activation: float = CODELET_BASE_ACTIVATION
    cooldown_ticks: int = CODELET_COOLDOWN_TICKS
    last_fired_tick: int | None = None
    total_runs: int = 0
    last_percept_ids: List[int] = Field(default_factory=list)
    precursor_ids: List[int] = Field(default_factory=list)

    def rekey(self, mapping: Dict[int, int]) -> None:
        """Replace pending percept ids by record ids. See ``SituationalModel.rekey``."""
        self.last_percept_ids = [mapping.get(item, item) for item in self.last_percept_ids]
        self.precursor_ids = [mapping.get(item, item) for item in self.precursor_ids]


class CodeletRegistryState(BaseModel):
    """The runtime state of every registered codelet, keyed by codelet name.

    Port of ``CodeletState``. The cycle state owns it and saves it with each
    cycle. The codelet definitions are data in
    ``workflows/main_cognition/codelets.yaml`` and are not saved here. A
    codelet without an entry has the default runtime state.
    """

    codelets: Dict[str, CodeletRuntimeState] = Field(default_factory=dict)

    def runtime(self, name: str) -> CodeletRuntimeState:
        """Return the runtime state of ``name``. Create the default state on first use."""
        if name not in self.codelets:
            self.codelets[name] = CodeletRuntimeState(name=name)
        return self.codelets[name]

    def rekey(self, mapping: Dict[int, int]) -> None:
        """Replace pending percept ids by record ids in every runtime state."""
        for state in self.codelets.values():
            state.rekey(mapping)
