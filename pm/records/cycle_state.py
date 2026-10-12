from datetime import datetime
from typing import Dict, Any, List, Optional

from pydantic import AliasChoices, BaseModel, Field, PrivateAttr, SerializeAsAny

from pm.records.base import KnoxelBase
from pm.records.mental_state import FullMentalState, VectorModelReservedSize, create_empty_ms_vector
from pm.records.needs_axes import NeedsAxesModel, CognitionAxesModel


def _coerce_full_mental_state(value: FullMentalState | List[float] | dict) -> FullMentalState:
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


class CycleState(BaseModel):
    """
    The state for a cycle with ticks and all that.

    v2 fields (spec §6, decision D-M4-5): the cycle ledger records which
    stimuli the cycle claimed, which GWT slots it published, the attention
    frame it produced, and the git commit that sealed the cycle.
    """
    tick_id: int
    previous_tick_id: int = -1
    timestamp: datetime = Field(default_factory=datetime.now)
    rating: int = Field(default=0)

    mental_state_latent_vec: List[float] = Field(default_factory=create_empty_ms_vector, description="current latent state, for quick access")
    state_needs: NeedsAxesModel = Field(default_factory=NeedsAxesModel, description="keeping track of physiological needs")
    state_cognition: CognitionAxesModel = Field(default_factory=CognitionAxesModel, description="keeping track of cognitive modifiers state")

    stimuli_ids: List[int] = Field(default_factory=list, description="Stimulus knoxel ids claimed by this cycle.")
    published_slots: List[str] = Field(default_factory=list, description="GWT slot ids republished by this cycle (v1: main).")
    attention_frame_id: Optional[int] = Field(default=None, description="AttentionFrameKnoxel id produced by this cycle.")
    commit_hash: Optional[str] = Field(default=None, description="Git commit of this cycle (audit trail). Filled after the commit lands.")

    internal_data: Dict[str, Any] = Field(default_factory=dict, description="working object that is modified by subsystems and codelets")

    @property
    def mental_state_latent(self) -> FullMentalState:
        return FullMentalState.init_from_list(self.mental_state_latent_vec)

    @mental_state_latent.setter
    def mental_state_latent(self, value: FullMentalState | List[float] | dict) -> None:
        self.mental_state_latent_vec = _coerce_full_mental_state(value).to_list()

class GhostConfig(BaseModel):
    user_name: str
    companion_name: str
    timestamp_format: str = "%Y-%m-%d %H:%M"

    character_card_story: str = Field(
        default="",
        validation_alias=AliasChoices("character_card_story", "universal_character_card"),
    )
    character_card_assistant: str | None = None
    supported_capabilities: list[str] = Field(default_factory=list)
    unsupported_capabilities: list = Field(default_factory=list)
    capability_notes: list = Field(default_factory=list)
    personality_profile: List[float] = Field(default_factory=create_empty_ms_vector)

    @property
    def universal_character_card(self) -> str:
        return self.character_card_story

class CycleStateModel(BaseModel):
    """
    Container for all data to daaamp and load it.
    """
    _knoxel_index: Dict[int, KnoxelBase] = PrivateAttr(default_factory=dict)

    knoxels: List[SerializeAsAny[KnoxelBase]]
    cycles: List[CycleState]

    previous_emb_model: str = ""

    user_name: str
    companion_name: str
    timestamp_format: str

    character_card_story: str
    character_card_assistant: str | None
    supported_capabilities: list[str] = Field(default_factory=list)
    unsupported_capabilities: list = Field(default_factory=list)
    capability_notes: list = Field(default_factory=list)
    personality_profile: List[float]

    def model_post_init(self, __context: Any) -> None:
        self.rebuild_knoxel_index()

    def rebuild_knoxel_index(self) -> None:
        self._knoxel_index = {knoxel.id: knoxel for knoxel in self.knoxels}
        for knoxel in self.knoxels:
            knoxel._owner = self

    @property
    def all_knoxels(self) -> Dict[int, KnoxelBase]:
        if len(self._knoxel_index) != len(self.knoxels):
            self.rebuild_knoxel_index()
        return self._knoxel_index

    @property
    def sorted_knoxels_raw(self) -> List[KnoxelBase]:
        return sorted(self.knoxels, key=lambda x: x.timestamp_creation)

    @property
    def sorted_knoxels_causal(self) -> List[KnoxelBase]:
        return sorted(self.knoxels, key=lambda x: x.timestamp_world_begin)

    def get_knoxel_by_id(self, knoxel_id: int) -> KnoxelBase | None:
        return self.all_knoxels.get(knoxel_id)
