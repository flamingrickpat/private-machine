from datetime import datetime
from typing import Dict, Any, List, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, PrivateAttr, SerializeAsAny

from pm.records.base import RecordBase
from pm.records.codelet_records import CodeletRegistryState
from pm.records.cycle_records import ActionDecision, ActionDispatchRef, BroadcastRef, StimulusGroup
from pm.records.mental_state import FullMentalState, PersonalityProfile
from pm.records.qualia_records import QualeChannelState
from pm.records.situational_model import SituationalModel
from pm.records.workspace import GlobalWorkspace


class CycleState(BaseModel):
    """The typed state of one accepted cycle.

    The next cycle copies this state and changes the copy. It does not
    rebuild the state from nothing. The memory store saves one state for each
    accepted tick. Unknown fields raise a validation error, so old data that
    was not migrated fails loudly and does not lose its mental state.

    ``timestamp`` is naive UTC, like the record timestamps in
    ``pm.records.base``. Local wall time corrupts restart-gap comparisons.

    The ledger fields ``stimuli_ids``, ``published_slots``,
    ``attention_frame_id``, and ``commit_hash`` remain for the current
    consumers (spec v2 section 6, decision D-M4-5). A later A19F step
    removes them.
    """

    model_config = ConfigDict(extra="forbid")

    tick_id: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    mental_state: FullMentalState = Field(default_factory=FullMentalState, description="Current EMA-normalized mental state.")
    csm: SituationalModel = Field(default_factory=SituationalModel, description="Unconscious content with activation.")
    structure_ids: List[int] = Field(default_factory=list, description="Structure record ids that are active in the CSM.")
    workspace: GlobalWorkspace = Field(default_factory=GlobalWorkspace, description="Conscious content in shards.")
    codelets: CodeletRegistryState = Field(default_factory=CodeletRegistryState, description="Runtime state of every registered codelet.")
    broadcasts: List[BroadcastRef] = Field(default_factory=list, description="The last CCQ_LENGTH workspace turns, oldest first.")
    stimulus_groups: List[StimulusGroup] = Field(default_factory=list, description="Groups of the stimuli that this cycle claimed.")
    decisions: List[ActionDecision] = Field(default_factory=list, description="In-character decisions of this cycle.")
    dispatches: List[ActionDispatchRef] = Field(default_factory=list, description="Routed actions of this cycle.")

    control_generation: int = Field(default=0, description="Cognitive control generation that admitted this cycle. It survives restarts.")
    qualia_channels: Dict[str, QualeChannelState] = Field(default_factory=dict, description="Qualia channel states of a private cycle, keyed by channel id.")

    stimuli_ids: List[int] = Field(default_factory=list, description="Stimulus knoxel ids claimed by this cycle.")
    published_slots: List[str] = Field(default_factory=list, description="GWT slot ids republished by this cycle (v1: main).")
    attention_frame_id: Optional[int] = Field(default=None, description="AttentionFrameKnoxel id produced by this cycle.")
    commit_hash: Optional[str] = Field(default=None, description="Git commit of this cycle (audit trail). Filled after the commit lands.")

class PersonaConfig(BaseModel):
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
    # The stable traits as a vector at positions 800-814. A persona without
    # traits gets the neutral PersonalityProfile defaults (0.5), not zeros.
    personality_profile: List[float] = Field(default_factory=lambda: PersonalityProfile().to_vector())

    @property
    def universal_character_card(self) -> str:
        return self.character_card_story

class CycleStateModel(BaseModel):
    """
    Container for all data to daaamp and load it.
    """
    _knoxel_index: Dict[int, RecordBase] = PrivateAttr(default_factory=dict)

    knoxels: List[SerializeAsAny[RecordBase]]
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
    def all_knoxels(self) -> Dict[int, RecordBase]:
        if len(self._knoxel_index) != len(self.knoxels):
            self.rebuild_knoxel_index()
        return self._knoxel_index

    @property
    def sorted_knoxels_raw(self) -> List[RecordBase]:
        return sorted(self.knoxels, key=lambda x: x.timestamp_creation)

    @property
    def sorted_knoxels_causal(self) -> List[RecordBase]:
        return sorted(self.knoxels, key=lambda x: x.timestamp_world_begin)

    def get_knoxel_by_id(self, knoxel_id: int) -> RecordBase | None:
        return self.all_knoxels.get(knoxel_id)
