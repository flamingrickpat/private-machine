"""The per-tick container and the checkpoint container of the runtime state.

owns: ``CycleState``, the typed state that one accepted cycle copies and
changes, and ``CycleStateModel``, the whole saved character (records, cycle
states, and persona fields) that the memory store writes to a checkpoint.
not own: the mental state, situational model, workspace, or record models that
``CycleState`` references, and not the persistence code that reads and writes
the checkpoint (see ``pm.memory.store`` and ``pm.memory.yaml_codec``).

``CycleState`` forbids unknown fields on purpose: a checkpoint that predates a
field must fail loudly instead of silently loading a partial mental state.
Read ``pm.records.base`` for the package overview.
"""
from pm import vclock
from datetime import datetime
from typing import Any

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

    A main cycle claims its stimuli in ``stimulus_groups``. A private cycle
    of the inner loop claims none, so its groups are empty. The GWT turn
    record of the same tick names the loop (``GwtTurnRecord.slot_id``).
    """

    model_config = ConfigDict(extra="forbid")

    tick_id: int
    timestamp: datetime = Field(default_factory=vclock.utcnow)

    mental_state: FullMentalState = Field(default_factory=FullMentalState, description="Current EMA-normalized mental state.")
    csm: SituationalModel = Field(default_factory=SituationalModel, description="Unconscious content with activation.")
    structure_ids: list[int] = Field(default_factory=list, description="Structure record ids that are active in the CSM.")
    workspace: GlobalWorkspace = Field(default_factory=GlobalWorkspace, description="Conscious content in shards.")
    codelets: CodeletRegistryState = Field(default_factory=CodeletRegistryState, description="Runtime state of every registered codelet.")
    broadcasts: list[BroadcastRef] = Field(default_factory=list, description="The last CCQ_LENGTH workspace turns, oldest first.")
    stimulus_groups: list[StimulusGroup] = Field(default_factory=list, description="Groups of the stimuli that this cycle claimed.")
    decisions: list[ActionDecision] = Field(default_factory=list, description="In-character decisions of this cycle.")
    dispatches: list[ActionDispatchRef] = Field(default_factory=list, description="Routed actions of this cycle.")

    control_generation: int = Field(default=0, description="Cognitive control generation that admitted this cycle. It survives restarts.")
    qualia_channels: dict[str, QualeChannelState] = Field(default_factory=dict, description="Qualia channel states of a private cycle, keyed by channel id.")

    def claimed_ids(self) -> list[int]:
        """Return the ids of the stimuli that this cycle claimed, in group order."""
        return [record_id for group in self.stimulus_groups for record_id in group.stimulus_ids]

class PersonaConfig(BaseModel):
    """The persona fields that a character home supplies to the runtime."""

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
    personality_profile: list[float] = Field(default_factory=lambda: PersonalityProfile().to_vector())

    @property
    def universal_character_card(self) -> str:
        return self.character_card_story

class CycleStateModel(BaseModel):
    """One saved character: every record, every cycle state, and the persona.

    The memory store dumps this model to a checkpoint and loads it again. The
    knoxel index is a runtime cache, so it is private and rebuilt after load.
    Field names here are persisted YAML keys.
    """
    _knoxel_index: dict[int, RecordBase] = PrivateAttr(default_factory=dict)

    knoxels: list[SerializeAsAny[RecordBase]]
    cycles: list[CycleState]

    previous_emb_model: str = ""

    user_name: str
    companion_name: str
    timestamp_format: str

    character_card_story: str
    character_card_assistant: str | None
    supported_capabilities: list[str] = Field(default_factory=list)
    unsupported_capabilities: list = Field(default_factory=list)
    capability_notes: list = Field(default_factory=list)
    personality_profile: list[float]

    def model_post_init(self, __context: Any) -> None:
        self.rebuild_knoxel_index()

    def rebuild_knoxel_index(self) -> None:
        """Rebuild the id lookup after a load or after in-place record changes."""
        self._knoxel_index = {knoxel.id: knoxel for knoxel in self.knoxels}

    @property
    def all_knoxels(self) -> dict[int, RecordBase]:
        if len(self._knoxel_index) != len(self.knoxels):
            self.rebuild_knoxel_index()
        return self._knoxel_index

    @property
    def sorted_knoxels_raw(self) -> list[RecordBase]:
        """Return the records ordered by real creation time."""
        return sorted(self.knoxels, key=lambda x: x.timestamp_creation)

    @property
    def sorted_knoxels_causal(self) -> list[RecordBase]:
        """Return the records in world-time order, which is story order."""
        return sorted(self.knoxels, key=lambda x: x.timestamp_world_begin)

    def get_knoxel_by_id(self, knoxel_id: int) -> RecordBase | None:
        """Resolve one record id against the rebuilt index."""
        return self.all_knoxels.get(knoxel_id)
