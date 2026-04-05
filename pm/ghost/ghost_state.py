from datetime import datetime
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from pm.model.csm import CSMState
from pm.model.knoxel_common import Stimulus, Action
from pm.model.knoxel_core import KnoxelBase
from pm.model.knoxel_feature import Feature
from pm.model.knoxel_list import KnoxelList
from pm.model.mental_state_vectors import FullMentalState, create_empty_state
from pm.model.needs_state_vectors import NeedsAxesModel, CognitionAxesModel
from pm.subsystems.codelet.codelet import CodeletState


class GhostState(BaseModel):
    tick_id: int
    previous_tick_id: int = -1
    timestamp: datetime = Field(default_factory=datetime.now)
    rating: int = Field(default=0)

    latent_mental_state: FullMentalState = Field(default_factory=create_empty_state)
    csm_state: CSMState = Field(default_factory=CSMState)
    codelet_state: CodeletState = Field(default_factory=CodeletState)
    ccq_state: Dict[int, float] = Field(default_factory=dict)

    #primary_stimulus: Optional[Stimulus] = None
    #attention_candidates: KnoxelList = KnoxelList()
    #attention_focus: KnoxelList = KnoxelList()
    #conscious_workspace: KnoxelList = KnoxelList()
#
    #coalitions_hard: Dict[int, List[KnoxelBase]] = {}
    #coalitions_balanced: Dict[int, List[KnoxelBase]] = {}
    #subjective_experience: Optional[Feature] = None
    #subjective_experience_tool: Optional[Feature] = None
#
    ## Action Deliberation & Simulation Results
    #action_simulations: List[Dict[str, Any]] = []  # Store results of MC simulations [{sim_id, type, content, rating, predicted_state, user_reaction}, ...]
    #selected_action_details: Optional[Dict[str, Any]] = None  # Details of the chosen simulation/action before execution
    #selected_action_knoxel: Optional[Action] = None  # The final Action knoxel generated (set in execute)

    # State snapshots
    state_needs: NeedsAxesModel = NeedsAxesModel()
    state_cognition: CognitionAxesModel = CognitionAxesModel()