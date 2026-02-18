from typing import TYPE_CHECKING, Protocol, Any, List, Dict, Optional
from pm.data_structures import Stimulus, Feature, KnoxelList, ShellCCQUpdate
from pm.llm.llm_proxy import LlmManagerProxy
from pm.mental_state_vectors import FullMentalState

if TYPE_CHECKING:
    from pm.ghosts.ghost_codelets import GhostCodelets

class GhostProtocol(Protocol):
    """Protocol defining the interface that procedures expect from the Ghost."""
    llm: LlmManagerProxy
    current_tick_id: int
    all_features: List[Feature]
    
    # State containers
    input_knoxels: List[Feature]
    primary_stimulus: Optional[Stimulus]
    
    def add_knoxel(self, knoxel: Any) -> int: ...
    def get_knoxel_by_id(self, id: int) -> Any: ...

class BaseProc:
    """Base class for all cognitive procedures."""
    
    @staticmethod
    def run(ghost: GhostProtocol) -> None:
        """Main execution method for the procedure."""
        raise NotImplementedError
