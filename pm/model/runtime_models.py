from dataclasses import dataclass

from pm.model.knoxel_list import KnoxelList


@dataclass
class ShellCCQUpdate:
    last_causal_id: int
    current_tick: int
    knoxels: KnoxelList
    as_story: str = ""
    as_assistant: str = ""
