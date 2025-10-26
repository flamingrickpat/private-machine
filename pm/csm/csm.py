from pydantic import BaseModel, Field
from typing import List, Dict, Set, Optional

from pm.data_structures import FeatureType, KnoxelHaver, KnoxelList
from pm.utils.emb_utils import cosine_pair

class CSMItem(BaseModel):
    knoxel_id: int
    first_tick: int
    last_tick: int
    ticks_in_csm: int = 0
    activation: float = 0.0
    merged_ids: List[int] = Field(default_factory=list)

class CSMState(BaseModel):
    items: Dict[int, CSMItem] = Field(default_factory=dict)

class CSMBuffer:
    def __init__(self, knoxel_haver: KnoxelHaver, max_items: int = 128, decay: float = 0.92, merge_sim_threshold: float = 0.86, state: CSMState = None):
        self.knoxel_haver = knoxel_haver
        self.max_items = max_items
        self.decay = decay
        self.merge_sim_threshold = merge_sim_threshold
        self.state: CSMState = state

    def items(self) -> List[CSMItem]:
        return list(self.state.items.values())

    def decay_step(self):
        for it in self.state.items.values():
            it.activation *= self.decay
            it.ticks_in_csm += 1

    def _similar(self, a: CSMItem, b: CSMItem) -> bool:
        return cosine_pair(
            self.knoxel_haver.all_knoxels[a.knoxel_id].embedding,
            self.knoxel_haver.all_knoxels[b.knoxel_id].embedding
        ) >= self.merge_sim_threshold

    def add_or_boost(self, item: CSMItem):
        # Try merge with best match
        best_id = None
        best_score = 0.0
        for fid, it in self.state.items.items():
            if self._similar(it, item):
                best_id = fid
                best_score = 1.0
                break
        if best_id is not None:
            ref = self.state.items[best_id]
            ref.activation = min(1.0, ref.activation + 0.25)
            ref.last_tick = item.last_tick
            ref.merged_ids.append(item.knoxel_id)
        else:
            # Insert new
            self.state.items[item.knoxel_id] = item

        # Prune low-activation if over capacity
        if len(self.state.items) > self.max_items:
            to_drop = sorted(self.state.items.values(), key=lambda x: x.activation)[:len(self.state.items)-self.max_items]
            for d in to_drop:
                self.state.items.pop(d.knoxel_id, None)

    def prune_low(self, min_activation: float = 0.1):
        for fid in list(self.state.items.keys()):
            if self.state.items[fid].activation < min_activation:
                self.state.items.pop(fid, None)

    def get_snapshot(self) -> str:
        kl = KnoxelList()

        for _id in self.state.items.keys():
            kl.add(self.knoxel_haver.all_knoxels[_id])

        res = kl.get_story(self.knoxel_haver)
        return res