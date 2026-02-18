from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from pm.data_structures import Feature, KnoxelHaver
from pm.utils.emb_utils import cosine_sim


class CSMItem(BaseModel):
    knoxel_id: int
    first_tick: int
    last_tick: int
    ticks_in_csm: int = 0
    activation: float = 0.0
    peak_activation: float = 0.0
    merged_ids: List[int] = Field(default_factory=list)


class CSMState(BaseModel):
    csm_item_states: Dict[int, CSMItem] = Field(default_factory=dict)
    gist: str = ""

    @property
    def items(self) -> Dict[int, CSMItem]:
        """
        Backward-compatible alias for older procedure modules that still expect
        `state.items` rather than `state.csm_item_states`.
        """
        return self.csm_item_states


class CSMManager:
    def __init__(
        self,
        knoxel_haver: KnoxelHaver,
        max_items: int = 128,
        decay: float = 0.92,
        merge_sim_threshold: float = 0.86,
        state: CSMState = None,
    ):
        self.knoxel_haver = knoxel_haver
        self.max_items = max_items
        self.decay = decay
        self.merge_sim_threshold = merge_sim_threshold
        self.state: CSMState = state or CSMState()

    def items(self) -> List[CSMItem]:
        return list(self.state.csm_item_states.values())

    def get_active_items(self, min_activation: float = 0.0, limit: Optional[int] = None) -> List[CSMItem]:
        """
        Backward-compatible helper used by several procedure modules.
        """
        active = [x for x in self.items() if x.activation >= min_activation]
        active.sort(key=lambda x: (-x.activation, -x.last_tick, x.knoxel_id))
        if limit is not None:
            return active[:limit]
        return active

    def decay_step(self, decay_factor: Optional[float] = None):
        factor = decay_factor if decay_factor is not None else self.decay
        factor = max(0.0, min(1.0, factor))
        for it in self.state.csm_item_states.values():
            it.activation *= factor
            it.ticks_in_csm += 1
            it.peak_activation = max(it.peak_activation, it.activation)

    def _get_embedding(self, knoxel_id: int) -> Optional[List[float]]:
        knoxel = self.knoxel_haver.all_knoxels.get(knoxel_id)
        if not knoxel or not getattr(knoxel, "embedding", None):
            return None
        emb = knoxel.embedding
        if not emb or len(emb) == 0:
            return None
        return emb

    def _safe_cosine(self, a: List[float], b: List[float]) -> float:
        if not a or not b:
            return 0.0
        if len(a) != len(b):
            return 0.0
        try:
            return float(cosine_sim(a, b))
        except Exception:
            return 0.0

    def _similar(self, a: CSMItem, b: CSMItem) -> bool:
        a_emb = self._get_embedding(a.knoxel_id)
        b_emb = self._get_embedding(b.knoxel_id)
        if not a_emb or not b_emb:
            return False
        return self._safe_cosine(a_emb, b_emb) >= self.merge_sim_threshold

    def _is_permanent_candidate(self, item: CSMItem) -> bool:
        knoxel = self.knoxel_haver.all_knoxels.get(item.knoxel_id)
        if item.peak_activation >= 0.85:
            return True
        if item.ticks_in_csm >= 20:
            return True
        if item.activation >= 0.65:
            return True
        if isinstance(knoxel, Feature) and knoxel.causal:
            return True
        return False

    def add_or_boost(self, item: CSMItem):
        # Try merge with best match
        best_id = None
        best_score = -1.0

        incoming = item.model_copy(deep=True)
        incoming.peak_activation = max(incoming.activation, incoming.peak_activation)

        for fid, it in self.state.csm_item_states.items():
            if self._similar(it, incoming):
                score = 1.0
                if score > best_score:
                    best_score = score
                    best_id = fid

        if best_id is not None:
            ref = self.state.csm_item_states[best_id]
            ref.activation = min(1.0, max(ref.activation, incoming.activation) + 0.15)
            ref.last_tick = max(ref.last_tick, incoming.last_tick)
            ref.peak_activation = max(ref.peak_activation, ref.activation, incoming.peak_activation)
            ref.ticks_in_csm = max(ref.ticks_in_csm, incoming.ticks_in_csm)
            if incoming.knoxel_id != ref.knoxel_id:
                ref.merged_ids.append(incoming.knoxel_id)
        else:
            self.state.csm_item_states[incoming.knoxel_id] = incoming

        if len(self.state.csm_item_states) > self.max_items:
            self._prune_to_capacity(current_tick=incoming.last_tick, max_items=self.max_items)

    def spread_activation(
        self,
        min_source_activation: float = 0.65,
        similarity_threshold: float = 0.50,
        spread_factor: float = 0.12,
        max_neighbors: int = 3,
    ) -> None:
        """
        Spread activation through semantic similarity between CSM items.
        This is deterministic and independent of tick-neighborhood hacks.
        """
        items = self.get_active_items(min_activation=0.0)
        if len(items) < 2:
            return

        deltas: Dict[int, float] = {}
        for src in items:
            if src.activation < min_source_activation:
                continue
            src_emb = self._get_embedding(src.knoxel_id)
            if not src_emb:
                continue

            scored_neighbors: List[Tuple[int, float]] = []
            for dst in items:
                if dst.knoxel_id == src.knoxel_id:
                    continue
                dst_emb = self._get_embedding(dst.knoxel_id)
                if not dst_emb:
                    continue
                sim = self._safe_cosine(src_emb, dst_emb)
                if sim >= similarity_threshold:
                    scored_neighbors.append((dst.knoxel_id, sim))

            scored_neighbors.sort(key=lambda x: (-x[1], x[0]))
            for dst_id, sim in scored_neighbors[:max_neighbors]:
                delta = src.activation * sim * spread_factor
                deltas[dst_id] = deltas.get(dst_id, 0.0) + delta

        for dst_id, delta in deltas.items():
            item = self.state.csm_item_states.get(dst_id)
            if not item:
                continue
            item.activation = min(1.0, item.activation + delta)
            item.peak_activation = max(item.peak_activation, item.activation)

    def _prune_to_capacity(self, current_tick: int, max_items: int) -> None:
        if len(self.state.csm_item_states) <= max_items:
            return

        ranked = list(self.state.csm_item_states.values())
        ranked.sort(
            key=lambda x: (
                not self._is_permanent_candidate(x),
                -x.activation,
                -(current_tick - x.last_tick),
                x.knoxel_id,
            )
        )
        keep = {x.knoxel_id for x in ranked[:max_items]}
        for fid in list(self.state.csm_item_states.keys()):
            if fid not in keep:
                self.state.csm_item_states.pop(fid, None)

    def prune(
        self,
        current_tick: int,
        min_activation: float = 0.1,
        max_idle_ticks: int = 60,
        max_items: Optional[int] = None,
        permanent_max_idle_ticks: int = 240,
    ) -> List[CSMItem]:
        """
        Remove stale/low-value items while preserving permanent candidates longer.
        Returns removed items for downstream persistence hooks.
        """
        removed: List[CSMItem] = []
        for fid, item in list(self.state.csm_item_states.items()):
            idle_ticks = max(0, current_tick - item.last_tick)
            permanent = self._is_permanent_candidate(item)

            remove = False
            if permanent:
                if idle_ticks > permanent_max_idle_ticks and item.activation < (min_activation * 0.5):
                    remove = True
            else:
                if item.activation < min_activation:
                    remove = True
                if idle_ticks > max_idle_ticks:
                    remove = True

            if remove:
                removed.append(item)
                self.state.csm_item_states.pop(fid, None)

        cap = max_items if max_items is not None else self.max_items
        self._prune_to_capacity(current_tick=current_tick, max_items=cap)
        return removed

    def build_gist(self, max_items: int = 3, max_chars_per_item: int = 120) -> str:
        top = self.get_active_items(min_activation=0.15, limit=max_items)
        if not top:
            self.state.gist = ""
            return self.state.gist

        parts: List[str] = []
        for item in top:
            knoxel = self.knoxel_haver.all_knoxels.get(item.knoxel_id)
            if not knoxel or not getattr(knoxel, "content", None):
                continue
            content = knoxel.content.strip().replace("\n", " ")
            if len(content) > max_chars_per_item:
                content = content[: max_chars_per_item - 3].rstrip() + "..."
            parts.append(f"[{item.activation:.2f}] {content}")

        self.state.gist = " | ".join(parts)
        return self.state.gist

    def prune_low(self, min_activation: float = 0.1):
        # Backward compatibility path for older callers.
        tick = max((x.last_tick for x in self.state.csm_item_states.values()), default=0)
        self.prune(current_tick=tick, min_activation=min_activation, max_idle_ticks=10_000)

