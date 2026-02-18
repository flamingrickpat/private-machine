import logging
import math
from dataclasses import dataclass
from typing import List, Optional

from pm.data_structures import ClusterType, Feature, FeatureType, MemoryClusterKnoxel
from pm.ghosts.knoxel_trace import trace_knoxel_flow
from pm.ghosts.procedures.base import BaseProc, GhostProtocol
from pm.utils.emb_utils import cosine_sim

logger = logging.getLogger(__name__)


@dataclass
class _MemoryScore:
    memory: object
    total: float
    semantic: float
    recency: float
    state: float
    mood: float


class PamProc(BaseProc):
    """
    Perceptual Associative Memory (PAM) Procedure.
    Retrieves memories via deterministic blended ranking:
    semantic relevance + recency + mental-state proximity + mood bias.
    """

    @staticmethod
    def _safe_cosine(a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        try:
            return float(cosine_sim(a, b))
        except Exception:
            return 0.0

    @staticmethod
    def _get_current_state_signature(ghost: GhostProtocol) -> dict:
        sig = {
            "valence": 0.0,
            "arousal": 0.5,
            "dominance": 0.5,
        }
        if not getattr(ghost, "current_state", None):
            return sig
        ms = getattr(ghost.current_state, "latent_mental_state", None)
        if not ms or not getattr(ms, "state_core", None):
            return sig

        core = ms.state_core
        sig["valence"] = float(getattr(core, "valence", 0.0))
        sig["arousal"] = float(getattr(core, "arousal", 0.5))
        sig["dominance"] = float(getattr(core, "dominance", 0.5))
        return sig

    @staticmethod
    def _get_memory_state_signature(memory) -> Optional[dict]:
        meta = getattr(memory, "metadata", None) or {}
        sig = meta.get("mental_state_signature")
        if isinstance(sig, dict):
            return sig
        return None

    @staticmethod
    def _state_proximity(current_sig: dict, memory_sig: Optional[dict]) -> float:
        if not memory_sig:
            return 0.5

        def _delta(key: str, default: float) -> float:
            return abs(float(current_sig.get(key, default)) - float(memory_sig.get(key, default)))

        # valence in [-1,1] so normalize by 2; others in [0,1].
        valence_diff = _delta("valence", 0.0) / 2.0
        arousal_diff = _delta("arousal", 0.5)
        dominance_diff = _delta("dominance", 0.5)
        avg_diff = (valence_diff + arousal_diff + dominance_diff) / 3.0
        return max(0.0, min(1.0, 1.0 - avg_diff))

    @staticmethod
    def _mood_bias(current_sig: dict, memory) -> float:
        current_valence = float(current_sig.get("valence", 0.0))
        meta = getattr(memory, "metadata", None) or {}

        mem_valence = None
        if "valence_hint" in meta:
            mem_valence = float(meta["valence_hint"])
        elif hasattr(memory, "affective_valence") and getattr(memory, "affective_valence") is not None:
            mem_valence = float(getattr(memory, "affective_valence"))

        if mem_valence is None:
            return 0.5

        diff = abs(current_valence - mem_valence) / 2.0
        return max(0.0, min(1.0, 1.0 - diff))

    @staticmethod
    def _recency_weight(current_tick: int, memory) -> float:
        tick_id = int(getattr(memory, "tick_id", -1))
        if tick_id < 0 or current_tick < 0:
            return 0.5

        ticks_ago = max(0, current_tick - tick_id)
        # Half-life around 40 ticks.
        return math.pow(0.5, ticks_ago / 40.0)

    @staticmethod
    def _rank_memories(ghost: GhostProtocol, query_embedding: List[float], limit: int = 5) -> List[_MemoryScore]:
        candidates = []
        candidates.extend(list(getattr(ghost, "all_declarative_facts", []) or []))
        candidates.extend(list(getattr(ghost, "all_episodic_memories", []) or []))
        if not candidates:
            return []

        current_sig = PamProc._get_current_state_signature(ghost)
        current_tick = int(getattr(ghost, "current_tick_id", 0))

        ranked: List[_MemoryScore] = []
        for mem in candidates:
            emb = getattr(mem, "embedding", None)
            if not emb:
                continue

            semantic = PamProc._safe_cosine(query_embedding, emb)
            recency = PamProc._recency_weight(current_tick, mem)
            state = PamProc._state_proximity(current_sig, PamProc._get_memory_state_signature(mem))
            mood = PamProc._mood_bias(current_sig, mem)

            # Deterministic blending.
            total = (semantic * 0.58) + (recency * 0.20) + (state * 0.17) + (mood * 0.05)
            ranked.append(
                _MemoryScore(
                    memory=mem,
                    total=total,
                    semantic=semantic,
                    recency=recency,
                    state=state,
                    mood=mood,
                )
            )

        ranked.sort(
            key=lambda x: (
                -x.total,
                -x.semantic,
                -x.recency,
                getattr(x.memory, "id", 0),
            )
        )
        return ranked[:limit]

    @staticmethod
    def _build_query_embedding(ghost: GhostProtocol) -> List[float]:
        if not getattr(ghost, "primary_stimulus", None) or not ghost.primary_stimulus.content:
            return []
        try:
            return ghost.llm.get_embedding(ghost.primary_stimulus.content)
        except Exception as e:
            logger.error("PAM: Failed to embed stimulus: %s", e)
            return []

    @staticmethod
    def _collect_candidates(ghost: GhostProtocol) -> List[object]:
        candidates = []
        candidates.extend(list(getattr(ghost, "all_declarative_facts", []) or []))
        candidates.extend(list(getattr(ghost, "all_episodic_memories", []) or []))
        return candidates

    @staticmethod
    def _rank_candidates(
        ghost: GhostProtocol,
        query_embedding: List[float],
        candidates: List[object],
        limit: int = 5,
    ) -> List[_MemoryScore]:
        if not candidates:
            return []

        current_sig = PamProc._get_current_state_signature(ghost)
        current_tick = int(getattr(ghost, "current_tick_id", 0))

        ranked: List[_MemoryScore] = []
        for mem in candidates:
            emb = getattr(mem, "embedding", None)
            if not emb:
                continue

            semantic = PamProc._safe_cosine(query_embedding, emb)
            recency = PamProc._recency_weight(current_tick, mem)
            state = PamProc._state_proximity(current_sig, PamProc._get_memory_state_signature(mem))
            mood = PamProc._mood_bias(current_sig, mem)

            total = (semantic * 0.58) + (recency * 0.20) + (state * 0.17) + (mood * 0.05)
            ranked.append(
                _MemoryScore(
                    memory=mem,
                    total=total,
                    semantic=semantic,
                    recency=recency,
                    state=state,
                    mood=mood,
                )
            )

        ranked.sort(
            key=lambda x: (
                -x.total,
                -x.semantic,
                -x.recency,
                getattr(x.memory, "id", 0),
            )
        )
        return ranked[:limit]

    @staticmethod
    def _emit_recall_features(
        ghost: GhostProtocol,
        top_memories: List[_MemoryScore],
        source_memories: List[object],
    ) -> None:
        source_ids = []
        for sm in source_memories:
            sid = int(getattr(sm, "id", -1) or -1)
            if sid >= 0:
                source_ids.append(sid)

        for scored in top_memories:
            mem = scored.memory
            logger.info(
                "PAM: Recall '%s...' total=%.3f sem=%.3f rec=%.3f state=%.3f mood=%.3f",
                getattr(mem, "content", "")[:30],
                scored.total,
                scored.semantic,
                scored.recency,
                scored.state,
                scored.mood,
            )

            f = Feature(
                content=getattr(mem, "content", ""),
                feature_type=FeatureType.MemoryRecall,
                source="PAM_Recall",
                interlocus=1,
                causal=False,
                embedding=getattr(mem, "embedding", []),
                metadata={
                    "pam_score": scored.total,
                    "pam_components": {
                        "semantic": scored.semantic,
                        "recency": scored.recency,
                        "state": scored.state,
                        "mood": scored.mood,
                    },
                    "source_memory_id": getattr(mem, "id", -1),
                    "source_feature_ids": source_ids,
                },
            )
            ghost.add_knoxel(f)

            if hasattr(ghost, "csm_manager") and ghost.csm_manager is not None:
                from pm.csm.csm import CSMItem

                activation = max(0.30, min(1.0, 0.35 + (scored.total * 0.65)))
                ghost.csm_manager.add_or_boost(
                    CSMItem(
                        knoxel_id=f.id,
                        first_tick=ghost.current_tick_id,
                        last_tick=ghost.current_tick_id,
                        activation=activation,
                        peak_activation=activation,
                    )
                )

    @staticmethod
    def run(ghost: GhostProtocol) -> None:
        logger.info("PAM: Retrieving memories...")

        query_embedding = trace_knoxel_flow(name="PamProc._build_query_embedding", phase="pam")(
            PamProc._build_query_embedding
        )(ghost)
        if not query_embedding:
            return

        candidates = trace_knoxel_flow(name="PamProc._collect_candidates", phase="pam")(PamProc._collect_candidates)(
            ghost
        )
        top_memories = trace_knoxel_flow(name="PamProc._rank_candidates", phase="pam")(PamProc._rank_candidates)(
            ghost,
            query_embedding,
            candidates,
            5,
        )
        trace_knoxel_flow(name="PamProc._emit_recall_features", phase="pam")(PamProc._emit_recall_features)(
            ghost,
            top_memories,
            candidates,
        )

    @staticmethod
    def retrieve_context(ghost: GhostProtocol, query: str) -> List[Feature]:
        """Deterministic retrieval helper for callers that need recall features."""
        if not query:
            return []
        try:
            emb = ghost.llm.get_embedding(query)
        except Exception:
            return []

        ranked = PamProc._rank_memories(ghost, emb, limit=3)
        out: List[Feature] = []
        for s in ranked:
            mem = s.memory
            out.append(
                Feature(
                    content=getattr(mem, "content", ""),
                    feature_type=FeatureType.MemoryRecall,
                    source="PAM_Recall",
                    interlocus=1,
                    causal=False,
                    embedding=getattr(mem, "embedding", []),
                )
            )
        return out

    @staticmethod
    def store_memory(
        ghost: GhostProtocol,
        content: str,
        embedding: List[float],
        memory_type: FeatureType = FeatureType.MemoryRecall,
        tags: Optional[List[str]] = None,
    ) -> None:
        """
        Store a new memory in PAM as a topical memory cluster.
        Includes a lightweight mental-state snapshot for future state-proximity recall.
        """
        logger.info("PAM: Storing memory: '%s...'", content[:50])

        if not embedding and hasattr(ghost, "llm"):
            embedding = ghost.llm.get_embedding(content)

        state_sig = PamProc._get_current_state_signature(ghost)
        metadata = {
            "stored_by": "PamProc",
            "memory_type": str(memory_type),
            "tags": tags or [],
            "mental_state_signature": state_sig,
            "valence_hint": state_sig.get("valence", 0.0),
        }

        mem = MemoryClusterKnoxel(
            content=content,
            embedding=embedding,
            cluster_type=ClusterType.Topical,
            level=100,
            tick_id=ghost.current_tick_id,
            metadata=metadata,
        )
        ghost.add_knoxel(mem)

