import logging
from time import perf_counter
from typing import Any, Dict

from pm.ghosts.knoxel_trace import trace_substep
from pm.ghosts.procedures.base import BaseProc, GhostProtocol

logger = logging.getLogger(__name__)


def _count_memory_artifacts(ghost: GhostProtocol) -> Dict[str, int]:
    return {
        "knoxels": len(dict(getattr(ghost, "all_knoxels", {}) or {})),
        "features": len(list(getattr(ghost, "all_features", []) or [])),
        "facts": len(list(getattr(ghost, "all_declarative_facts", []) or [])),
        "narratives": len(list(getattr(ghost, "all_narratives", []) or [])),
        "episodic_memories": len(list(getattr(ghost, "all_episodic_memories", []) or [])),
        "intentions": len(list(getattr(ghost, "all_intentions", []) or [])),
        "graph_nodes": len(list(getattr(ghost, "all_graph_nodes", []) or [])),
        "graph_edges": len(list(getattr(ghost, "all_graph_edges", []) or [])),
        "concepts": len(list(getattr(ghost, "all_concepts", []) or [])),
    }


class MemoryReconsolidationProc(BaseProc):
    """
    Wrapper around the legacy memory consolidator.

    Responsibilities:
    - Call `ghost.memory_consolidator.consolidate_memory_if_needed()`.
    - Persist run telemetry + before/after deltas on the ghost.
    - Invalidate context caches that depend on historical memory retrieval.
    """

    @staticmethod
    def run(ghost: GhostProtocol) -> Dict[str, Any]:
        consolidator = getattr(ghost, "memory_consolidator", None)
        if consolidator is None:
            payload = {
                "tick": int(getattr(ghost, "current_tick_id", 0) or 0),
                "ok": False,
                "skipped": True,
                "reason": "missing_memory_consolidator",
                "before": _count_memory_artifacts(ghost),
                "after": _count_memory_artifacts(ghost),
                "delta": {},
                "duration_ms": 0.0,
            }
            ghost.memory_recon_last = payload
            return payload

        before = _count_memory_artifacts(ghost)
        started = perf_counter()
        ok = True
        error = ""
        try:
            trace_substep(
                ghost,
                "MemoryReconsolidationProc.consolidate_memory_if_needed",
                "memory_recon",
                lambda g: consolidator.consolidate_memory_if_needed(),
            )
        except Exception as e:
            ok = False
            error = f"{type(e).__name__}: {e}"
            logger.exception("Memory reconsolidation failed.")

        duration_ms = round((perf_counter() - started) * 1000.0, 3)
        after = _count_memory_artifacts(ghost)
        delta = {k: int(after.get(k, 0) - before.get(k, 0)) for k in set(before.keys()) | set(after.keys())}

        # Consolidation changes long-term memory landscape; invalidate stale retrieval caches.
        ghost.story_context_embedding_cache = {}
        ghost.agent_context_embedding_cache = {}

        payload = {
            "tick": int(getattr(ghost, "current_tick_id", 0) or 0),
            "ok": bool(ok),
            "skipped": False,
            "reason": "" if ok else error,
            "before": before,
            "after": after,
            "delta": delta,
            "duration_ms": duration_ms,
        }
        ghost.memory_recon_last = payload

        hist = list(getattr(ghost, "memory_recon_history", []) or [])
        hist.append(payload)
        if len(hist) > 100:
            hist = hist[-100:]
        ghost.memory_recon_history = hist

        if ok:
            logger.info(
                "Memory reconsolidation done: dt=%sms delta(mem=%s facts=%s narratives=%s)",
                duration_ms,
                delta.get("episodic_memories", 0),
                delta.get("facts", 0),
                delta.get("narratives", 0),
            )
        return payload
