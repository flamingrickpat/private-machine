import logging

from pm.csm.csm import CSMItem, CSMManager, CSMState
from pm.ghosts.knoxel_trace import trace_substep
from pm.ghosts.procedures.base import BaseProc, GhostProtocol

logger = logging.getLogger(__name__)


class CsmProc(BaseProc):
    """
    Current Situational Model (CSM) Procedure.
    Maintains the short-term situational workspace with deterministic
    decay, activation spread, pruning, and gist synthesis.
    """

    @staticmethod
    def _get_or_init_manager(ghost: GhostProtocol) -> CSMManager:
        if not hasattr(ghost, "csm_manager") or ghost.csm_manager is None:
            seed_state = None
            if getattr(ghost, "current_state", None) and getattr(ghost.current_state, "csm_state", None):
                seed_state = ghost.current_state.csm_state
            ghost.csm_manager = CSMManager(ghost, state=seed_state or CSMState())
        return ghost.csm_manager

    @staticmethod
    def _ingest_input_knoxels(ghost: GhostProtocol, manager: CSMManager) -> None:
        if hasattr(ghost, "input_knoxels") and ghost.input_knoxels:
            for f in ghost.input_knoxels:
                item = CSMItem(
                    knoxel_id=f.id,
                    first_tick=ghost.current_tick_id,
                    last_tick=ghost.current_tick_id,
                    activation=1.0,
                    peak_activation=1.0,
                )
                manager.add_or_boost(item)
            ghost.input_knoxels = []

    @staticmethod
    def _persist_removed_significant(ghost: GhostProtocol, removed: list[CSMItem]) -> None:
        if not removed:
            return
        from pm.ghosts.procedures.pam import PamProc

        for item in removed:
            if item.peak_activation < 0.8 and item.ticks_in_csm < 15:
                continue
            knoxel = ghost.get_knoxel_by_id(item.knoxel_id)
            if not knoxel or not getattr(knoxel, "content", None):
                continue
            try:
                PamProc.store_memory(
                    ghost,
                    content=knoxel.content,
                    embedding=getattr(knoxel, "embedding", None) or [],
                    tags=["csm_object_permanence"],
                )
            except Exception:
                logger.exception("CSM: Failed to store object-permanence memory for knoxel %s", item.knoxel_id)

    @staticmethod
    def run(ghost: GhostProtocol) -> None:
        logger.info("CSM: Updating situational model...")

        manager: CSMManager = trace_substep(
            ghost, "CsmProc._get_or_init_manager", "csm", CsmProc._get_or_init_manager
        )

        trace_substep(ghost, "CsmProc.decay_step", "csm", lambda g: manager.decay_step(decay_factor=0.9))
        trace_substep(ghost, "CsmProc._ingest_input_knoxels", "csm", CsmProc._ingest_input_knoxels, manager)
        trace_substep(
            ghost,
            "CsmProc.spread_activation",
            "csm",
            lambda g: manager.spread_activation(
            min_source_activation=0.65,
            similarity_threshold=0.55,
            spread_factor=0.14,
            max_neighbors=3,
            ),
        )
        removed = trace_substep(
            ghost,
            "CsmProc.prune",
            "csm",
            lambda g: manager.prune(
            current_tick=ghost.current_tick_id,
            min_activation=0.12,
            max_idle_ticks=72,
            permanent_max_idle_ticks=240,
            ),
        )
        trace_substep(
            ghost,
            "CsmProc._persist_removed_significant",
            "csm",
            CsmProc._persist_removed_significant,
            removed or [],
        )
        gist = trace_substep(
            ghost, "CsmProc.build_gist", "csm", lambda g: manager.build_gist(max_items=4, max_chars_per_item=120)
        )
        logger.debug("CSM Gist: %s", gist)

