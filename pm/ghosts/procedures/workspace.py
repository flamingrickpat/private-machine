import logging

from pm.csm.csm import CSMItem
from pm.data_structures import Feature, FeatureType
from pm.ghosts.knoxel_trace import trace_substep
from pm.ghosts.procedures.base import BaseProc, GhostProtocol

logger = logging.getLogger(__name__)


class WorkspaceProc(BaseProc):
    """
    Global Workspace / broadcast step.
    Uses attention coalition winner as conscious broadcast while preserving
    coalition context for downstream reasoning.
    """

    @staticmethod
    def run(ghost: GhostProtocol) -> None:
        logger.info("GWT: broadcasting conscious content...")

        if not hasattr(ghost, "current_coalition") or not ghost.current_coalition:
            logger.info("GWT: No coalition to broadcast.")
            return

        winner_id, score = ghost.current_coalition[0]

        try:
            winner_feature = trace_substep(
                ghost, "WorkspaceProc.get_winner_feature", "workspace", lambda g: ghost.get_knoxel_by_id(winner_id)
            )
            if not winner_feature:
                return

            ghost.conscious_broadcast = winner_feature

            # Keep coalition members as explicit conscious candidates.
            candidates = trace_substep(
                ghost,
                "WorkspaceProc.collect_candidates",
                "workspace",
                lambda g: [ghost.get_knoxel_by_id(kid) for kid, _ in ghost.current_coalition if ghost.get_knoxel_by_id(kid) is not None],
            )
            ghost.conscious_candidates = candidates

            logger.info("GWT: BROADCASTING: %s... (Score: %.2f)", winner_feature.content[:50], score)

            if getattr(ghost, "current_state", None):
                ghost.current_state.rating += 1

            if hasattr(ghost, "csm_manager") and ghost.csm_manager is not None:
                ghost.csm_manager.add_or_boost(
                    CSMItem(
                        knoxel_id=winner_feature.id,
                        first_tick=ghost.current_tick_id,
                        last_tick=ghost.current_tick_id,
                        activation=1.0,
                        peak_activation=1.0,
                    )
                )

            # Inject arbitration steering into workspace as an explicit signal.
            ego_directive = str(getattr(ghost, "ego_directive", "") or "").strip()
            if ego_directive:
                steer = Feature(
                    content=f"Workspace steer: {ego_directive}",
                    feature_type=FeatureType.MetaInsight,
                    source="WorkspaceArbitration",
                    interlocus=-1,
                    causal=False,
                    metadata={
                        "winner_lens": (getattr(ghost, "ego_decision_last", {}) or {}).get("winner_lens", ""),
                        "dissonance": float((getattr(ghost, "ego_decision_last", {}) or {}).get("dissonance", 0.0) or 0.0),
                    },
                )
                ghost.add_knoxel(steer)
                if hasattr(ghost, "csm_manager") and ghost.csm_manager is not None:
                    ghost.csm_manager.add_or_boost(
                        CSMItem(
                            knoxel_id=steer.id,
                            first_tick=ghost.current_tick_id,
                            last_tick=ghost.current_tick_id,
                            activation=0.68,
                            peak_activation=0.68,
                        )
                    )

            thought_directive = str(getattr(ghost, "thought_blueprint_directive", "") or "").strip()
            if thought_directive:
                thought_steer = Feature(
                    content=f"Thought blueprint steer: {thought_directive}",
                    feature_type=FeatureType.MetaInsight,
                    source="WorkspaceThoughtBlueprint",
                    interlocus=-1,
                    causal=False,
                    metadata={
                        "final_thought": (getattr(ghost, "thought_blueprint_last", {}) or {}).get("final_thought", ""),
                        "path": (getattr(ghost, "thought_blueprint_last", {}) or {}).get("path", []),
                        "action_hints": list(getattr(ghost, "thought_blueprint_action_hints", []) or []),
                    },
                )
                ghost.add_knoxel(thought_steer)
                if hasattr(ghost, "csm_manager") and ghost.csm_manager is not None:
                    ghost.csm_manager.add_or_boost(
                        CSMItem(
                            knoxel_id=thought_steer.id,
                            first_tick=ghost.current_tick_id,
                            last_tick=ghost.current_tick_id,
                            activation=0.66,
                            peak_activation=0.66,
                        )
                    )

            # Optional high-gain introspective interjection signal.
            gain = float(getattr(ghost, "workspace_gain", 0.0) or 0.0)
            last_tick = int(getattr(ghost, "last_introspection_tick", -9999))
            if gain >= 0.80 and (ghost.current_tick_id - last_tick) >= 3:
                text = (
                    f"Internal gate tension is high (gain={gain:.2f}); narrowing focus "
                    f"around '{winner_feature.content[:80]}'."
                )
                insight = Feature(
                    content=text,
                    feature_type=FeatureType.MetaInsight,
                    source="WorkspaceProc",
                    interlocus=-1,
                    causal=False,
                    metadata={
                        "workspace_gain": gain,
                        "winner_id": winner_feature.id,
                        "coalition_size": len(ghost.current_coalition),
                    },
                )
                ghost.add_knoxel(insight)
                ghost.last_introspection_tick = ghost.current_tick_id

            if hasattr(ghost, "broadcast_history"):
                ghost.broadcast_history.append(winner_id)
                if len(ghost.broadcast_history) > 20:
                    ghost.broadcast_history.pop(0)

        except Exception as e:
            logger.error("GWT: Error broadcasting: %s", e)
