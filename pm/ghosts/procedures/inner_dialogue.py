import logging
from typing import Dict

from pm.data_structures import Feature, FeatureType
from pm.ghosts.procedures.base import BaseProc, GhostProtocol
from pm.thoughts import InnerDialogueEngine

logger = logging.getLogger(__name__)


class InnerDialogueProc(BaseProc):
    """
    Ghost-aware inner dialogue procedure.
    Generates internal monologue turns and persists them as knoxels.
    """

    @staticmethod
    def run(ghost: GhostProtocol) -> Dict:
        if not hasattr(ghost, "_inner_dialogue_engine") or ghost._inner_dialogue_engine is None:
            ghost._inner_dialogue_engine = InnerDialogueEngine(ghost)

        context = ""
        if hasattr(ghost, "conscious_broadcast") and ghost.conscious_broadcast:
            context = str(getattr(ghost.conscious_broadcast, "content", "") or "")
        if hasattr(ghost, "csm_manager") and ghost.csm_manager is not None:
            gist = str(getattr(ghost.csm_manager.state, "gist", "") or "")
            if gist:
                context = (context + "\n" + gist).strip()

        last_user = ""
        for f in reversed(list(getattr(ghost, "all_features", []) or [])):
            if getattr(f, "feature_type", None) == FeatureType.Dialogue and str(getattr(f, "source", "")).lower() in {"user", str(getattr(getattr(ghost, "config", None), "user_name", "")).lower()}:
                last_user = str(getattr(f, "content", "") or "")
                break

        packet = ghost._inner_dialogue_engine.generate_inner_dialogue_packet(
            context=context,
            last_user_message=last_user,
        )

        inner_text = str(packet.get("inner_dialogue", "") or "").strip()
        if inner_text:
            feat = Feature(
                content=inner_text,
                feature_type=FeatureType.Thought,
                source="InnerDialogueProc",
                interlocus=-1,
                causal=False,
                metadata={
                    "topic": packet.get("topic", ""),
                    "candidate_openers": packet.get("candidate_openers", []),
                    "proactive_opener": packet.get("proactive_opener", ""),
                },
            )
            ghost.add_knoxel(feat)

        ghost.inner_dialogue_last = packet
        if not hasattr(ghost, "inner_dialogue_history"):
            ghost.inner_dialogue_history = []
        ghost.inner_dialogue_history.append(packet)
        if len(ghost.inner_dialogue_history) > 100:
            ghost.inner_dialogue_history = ghost.inner_dialogue_history[-100:]

        return packet
