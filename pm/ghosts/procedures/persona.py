import logging
import re
from typing import Optional

from pm.data_structures import CauseEffectKnoxel, DeclarativeFactKnoxel, Feature, FeatureType
from pm.ghosts.knoxel_trace import trace_substep
from pm.ghosts.procedures.base import BaseProc, GhostProtocol

logger = logging.getLogger(__name__)


def _norm(text: str) -> str:
    t = str(text or "").lower().strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^a-z0-9\s:_-]", "", t)
    return t


def _excerpt(text: str, max_len: int = 180) -> str:
    t = str(text or "").replace("\n", " ").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 3].rstrip() + "..."


def _latest_reply_feature(ghost: GhostProtocol) -> Optional[Feature]:
    for f in reversed(list(getattr(ghost, "all_features", []) or [])):
        if getattr(f, "tick_id", None) != getattr(ghost, "current_tick_id", None):
            break
        if getattr(f, "feature_type", None) == FeatureType.Dialogue:
            src = str(getattr(f, "source", "") or "")
            if src == str(getattr(getattr(ghost, "config", None), "companion_name", "") or "assistant"):
                return f
    return None


class PersonaPersistenceProc(BaseProc):
    """
    Persist emergent action/reply quirks as persona memories once they become causal.
    """

    @staticmethod
    def run(ghost: GhostProtocol) -> None:
        behavior = getattr(ghost, "selected_action_schema", None)
        reply = str(getattr(ghost, "simulated_reply", "") or "").strip()
        if behavior is None or not reply:
            return

        reply_feature = trace_substep(
            ghost, "PersonaPersistenceProc.latest_reply_feature", "persona", _latest_reply_feature
        )
        blueprint = dict(getattr(ghost, "reply_blueprint_last", {}) or {})
        bname = str(blueprint.get("name", "") or "none")
        winner_lens = str((getattr(ghost, "ego_decision_last", {}) or {}).get("winner_lens", "") or "none")
        sim_winner = str((getattr(ghost, "simulation_bundle_last", {}) or {}).get("winner_lens", "") or "none")
        stimulus = str(getattr(getattr(ghost, "primary_stimulus", None), "content", "") or "")
        broadcast = str(getattr(getattr(ghost, "conscious_broadcast", None), "content", "") or "")
        action_desc = str(getattr(behavior, "action_description", "") or "")

        signature = _norm(f"{action_desc}|{bname}|{winner_lens}|{sim_winner}")
        seen = set(getattr(ghost, "persona_pattern_signatures", set()) or set())
        if signature in seen:
            return
        seen.add(signature)
        ghost.persona_pattern_signatures = seen

        situation = _excerpt(
            f"When the user context is '{stimulus or broadcast}' and workspace focus is '{broadcast}'", 220
        )
        cause = _excerpt(
            f"I select action='{action_desc}', blueprint='{bname}', arbitration='{winner_lens}', simulation='{sim_winner}'",
            220,
        )
        effect = _excerpt(
            f"I produce reply style: '{reply}'.",
            220,
        )

        ce = CauseEffectKnoxel(
            situation=situation,
            cause=cause,
            effect=effect,
            category="persona_action_quirk",
            metadata={
                "persona_signal": True,
                "action_description": action_desc,
                "reply_blueprint": bname,
                "winner_lens": winner_lens,
                "simulation_winner": sim_winner,
                "reply_feature_id": int(getattr(reply_feature, "id", -1) or -1),
            },
        )
        ghost.add_knoxel(ce)

        fact = DeclarativeFactKnoxel(
            content=_excerpt(
                f"Persona tendency: under context '{stimulus or broadcast}', prefer '{bname}' pattern for '{action_desc}'.",
                220,
            ),
            reason="Derived from causal action->reply execution pattern in current tick.",
            category=["persona_quirks", "behavior_action_selection"],
            importance=0.70,
            time_dependent=0.45,
            metadata={
                "persona_signal": True,
                "source_ce_id": int(getattr(ce, "id", -1) or -1),
                "reply_excerpt": _excerpt(reply, 140),
            },
        )
        ghost.add_knoxel(fact)
        logger.info("Persona persistence: stored quirk CE=%s FACT=%s", ce.id, fact.id)
