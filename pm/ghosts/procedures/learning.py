import logging
import re
from typing import Dict, List, Optional

from pm.csm.csm import CSMItem
from pm.data_structures import CauseEffectKnoxel, Feature, FeatureType
from pm.ghosts.knoxel_trace import trace_substep
from pm.ghosts.procedures.base import BaseProc, GhostProtocol
from pm.mental_state_vectors import StateNeurochemical

logger = logging.getLogger(__name__)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _norm(text: str) -> str:
    t = (text or "").lower().strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^a-z0-9\s]", "", t)
    return t


def _token_overlap(a: str, b: str) -> float:
    sa = set(_norm(a).split())
    sb = set(_norm(b).split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _safe_excerpt(text: str, max_len: int = 180) -> str:
    t = (text or "").replace("\n", " ").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 3].rstrip() + "..."


def _latest_action_feature(ghost: GhostProtocol) -> Optional[Feature]:
    for f in reversed(list(getattr(ghost, "all_features", []) or [])):
        if getattr(f, "tick_id", None) != ghost.current_tick_id:
            break
        if getattr(f, "source", "") == "ActionProc":
            return f
    return None


def _current_reality_text(ghost: GhostProtocol) -> str:
    bits: List[str] = []
    stim = getattr(ghost, "primary_stimulus", None)
    if stim is not None and getattr(stim, "content", None):
        bits.append(str(stim.content))
    b = getattr(ghost, "conscious_broadcast", None)
    if b is not None and getattr(b, "content", None):
        bits.append(str(b.content))
    if not bits:
        return ""
    return " | ".join(bits)


def _score_outcome(expected: str, reality: str, strictness: float = 0.0) -> Dict[str, float | str]:
    if not expected or not reality:
        return {"match_score": 0.0, "outcome": "unknown", "reward": 0.0}

    score = _token_overlap(expected, reality)
    strictness = _clamp(float(strictness or 0.0), 0.0, 1.0)
    matched_threshold = 0.62 + (0.10 * strictness)
    partial_threshold = 0.30 + (0.08 * strictness)

    if score >= matched_threshold:
        outcome = "matched"
        reward = 1.0 - (0.20 * strictness)
    elif score >= partial_threshold:
        outcome = "partial"
        reward = 0.25 - (0.15 * strictness)
    else:
        outcome = "missed"
        reward = -0.70 - (0.25 * strictness)
    return {"match_score": score, "outcome": outcome, "reward": reward}


def _store_policy_tendencies(ghost: GhostProtocol, pending: Dict, reward: float) -> None:
    tendencies = dict(getattr(ghost, "policy_tendencies", {}) or {})
    lens = str((pending.get("ego", {}) or {}).get("winner_lens", "") or "none")
    thought = str((pending.get("thought", {}) or {}).get("final_thought", "") or "none")
    key = f"lens:{lens}|thought:{thought}"

    entry = tendencies.get(key, {"reward_sum": 0.0, "count": 0, "avg_reward": 0.0})
    entry["reward_sum"] = float(entry.get("reward_sum", 0.0) or 0.0) + float(reward)
    entry["count"] = int(entry.get("count", 0) or 0) + 1
    entry["avg_reward"] = entry["reward_sum"] / max(1, entry["count"])
    tendencies[key] = entry
    ghost.policy_tendencies = tendencies


def _emit_expectation_outcome_feature(ghost: GhostProtocol, pending: Dict, reality: str, result: Dict[str, float | str]) -> None:
    text = (
        f"Expectation outcome: {result['outcome']} score={float(result['match_score']):.2f} | "
        f"expected='{_safe_excerpt(str(pending.get('expected_outcome', '')), 90)}' | "
        f"reality='{_safe_excerpt(reality, 90)}'"
    )
    feat = Feature(
        content=text,
        feature_type=FeatureType.ExpectationOutcome,
        source="ExpectationRealityProc",
        interlocus=1,
        causal=False,
        metadata={
            "pending_tick": pending.get("tick", -1),
            "match_score": float(result["match_score"]),
            "outcome": result["outcome"],
            "reward": float(result["reward"]),
            "expected": pending.get("expected_outcome", ""),
            "reality": _safe_excerpt(reality, 240),
        },
    )
    ghost.add_knoxel(feat)
    if hasattr(ghost, "csm_manager") and ghost.csm_manager is not None:
        act = 0.78 if result["outcome"] == "missed" else 0.60
        ghost.csm_manager.add_or_boost(
            CSMItem(
                knoxel_id=feat.id,
                first_tick=ghost.current_tick_id,
                last_tick=ghost.current_tick_id,
                activation=act,
                peak_activation=act,
            )
        )


def _maybe_emit_cause_effect(ghost: GhostProtocol, pending: Dict, reality: str, result: Dict[str, float | str]) -> None:
    if not reality:
        return

    cause = str(pending.get("causal_action", "") or pending.get("expected_outcome", "") or "action executed")
    effect = f"Observed outcome={result['outcome']} based on reality: {_safe_excerpt(reality, 150)}"
    ce = CauseEffectKnoxel(
        situation=str(pending.get("situation_signature", "") or ""),
        cause=_safe_excerpt(cause, 200),
        effect=_safe_excerpt(effect, 220),
        category="expectation_reality_delta",
        source_cluster_id=None,
    )
    ghost.add_knoxel(ce)

    # Optional richer extraction via analyzer; safe-fail.
    try:
        from pm.cause_effect import CauseEffectAnalyzer
        analyzer = CauseEffectAnalyzer(ghost.llm)
        block = f"Expected: {pending.get('expected_outcome','')}\nReality: {reality}\nOutcome: {result['outcome']}"
        extracted = analyzer.extract_cause_effect(block) or []
        for item in extracted[:3]:
            extra = CauseEffectKnoxel(
                situation=str(pending.get("situation_signature", "") or ""),
                cause=_safe_excerpt(item.cause, 200),
                effect=_safe_excerpt(item.effect, 220),
                category="extracted_expectation_reality",
                source_cluster_id=None,
            )
            ghost.add_knoxel(extra)
    except Exception:
        logger.debug("Expectation learning: rich cause-effect extraction unavailable.", exc_info=True)


class ExpectationRealityProc(BaseProc):
    """
    Task 10 expectation -> reality learning:
    - bind pending expectation from prior tick to current observed reality
    - compute delta/reward and update policy tendencies
    - persist deterministic cause-effect memory entries
    - store current tick expectation for next tick binding
    """

    @staticmethod
    def bind_previous_outcome(ghost: GhostProtocol) -> None:
        queue = list(getattr(ghost, "pending_expectation_queue", []) or [])
        if not queue:
            return

        reality = _current_reality_text(ghost)
        if not reality:
            return

        pending = queue.pop(0)
        profile = dict(pending.get("partner_expectation", {}) or getattr(ghost, "partner_expectation_profile", {}) or {})
        strictness = float(profile.get("disappointment_pressure", 0.0) or 0.0)
        result = trace_substep(
            ghost,
            "ExpectationRealityProc._score_outcome",
            "learn_bind",
            lambda g: _score_outcome(str(pending.get("expected_outcome", "")), reality, strictness=strictness),
        )
        reward = float(result["reward"])

        if hasattr(ghost, "state_deltas_buffer"):
            if reward > 0.2:
                ghost.state_deltas_buffer.append(StateNeurochemical(dopamine=0.14, cortisol=-0.05))
            elif reward < -0.2:
                ghost.state_deltas_buffer.append(StateNeurochemical(dopamine=0.0, cortisol=0.18))

        trace_substep(
            ghost, "ExpectationRealityProc._store_policy_tendencies", "learn_bind", _store_policy_tendencies, pending, reward
        )
        trace_substep(
            ghost,
            "ExpectationRealityProc._emit_expectation_outcome_feature",
            "learn_bind",
            _emit_expectation_outcome_feature,
            pending,
            reality,
            result,
        )
        trace_substep(
            ghost,
            "ExpectationRealityProc._maybe_emit_cause_effect",
            "learn_bind",
            _maybe_emit_cause_effect,
            pending,
            reality,
            result,
        )

        record = {
            "tick_bound": ghost.current_tick_id,
            "pending_tick": pending.get("tick", -1),
            "match_score": float(result["match_score"]),
            "outcome": result["outcome"],
            "reward": reward,
            "expected_outcome": pending.get("expected_outcome", ""),
            "reality_excerpt": _safe_excerpt(reality, 220),
        }
        ghost.last_expectation_outcome = record
        if not hasattr(ghost, "expectation_outcome_history"):
            ghost.expectation_outcome_history = []
        ghost.expectation_outcome_history.append(record)
        if len(ghost.expectation_outcome_history) > 100:
            ghost.expectation_outcome_history = ghost.expectation_outcome_history[-100:]

        ghost.pending_expectation_queue = queue
        logger.info(
            "Expectation learning: pending_tick=%s outcome=%s score=%.2f",
            pending.get("tick", -1),
            result["outcome"],
            float(result["match_score"]),
        )

    @staticmethod
    def store_current_expectation(ghost: GhostProtocol) -> None:
        queue = list(getattr(ghost, "pending_expectation_queue", []) or [])

        sim = getattr(ghost, "simulation_bundle_last", {}) or {}
        ego = getattr(ghost, "ego_decision_last", {}) or {}
        thought = getattr(ghost, "thought_blueprint_last", {}) or {}
        qualia = getattr(ghost, "qualia_reflection_last", {}) or {}
        action = _latest_action_feature(ghost)

        expected_outcome = ""
        if sim.get("outcomes"):
            winner = str(sim.get("winner_lens", "") or "")
            out = next((x for x in sim.get("outcomes", []) if x.get("lens", "") == winner), None)
            if out:
                expected_outcome = out.get("summary", "") or ""
            elif sim.get("outcomes"):
                expected_outcome = str(sim.get("outcomes", [])[0].get("summary", "") or "")
        if not expected_outcome:
            expected_outcome = str(getattr(ghost, "ego_directive", "") or "")
        if not expected_outcome and action is not None:
            expected_outcome = str((action.metadata or {}).get("sim_prediction", "") or "")

        if not expected_outcome:
            return

        broadcast = str(getattr(getattr(ghost, "conscious_broadcast", None), "content", "") or "")
        csm_gist = ""
        if hasattr(ghost, "csm_manager") and ghost.csm_manager is not None:
            csm_gist = getattr(ghost.csm_manager.state, "gist", "") or ""

        causal_action = ""
        if action is not None:
            causal_action = action.content
        elif broadcast:
            causal_action = broadcast

        pending = {
            "tick": ghost.current_tick_id,
            "situation_signature": _safe_excerpt(
                f"gist={csm_gist} | broadcast={broadcast} | thought={thought.get('final_thought','')} | "
                f"ego={ego.get('winner_lens','')} | emotion={qualia.get('dominant_emotion','')}",
                260,
            ),
            "expected_outcome": _safe_excerpt(expected_outcome, 240),
            "causal_action": _safe_excerpt(causal_action, 240),
            "simulation": sim,
            "ego": ego,
            "thought": thought,
            "qualia": {
                "dominant_emotion": qualia.get("dominant_emotion", ""),
                "tension": qualia.get("tension", 0.0),
            },
            "partner_expectation": dict(getattr(ghost, "partner_expectation_profile", {}) or {}),
        }
        queue.append(pending)
        if len(queue) > 20:
            queue = queue[-20:]
        ghost.pending_expectation_queue = queue
        ghost.last_expectation_pending = pending

        logger.info(
            "Expectation learning: stored pending expectation tick=%s len=%d",
            ghost.current_tick_id,
            len(queue),
        )
