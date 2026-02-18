import logging
import math
import random
from typing import Dict, List

from pm.data_structures import Feature, FeatureType
from pm.ghosts.llm_contracts import call_tool_with_contract
from pm.ghosts.knoxel_trace import trace_substep
from pm.ghosts.procedures.base import BaseProc, GhostProtocol
from pm.ghosts.prompts import (
    ATTENTION_MODULATION_EXAMPLE_TURNS,
    ATTENTION_MODULATION_SYSTEM,
    ATTENTION_MODULATION_USER,
)
from pm.ghosts.schemas import AttentionModulation
from pm.llm.llm_common import LlmPreset

logger = logging.getLogger(__name__)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _norm_unit_from_signed(x: float) -> float:
    # map [-1,1] -> [0,1]
    return _clamp((x + 1.0) * 0.5, 0.0, 1.0)


def _compute_workspace_gain(ghost: GhostProtocol) -> float:
    """
    Gain proxy for discrete vs continuous conscious control.
    High gain => narrower, more urgent focus.
    """
    arousal = 0.5
    ego_strength = 0.5
    interlocus = 0.0
    mental_aperture = 0.0
    energy = 0.5

    if getattr(ghost, "current_state", None) and ghost.current_state.latent_mental_state:
        ms = ghost.current_state.latent_mental_state
        core = getattr(ms, "state_core", None)
        cog = getattr(ms, "state_cognition", None)
        needs = getattr(ms, "state_needs", None)

        if core is not None:
            arousal = float(getattr(core, "arousal", arousal))
        if cog is not None:
            ego_strength = float(getattr(cog, "ego_strength", ego_strength))
            interlocus = float(getattr(cog, "interlocus", interlocus))
            mental_aperture = float(getattr(cog, "mental_aperture", mental_aperture))
            willpower = float(getattr(cog, "willpower", 0.0))
        else:
            willpower = 0.0

        if needs is not None:
            energy = (
                float(getattr(needs, "energy_stability", 0.5))
                + float(getattr(needs, "processing_power", 0.5))
            ) * 0.5
            # blend with willpower (signed axis)
            energy = (energy * 0.75) + (_norm_unit_from_signed(willpower) * 0.25)

    conflict = 0.55 * abs(interlocus) + 0.45 * abs(mental_aperture)
    low_energy_pressure = 1.0 - _clamp(energy, 0.0, 1.0)

    gain = (
        0.35 * _clamp(arousal, 0.0, 1.0)
        + 0.20 * _clamp(ego_strength, 0.0, 1.0)
        + 0.25 * _clamp(conflict, 0.0, 1.0)
        + 0.20 * low_energy_pressure
    )
    return _clamp(gain, 0.0, 1.0)


def _coalition_size_from_gain(gain: float) -> int:
    if gain >= 0.72:
        return 1
    if gain >= 0.42:
        return 2
    return 4


def _token_cap_from_gain(ghost: GhostProtocol, gain: float) -> int:
    base = 1024
    try:
        base = int(ghost.llm.get_max_tokens(LlmPreset.Default))
    except Exception:
        pass

    if gain >= 0.72:
        ratio = 0.22
    elif gain >= 0.42:
        ratio = 0.38
    else:
        ratio = 0.58

    return max(128, int(base * ratio))


class AttentionProc(BaseProc):
    """
    Attention procedure with gain-based control:
    - score active CSM features
    - apply inhibition-of-return
    - optionally apply LLM modulation
    - select coalition size from workspace gain
    - derive generation token cap from gain
    """

    @staticmethod
    def run(ghost: GhostProtocol) -> None:
        logger.info("Attention: scoring features...")

        if not hasattr(ghost, "csm_manager") or ghost.csm_manager is None:
            return

        csm_items = trace_substep(ghost, "AttentionProc.csm_items", "attention", lambda g: ghost.csm_manager.items())
        if not csm_items:
            return

        gain = trace_substep(ghost, "AttentionProc._compute_workspace_gain", "attention", _compute_workspace_gain)
        coalition_target = trace_substep(
            ghost, "AttentionProc._coalition_size_from_gain", "attention", lambda g: _coalition_size_from_gain(gain)
        )
        token_cap = trace_substep(
            ghost, "AttentionProc._token_cap_from_gain", "attention", lambda g: _token_cap_from_gain(ghost, gain)
        )
        ghost.workspace_gain = gain
        ghost.coalition_target_count = coalition_target
        ghost.generation_token_cap = token_cap

        scored_features: List[dict] = []
        history_window = 10
        inhibition_factor = 0.55
        recent_broadcasts = getattr(ghost, "broadcast_history", [])[-history_window:]

        for item in csm_items:
            knoxel = ghost.get_knoxel_by_id(item.knoxel_id)
            if not knoxel:
                continue

            salience = float(getattr(knoxel, "incentive_salience", 0.0) or 0.0)
            valence = float(getattr(knoxel, "affective_valence", 0.0) or 0.0)

            score = (item.activation * 0.50) + (salience * 0.30) + (abs(valence) * 0.20)
            if knoxel.id in recent_broadcasts:
                score *= inhibition_factor

            scored_features.append({"id": item.knoxel_id, "score": score, "knoxel": knoxel})

        if not scored_features:
            return

        temperature = _clamp(0.10 + (1.0 - gain) * 0.35, 0.05, 0.50)
        qmod = getattr(ghost, "qualia_attention_modulation", None) or {}
        if qmod:
            temperature = _clamp(temperature + float(qmod.get("temperature_delta", 0.0) or 0.0), 0.05, 1.0)
            for f in scored_features:
                content = (getattr(f["knoxel"], "content", "") or "").lower()
                if any(tag.lower() in content for tag in (qmod.get("boost_tags", []) or [])):
                    f["score"] *= 1.35
                if any(tag.lower() in content for tag in (qmod.get("suppress_tags", []) or [])):
                    f["score"] *= 0.70

        modulation = trace_substep(
            ghost, "AttentionProc._get_attention_modulation", "attention", AttentionProc._get_attention_modulation, scored_features
        )
        if modulation:
            temperature = _clamp(temperature + modulation.temperature_delta, 0.05, 1.0)
            for f in scored_features:
                content = (getattr(f["knoxel"], "content", "") or "").lower()
                if any(tag.lower() in content for tag in modulation.boost_tags):
                    f["score"] *= 1.5
                if any(tag.lower() in content for tag in modulation.suppress_tags):
                    f["score"] *= 0.5

        max_score = max(f["score"] for f in scored_features)
        for f in scored_features:
            f["weight"] = math.exp((f["score"] - max_score) / temperature)

        scored_features.sort(key=lambda x: x["weight"], reverse=True)

        # Winner-first stochastic pick, then deterministic fill by descending weight.
        total_weight = sum(f["weight"] for f in scored_features)
        r = random.random() * total_weight
        upto = 0.0
        winner = scored_features[0]
        for f in scored_features:
            upto += f["weight"]
            if upto >= r:
                winner = f
                break

        coalition_scores = [(winner["id"], winner["score"])]
        if len(coalition_scores) >= coalition_target:
            ghost.current_coalition = coalition_scores
            logger.info(
                "Attention: gain=%.2f coalition_target=%d token_cap=%d winner=%s",
                gain,
                coalition_target,
                token_cap,
                coalition_scores[0][0] if coalition_scores else None,
            )
            return

        for f in scored_features:
            if f["id"] == winner["id"]:
                continue
            if len(coalition_scores) >= coalition_target:
                break
            coalition_scores.append((f["id"], f["score"]))

        ghost.current_coalition = coalition_scores
        logger.info(
            "Attention: gain=%.2f coalition_target=%d token_cap=%d winner=%s",
            gain,
            coalition_target,
            token_cap,
            coalition_scores[0][0] if coalition_scores else None,
        )

    @staticmethod
    def _get_attention_modulation(ghost: GhostProtocol, scored_features: list) -> AttentionModulation | None:
        # Gate: only modulate every 5 ticks
        if ghost.current_tick_id % 5 != 0:
            return None

        csm_lines = []
        for f in scored_features[:10]:
            content = getattr(f["knoxel"], "content", "") or ""
            csm_lines.append(f"  [{f['score']:.2f}] {content[:100]}")
        csm_items_str = "\n".join(csm_lines) if csm_lines else "(empty)"

        arousal = 0.5
        valence = 0.0
        aperture = 0.5
        confidence = 0.5

        if getattr(ghost, "current_state", None) and ghost.current_state.latent_mental_state:
            core = ghost.current_state.latent_mental_state.state_core
            if core:
                arousal = core.arousal
                valence = core.valence
            cog = ghost.current_state.latent_mental_state.state_cognition
            if cog and hasattr(cog, "mental_aperture"):
                aperture = cog.mental_aperture

        if hasattr(ghost, "self_model") and ghost.self_model:
            confidence = ghost.self_model.confidence

        companion = ghost.config.companion_name
        sys_prompt = ATTENTION_MODULATION_SYSTEM.format(companion_name=companion)
        usr_prompt = ATTENTION_MODULATION_USER.format(
            csm_items=csm_items_str,
            arousal=arousal,
            valence=valence,
            aperture=aperture,
            confidence=confidence,
        )

        try:
            model, _ = call_tool_with_contract(
                ghost,
                phase="attention_modulation",
                schema=AttentionModulation,
                system_prompt=sys_prompt,
                user_prompt=usr_prompt,
                examples=ATTENTION_MODULATION_EXAMPLE_TURNS,
                preset=LlmPreset.Default,
                max_retries=1,
            )
            if model is not None:
                return model
        except Exception as e:
            logger.warning("Attention: Modulation LLM failed: %s", e)

        return None
