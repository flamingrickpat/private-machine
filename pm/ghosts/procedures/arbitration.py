import logging
import random
from typing import Dict, Optional, Tuple

from pm.csm.csm import CSMItem
from pm.data_structures import Feature, FeatureType
from pm.ghosts.knoxel_trace import trace_substep
from pm.ghosts.procedures.base import BaseProc, GhostProtocol
from pm.ghosts.schemas import EgoArbitrationDecision, SimBundle, SimLensOutcome

logger = logging.getLogger(__name__)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _safe_excerpt(text: str, max_len: int = 120) -> str:
    t = (text or "").replace("\n", " ").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 3].rstrip() + "..."


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, v) for v in weights.values())
    if total <= 1e-8:
        return {"world": 1.0 / 3.0, "self": 1.0 / 3.0, "meta": 1.0 / 3.0}
    return {k: max(0.0, v) / total for k, v in weights.items()}


def _domain_weights_from_state(ghost: GhostProtocol) -> Dict[str, float]:
    weights = {"world": 0.34, "self": 0.33, "meta": 0.33}

    ms = getattr(getattr(ghost, "current_state", None), "latent_mental_state", None)
    if ms is None:
        return _normalize_weights(weights)

    cog = getattr(ms, "state_cognition", None)
    needs = getattr(ms, "state_needs", None)

    interlocus = float(getattr(cog, "interlocus", 0.0) or 0.0) if cog else 0.0
    ego = float(getattr(cog, "ego_strength", 0.5) or 0.5) if cog else 0.5
    aperture = abs(float(getattr(cog, "mental_aperture", 0.0) or 0.0)) if cog else 0.0

    energy = 0.5
    if needs:
        energy = (
            float(getattr(needs, "energy_stability", 0.5))
            + float(getattr(needs, "processing_power", 0.5))
        ) * 0.5
    energy = _clamp(energy, 0.0, 1.0)
    low_energy = 1.0 - energy

    weights["world"] += max(0.0, interlocus) * 0.22
    weights["self"] += max(0.0, -interlocus) * 0.16 + aperture * 0.05
    weights["meta"] += ego * 0.12 + aperture * 0.05

    # Low-energy behavior drifts inward / guardrail-heavy.
    weights["self"] += low_energy * 0.08
    weights["meta"] += low_energy * 0.05
    weights["world"] -= low_energy * 0.06

    return _normalize_weights(weights)


def _stochasticity_from_state(ghost: GhostProtocol) -> float:
    ms = getattr(getattr(ghost, "current_state", None), "latent_mental_state", None)
    if ms is None:
        return 0.03

    needs = getattr(ms, "state_needs", None)
    if needs is None:
        return 0.03

    energy = (
        float(getattr(needs, "energy_stability", 0.5))
        + float(getattr(needs, "processing_power", 0.5))
    ) * 0.5
    energy = _clamp(energy, 0.0, 1.0)
    return _clamp(0.01 + (1.0 - energy) * 0.12, 0.01, 0.14)


def _score_lens(
    outcome: SimLensOutcome,
    lens_weight: float,
    stochasticity: float,
    rng: random.Random,
) -> float:
    polarity_bias = {"positive": 0.08, "neutral": 0.0, "negative": -0.09}
    att_count = min(6, len(outcome.attractors))
    rep_count = min(6, len(outcome.repellers))
    evidence_count = min(4, len(outcome.evidence))

    score = (
        0.55 * float(outcome.utility)
        + 0.25 * float(outcome.confidence)
        + 0.20 * float(lens_weight)
        + polarity_bias.get((outcome.polarity or "").lower(), 0.0)
        + 0.012 * att_count
        - 0.018 * rep_count
        + 0.01 * evidence_count
    )
    score += rng.uniform(-stochasticity, stochasticity)
    return _clamp(score, 0.0, 1.0)


def _build_directive(
    winner: SimLensOutcome,
    runner_up: Optional[SimLensOutcome],
    dissonance: float,
) -> Tuple[str, str]:
    guard = ",".join(winner.repellers[:2]) if winner.repellers else "none"
    attract = ",".join(winner.attractors[:2]) if winner.attractors else "none"
    winner_summary = _safe_excerpt(winner.summary, max_len=100)

    if runner_up is None:
        directive = (
            f"Prioritize {winner.lens} lens: {winner_summary}. "
            f"Push toward [{attract}] and guard against [{guard}]."
        )
        rationale = f"{winner.lens} has dominant expected utility with low alternative support."
        return directive, rationale

    runner_anchor = ",".join(runner_up.attractors[:1]) if runner_up.attractors else runner_up.lens
    directive = (
        f"Prioritize {winner.lens}: {winner_summary}. "
        f"Guard [{guard}], anchor with {runner_up.lens}:{runner_anchor}, "
        f"and keep dissonance budget at {dissonance:.2f}."
    )
    rationale = (
        f"{winner.lens} won on utility/confidence blend; "
        f"{runner_up.lens} remains a stabilizing fallback."
    )
    return directive, rationale


def arbitrate_sim_bundle(
    bundle: SimBundle,
    ghost: Optional[GhostProtocol] = None,
    rng: Optional[random.Random] = None,
) -> EgoArbitrationDecision:
    local_rng = rng or random.Random(bundle.tick)
    weights = _domain_weights_from_state(ghost) if ghost is not None else _normalize_weights(
        {"world": 0.34, "self": 0.33, "meta": 0.33}
    )
    stochasticity = _stochasticity_from_state(ghost) if ghost is not None else 0.03

    scored: Dict[str, float] = {}
    lens_map = {o.lens: o for o in bundle.outcomes}
    for outcome in bundle.outcomes:
        scored[outcome.lens] = _score_lens(
            outcome=outcome,
            lens_weight=weights.get(outcome.lens, 1.0 / 3.0),
            stochasticity=stochasticity,
            rng=local_rng,
        )

    ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)
    if not ranked:
        return EgoArbitrationDecision(
            tick=bundle.tick,
            winner_lens="",
            runner_up_lens="",
            winner_score=0.0,
            runner_up_score=0.0,
            dissonance=0.0,
            stochasticity=stochasticity,
            domain_weights=weights,
            lens_scores=scored,
            directive="No simulation evidence available; proceed conservatively.",
            rationale="Simulation bundle was empty.",
        )

    winner_lens, winner_score = ranked[0]
    runner_up_lens, runner_up_score = ranked[1] if len(ranked) > 1 else ("", 0.0)
    dissonance = _clamp(winner_score - runner_up_score, 0.0, 1.0)

    winner = lens_map.get(winner_lens)
    runner = lens_map.get(runner_up_lens) if runner_up_lens else None
    if winner is None:
        directive = "No winner lens summary available; proceed with conservative response."
        rationale = "Winner lens object missing after scoring."
    else:
        directive, rationale = _build_directive(winner, runner, dissonance)

    return EgoArbitrationDecision(
        tick=bundle.tick,
        winner_lens=winner_lens,
        runner_up_lens=runner_up_lens,
        winner_score=_clamp(float(winner_score), 0.0, 1.0),
        runner_up_score=_clamp(float(runner_up_score), 0.0, 1.0),
        dissonance=dissonance,
        stochasticity=stochasticity,
        domain_weights=weights,
        lens_scores=scored,
        directive=directive,
        rationale=rationale,
    )


def _apply_csm_steering(
    ghost: GhostProtocol,
    bundle: SimBundle,
    decision: EgoArbitrationDecision,
) -> Dict[str, int]:
    manager = getattr(ghost, "csm_manager", None)
    if manager is None:
        return {"boosted": 0, "nerfed": 0}

    winner = next((o for o in bundle.outcomes if o.lens == decision.winner_lens), None)
    runner = next((o for o in bundle.outcomes if o.lens == decision.runner_up_lens), None)
    if winner is None:
        return {"boosted": 0, "nerfed": 0}

    attractors = [x.lower() for x in winner.attractors[:4]]
    if runner is not None:
        attractors.extend(x.lower() for x in runner.attractors[:2])
    repellers = [x.lower() for x in winner.repellers[:4]]

    boosted = 0
    nerfed = 0
    for item in manager.items():
        k = ghost.get_knoxel_by_id(item.knoxel_id)
        if k is None:
            continue
        content = (getattr(k, "content", "") or "").lower()
        if not content:
            continue

        delta = 0.0
        if any(a and a in content for a in attractors):
            delta += 0.10
        if any(r and r in content for r in repellers):
            delta -= 0.12

        if abs(delta) < 1e-8:
            continue
        item.activation = _clamp(item.activation + delta, 0.05, 1.0)
        item.peak_activation = max(item.peak_activation, item.activation)
        item.last_tick = ghost.current_tick_id
        if delta > 0:
            boosted += 1
        else:
            nerfed += 1

    return {"boosted": boosted, "nerfed": nerfed}


class ArbitrationProc(BaseProc):
    """
    Ego arbitration layer over simulation bundle.
    Produces winner/runner-up, conflict score, and steering directive used by
    workspace/action phases. Also applies gentle CSM boost/nerf from winner lens.
    """

    @staticmethod
    def run(ghost: GhostProtocol) -> None:
        logger.info("Arbitration: resolving simulation lens conflicts...")

        raw_bundle = getattr(ghost, "simulation_bundle_last_model", None)
        if raw_bundle is None:
            raw_bundle = getattr(ghost, "simulation_bundle_last", None)
        if raw_bundle is None:
            logger.info("Arbitration: no simulation bundle available.")
            return

        try:
            bundle = raw_bundle if isinstance(raw_bundle, SimBundle) else SimBundle.model_validate(raw_bundle)
        except Exception:
            logger.exception("Arbitration: invalid simulation bundle; skipping.")
            return

        decision = trace_substep(
            ghost,
            "ArbitrationProc.arbitrate_sim_bundle",
            "arbitration",
            lambda g: arbitrate_sim_bundle(bundle=bundle, ghost=ghost, rng=random.Random(ghost.current_tick_id)),
        )
        steering_effects = trace_substep(
            ghost, "ArbitrationProc._apply_csm_steering", "arbitration", _apply_csm_steering, bundle, decision
        )

        ghost.ego_decision_last_model = decision
        ghost.ego_decision_last = decision.model_dump()
        ghost.ego_directive = decision.directive
        if not hasattr(ghost, "ego_decision_history"):
            ghost.ego_decision_history = []
        ghost.ego_decision_history.append(decision.model_dump())
        if len(ghost.ego_decision_history) > 100:
            ghost.ego_decision_history = ghost.ego_decision_history[-100:]

        text = (
            f"Ego arbitration winner={decision.winner_lens} score={decision.winner_score:.2f} "
            f"runner={decision.runner_up_lens or '-'} dissonance={decision.dissonance:.2f}. "
            f"Directive: {decision.directive}"
        )
        feat = Feature(
            content=text,
            feature_type=FeatureType.MetaInsight,
            source="ArbitrationProc",
            interlocus=-1,
            causal=False,
            metadata={
                "winner_lens": decision.winner_lens,
                "runner_up_lens": decision.runner_up_lens,
                "dissonance": decision.dissonance,
                "domain_weights": decision.domain_weights,
                "lens_scores": decision.lens_scores,
                "stochasticity": decision.stochasticity,
                "steering_effects": steering_effects,
            },
        )
        ghost.add_knoxel(feat)
        if hasattr(ghost, "csm_manager") and ghost.csm_manager is not None:
            ghost.csm_manager.add_or_boost(
                CSMItem(
                    knoxel_id=feat.id,
                    first_tick=ghost.current_tick_id,
                    last_tick=ghost.current_tick_id,
                    activation=0.72,
                    peak_activation=0.72,
                )
            )

        logger.info(
            "Arbitration: winner=%s runner=%s dissonance=%.2f csm(boost=%d nerf=%d)",
            decision.winner_lens,
            decision.runner_up_lens or "-",
            decision.dissonance,
            steering_effects["boosted"],
            steering_effects["nerfed"],
        )
