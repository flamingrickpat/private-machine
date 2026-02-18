import logging
import re
from collections import Counter
from typing import Dict, List, Tuple

from pm.csm.csm import CSMItem
from pm.data_structures import Feature, FeatureType
from pm.ghosts.knoxel_trace import trace_substep
from pm.ghosts.persona_memory import collect_persona_signals
from pm.ghosts.agent_context import (
    AgentContextComposer,
    AgentContextConfig,
    AgentContextWeights,
    compute_agent_context_budget,
)
from pm.ghosts.capabilities import capability_context_text
from pm.ghosts.procedures.base import BaseProc, GhostProtocol
from pm.ghosts.schemas import SimBundle, SimLensOutcome, SimulationCodeletResult
from pm.ghosts.sims.simulation_codelets import SimulationCodeletRunner
from pm.ghosts.sims.simulators import SimMeta, SimSelf, SimWorld

logger = logging.getLogger(__name__)

_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "is", "are", "be", "as", "it",
    "this", "that", "by", "at", "from", "if", "then", "will", "would", "can", "could", "should", "around",
    "keep", "tone", "simworld", "simself", "simmeta", "prediction", "constraint",
}


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _parse_core_output(raw: str) -> Tuple[str, str, float]:
    """Return (summary, polarity, confidence)."""
    if not raw:
        return "", "neutral", 0.4

    text = raw.strip()
    polarity = "neutral"
    confidence = 0.6

    m = re.search(r"\(([^\)]*)\)\s*$", text)
    if m:
        sentiment = m.group(1).strip().lower()
        if "positive" in sentiment:
            polarity = "positive"
        elif "negative" in sentiment:
            polarity = "negative"
        elif "neutral" in sentiment:
            polarity = "neutral"
        text = text[: m.start()].strip()

    low = text.lower()
    if any(k in low for k in ["risk", "conflict", "harm", "unsafe", "violation", "worse"]):
        polarity = "negative"
    if any(k in low for k in ["help", "safe", "coherent", "improve", "calm", "better"]):
        polarity = "positive" if polarity != "negative" else "neutral"

    return text, polarity, confidence


def _extract_keywords(text: str, limit: int = 6) -> List[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_\-]+", (text or "").lower())
    filtered = [t for t in tokens if len(t) > 2 and t not in _STOPWORDS]
    counts = Counter(filtered)
    ranked = [w for w, _ in counts.most_common(limit)]
    return ranked


def _lens_defaults(lens: str) -> Tuple[List[str], List[str]]:
    if lens == "world":
        return ["safety", "clarity", "cooperation"], ["risk", "ambiguity", "conflict"]
    if lens == "self":
        return ["coherence", "growth", "calm"], ["dissonance", "shame", "fatigue"]
    return ["consistency", "guardrails", "persona"], ["drift", "violation", "incoherence"]


def _blend_codelet_signals(codelet_results: List[SimulationCodeletResult], lens: str) -> Tuple[List[str], List[str], List[str], float]:
    attractors: List[str] = []
    repellers: List[str] = []
    evidence: List[str] = []

    lens_keys = {
        "world": ["consequence", "social"],
        "self": ["emotional"],
        "meta": ["social", "consequence"],
    }
    matched = [r for r in codelet_results if any(k in r.scenario.lower() for k in lens_keys[lens])]
    if not matched:
        matched = codelet_results

    util_shift = 0.0
    for r in matched:
        evidence.append(f"[{r.scenario}] {r.predicted_outcome} (risk={r.risk:.2f}, benefit={r.benefit:.2f})")
        if r.benefit >= r.risk:
            attractors.extend(_extract_keywords(r.predicted_outcome, limit=2))
        else:
            repellers.extend(_extract_keywords(r.predicted_outcome, limit=2))
        util_shift += (r.benefit - r.risk) * 0.15

    return attractors, repellers, evidence[:4], util_shift


def _dedupe_keep_order(items: List[str], limit: int = 8) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in items:
        x = x.strip().lower()
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
        if len(out) >= limit:
            break
    return out


def _build_sim_bundle(
    tick: int,
    context_text: str,
    proposed_action: str,
    core_outputs: Dict[str, str],
    codelet_results: List[SimulationCodeletResult],
    expectation_profile: Dict[str, float] | None = None,
) -> SimBundle:
    outcomes: List[SimLensOutcome] = []
    profile = dict(expectation_profile or {})
    disappointment = _clamp(float(profile.get("disappointment_pressure", 0.0) or 0.0), 0.0, 1.0)
    surprise = _clamp(float(profile.get("positive_surprise_bias", 0.0) or 0.0), 0.0, 1.0)

    for lens in ["world", "self", "meta"]:
        summary, polarity, confidence = _parse_core_output(core_outputs.get(lens, ""))
        defaults_pos, defaults_neg = _lens_defaults(lens)

        kw = _extract_keywords(summary, limit=4)
        attractors = defaults_pos + (kw[:2] if polarity != "negative" else [])
        repellers = defaults_neg + (kw[:2] if polarity == "negative" else [])

        ca, cr, ce, util_shift = _blend_codelet_signals(codelet_results, lens)
        attractors.extend(ca)
        repellers.extend(cr)

        base_util = {"positive": 0.72, "neutral": 0.50, "negative": 0.30}[polarity]
        utility = base_util + util_shift
        if polarity == "positive":
            utility -= 0.22 * disappointment
            utility += 0.10 * surprise
        elif polarity == "negative":
            utility += 0.08 * disappointment
        utility = _clamp(utility, 0.0, 1.0)

        if disappointment > 0.10:
            repellers.append("under-delivers_vs_baseline")
        if surprise > 0.10:
            attractors.append("exceeds_baseline")

        outcome = SimLensOutcome(
            lens=lens,
            summary=summary or "(no output)",
            polarity=polarity,
            confidence=confidence,
            utility=utility,
            attractors=_dedupe_keep_order(attractors, limit=8),
            repellers=_dedupe_keep_order(repellers, limit=8),
            evidence=ce,
        )
        outcomes.append(outcome)

    outcomes.sort(key=lambda x: x.utility, reverse=True)
    winner = outcomes[0].lens if outcomes else ""
    low = outcomes[-1].utility if outcomes else 0.0
    high = outcomes[0].utility if outcomes else 0.0
    dissonance = _clamp(abs(high - low), 0.0, 1.0)

    merged_attractors = _dedupe_keep_order([x for o in outcomes for x in o.attractors], limit=12)
    merged_repellers = _dedupe_keep_order([x for o in outcomes for x in o.repellers], limit=12)

    return SimBundle(
        tick=tick,
        context_excerpt=context_text[:240],
        proposed_action=proposed_action[:240],
        outcomes=outcomes,
        consolidated_attractors=merged_attractors,
        consolidated_repellers=merged_repellers,
        winner_lens=winner,
        dissonance=dissonance,
    )


class SimulationProc(BaseProc):
    """
    Simulation Procedure.
    Produces structured world/self/meta simulation bundle with attractors,
    repellers, polarity, utilities, and consolidated arbitration context.
    """

    @staticmethod
    def _ensure_sims(ghost: GhostProtocol) -> None:
        if not hasattr(ghost, "sims"):
            ghost.sims = {
                "world": SimWorld(ghost),
                "self": SimSelf(ghost),
                "meta": SimMeta(ghost),
            }

    @staticmethod
    def _build_context_and_constraints(ghost: GhostProtocol) -> Tuple[str, str]:
        constraints = "Default"
        context_text = "No context"
        broadcast_text = str(getattr(getattr(ghost, "conscious_broadcast", None), "content", "") or "")
        persona_signals = collect_persona_signals(
            ghost,
            query=broadcast_text,
            limit=4,
        )

        if hasattr(ghost, "csm_manager") and ghost.csm_manager is not None and ghost.csm_manager.state.gist:
            context_text = ghost.csm_manager.state.gist

        if hasattr(ghost, "conscious_broadcast") and ghost.conscious_broadcast:
            context_text += f"\nConscious Base: {ghost.conscious_broadcast.content}"

        if hasattr(ghost, "self_model") and ghost.self_model and ghost.self_model.current_focus:
            constraints += (
                f"; Focus: {ghost.self_model.current_focus}; Mood: {ghost.self_model.attention_quality}"
            )
        blueprint = dict(getattr(ghost, "reply_blueprint_last", {}) or {})
        if blueprint:
            constraints += (
                f"; ReplyBlueprint: {str(blueprint.get('name', ''))}"
            )
            prefix = str(blueprint.get("generation_prefix", "") or "").strip()
            if prefix:
                context_text += f"\nReply Blueprint Seed: {prefix}"

        focus_text = "\n".join(
            [
                context_text,
                broadcast_text,
                str(getattr(getattr(ghost, "selected_action_schema", None), "action_description", "") or ""),
                str(getattr(ghost, "ego_directive", "") or ""),
                str(getattr(ghost, "thought_blueprint_directive", "") or ""),
            ]
        )
        prompt_budget = compute_agent_context_budget(
            ghost,
            output_tokens=220,
            min_budget=520,
        )
        packet = AgentContextComposer.build(
            ghost,
            focus_text=focus_text,
            config=AgentContextConfig(
                max_tokens=prompt_budget,
                weights=AgentContextWeights.from_config(
                    dict(getattr(getattr(ghost, "config", None), "agent_context_weights", {}) or {})
                ),
                latest_messages=int(getattr(getattr(ghost, "config", None), "agent_context_latest_messages", 40) or 40),
                workspace_items=int(getattr(getattr(ghost, "config", None), "agent_context_workspace_items", 48) or 48),
                min_section_tokens=int(getattr(getattr(ghost, "config", None), "agent_context_min_section_tokens", 96) or 96),
            ),
            purpose="simulation",
        )
        context_text = str(getattr(packet, "text", "") or "")

        if persona_signals:
            constraints += "; PersonaQuirks: " + " | ".join(persona_signals[:3])
            context_text += "\nPersona Signals:\n- " + "\n- ".join(persona_signals)

        constraints += "; CapabilityProfile: " + capability_context_text(ghost).replace("\n", " | ")

        return context_text, constraints

    @staticmethod
    def run(ghost: GhostProtocol) -> None:
        logger.info("Simulations: Running internal models...")
        trace_substep(ghost, "SimulationProc._ensure_sims", "simulation", SimulationProc._ensure_sims)

        context_text, constraints = trace_substep(
            ghost, "SimulationProc._build_context_and_constraints", "simulation", SimulationProc._build_context_and_constraints
        )

        res_world = trace_substep(
            ghost, "SimulationProc.sim_world", "simulation", lambda g: ghost.sims["world"].run(context_text, constraints)
        )
        res_self = trace_substep(
            ghost, "SimulationProc.sim_self", "simulation", lambda g: ghost.sims["self"].run(context_text, constraints)
        )
        res_meta = trace_substep(
            ghost, "SimulationProc.sim_meta", "simulation", lambda g: ghost.sims["meta"].run(context_text, constraints)
        )

        core_outputs = {
            "world": res_world or "",
            "self": res_self or "",
            "meta": res_meta or "",
        }

        proposed_action = ""
        if hasattr(ghost, "conscious_broadcast") and ghost.conscious_broadcast:
            proposed_action = ghost.conscious_broadcast.content[:200]
        blueprint_name = str((getattr(ghost, "reply_blueprint_last", {}) or {}).get("name", "") or "").strip()
        if blueprint_name:
            proposed_action = f"{proposed_action}\nBlueprint={blueprint_name}".strip()

        codelet_results: List[SimulationCodeletResult] = []
        if proposed_action:
            try:
                codelet_results = trace_substep(
                    ghost, "SimulationCodeletRunner.run_all", "simulation", SimulationCodeletRunner.run_all, proposed_action
                )
                logger.info("Simulations: %d simulation codelets completed.", len(codelet_results))
            except Exception as e:
                logger.error("Simulations: Codelet runner failed: %s", e, exc_info=True)

        bundle = trace_substep(
            ghost,
            "SimulationProc._build_sim_bundle",
            "simulation",
            lambda g: _build_sim_bundle(
            tick=ghost.current_tick_id,
            context_text=context_text,
            proposed_action=proposed_action,
            core_outputs=core_outputs,
            codelet_results=codelet_results,
            expectation_profile=dict(getattr(ghost, "partner_expectation_profile", {}) or {}),
            ),
        )

        # Persist structured bundle for arbitration/action phases.
        ghost.simulation_bundle_last_model = bundle
        ghost.simulation_bundle_last = bundle.model_dump()
        if not hasattr(ghost, "simulation_bundle_history"):
            ghost.simulation_bundle_history = []
        ghost.simulation_bundle_history.append(bundle.model_dump())
        if len(ghost.simulation_bundle_history) > 100:
            ghost.simulation_bundle_history = ghost.simulation_bundle_history[-100:]

        # Emit thought features from each lens outcome.
        for out in bundle.outcomes:
            text = (
                f"Sim[{out.lens}] {out.polarity} util={out.utility:.2f}: {out.summary} "
                f"| attractors={','.join(out.attractors[:3])} "
                f"| repellers={','.join(out.repellers[:3])}"
            )
            f = Feature(content=text, feature_type=FeatureType.Thought, source=f"Sim{out.lens.title()}", causal=False, interlocus=-1)
            ghost.add_knoxel(f)
            if hasattr(ghost, "csm_manager") and ghost.csm_manager is not None:
                act = _clamp(0.45 + out.utility * 0.45, 0.2, 1.0)
                ghost.csm_manager.add_or_boost(
                    CSMItem(
                        knoxel_id=f.id,
                        first_tick=ghost.current_tick_id,
                        last_tick=ghost.current_tick_id,
                        activation=act,
                        peak_activation=act,
                    )
                )

        logger.info(
            "SimBundle: winner=%s dissonance=%.2f attractors=%d repellers=%d",
            bundle.winner_lens,
            bundle.dissonance,
            len(bundle.consolidated_attractors),
            len(bundle.consolidated_repellers),
        )

    @staticmethod
    def predict_action_outcome(ghost: GhostProtocol, action_content: str) -> str:
        """Lightweight textual projection from structured simulation bundle."""
        SimulationProc._ensure_sims(ghost)

        context_text, constraints = SimulationProc._build_context_and_constraints(ghost)
        constraints = f"{constraints}; Predict consequence"

        core_outputs = {
            "world": ghost.sims["world"].run(f"Action Proposed: {action_content}\n{context_text}", constraints) or "",
            "self": ghost.sims["self"].run(f"Action Proposed: {action_content}\n{context_text}", constraints) or "",
            "meta": ghost.sims["meta"].run(f"Action Proposed: {action_content}\n{context_text}", constraints) or "",
        }

        codelet_results: List[SimulationCodeletResult] = []
        try:
            codelet_results = SimulationCodeletRunner.run_all(ghost, action_content)
        except Exception as e:
            logger.warning("Simulation codelet prediction failed: %s", e)

        bundle = _build_sim_bundle(
            tick=getattr(ghost, "current_tick_id", 0),
            context_text=context_text,
            proposed_action=action_content,
            core_outputs=core_outputs,
            codelet_results=codelet_results,
            expectation_profile=dict(getattr(ghost, "partner_expectation_profile", {}) or {}),
        )

        winner = bundle.winner_lens or "world"
        winner_obj = next((o for o in bundle.outcomes if o.lens == winner), None)
        if not winner_obj:
            return "Uncertain outcome."

        return (
            f"[{winner_obj.lens}] {winner_obj.summary} "
            f"(utility={winner_obj.utility:.2f}, polarity={winner_obj.polarity}, "
            f"risk_signals={','.join(winner_obj.repellers[:2]) or 'none'})"
        )
