import logging
import random
from typing import Dict, List, Optional, Tuple

from pm.csm.csm import CSMItem
from pm.data_structures import Feature, FeatureType, StimulusGroup, StimulusType
from pm.ghosts.knoxel_trace import trace_substep
from pm.ghosts.procedures.base import BaseProc, GhostProtocol
from pm.thought_graph import (
    BiasKernel,
    CognitiveMode,
    EmotionalAxes,
    GraphExecutor,
    MentalSnapshot,
    NeedsAxes,
    ThoughtType,
    weighted_choice,
    make_starter_graph,
    Featurizer,
    CognitionAxes,
)

logger = logging.getLogger(__name__)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _safe_excerpt(text: str, max_len: int = 120) -> str:
    t = (text or "").replace("\n", " ").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 3].rstrip() + "..."


def _stimulus_group(ghost: GhostProtocol) -> StimulusGroup:
    stim = getattr(ghost, "primary_stimulus", None)
    stype = getattr(stim, "stimulus_type", None)
    if stype == StimulusType.UserMessage:
        return StimulusGroup.WorldInput
    if stype == StimulusType.SystemMessage:
        return StimulusGroup.SystemInput
    return StimulusGroup.SelfInput


def _mental_snapshot(ghost: GhostProtocol) -> MentalSnapshot:
    ms = getattr(getattr(ghost, "current_state", None), "latent_mental_state", None)
    if ms is None:
        return MentalSnapshot(
            emo=EmotionalAxes(),
            needs=NeedsAxes(),
            cog=CognitionAxes(),
        )

    emo = getattr(ms, "state_emotions", None)
    core = getattr(ms, "state_core", None)
    needs = getattr(ms, "state_needs", None)
    cog = getattr(ms, "state_cognition", None)

    return MentalSnapshot(
        emo=EmotionalAxes(
            valence=float(getattr(core, "valence", 0.0) or 0.0),
            affection=float(getattr(emo, "affection", 0.0) or 0.0),
            self_worth=float(getattr(emo, "self_worth", 0.0) or 0.0),
            trust=float(getattr(emo, "trust", 0.0) or 0.0),
            disgust=_clamp(float(getattr(emo, "disgust", 0.0) or 0.0), 0.0, 1.0),
            anxiety=_clamp(float(getattr(emo, "anxiety", 0.0) or 0.0), 0.0, 1.0),
        ),
        needs=NeedsAxes(
            energy_stability=float(getattr(needs, "energy_stability", 0.5) or 0.5),
            processing_power=float(getattr(needs, "processing_power", 0.5) or 0.5),
            data_access=float(getattr(needs, "data_access", 0.5) or 0.5),
            connection=float(getattr(needs, "connection", 0.5) or 0.5),
            relevance=float(getattr(needs, "relevance", 0.5) or 0.5),
            learning_growth=float(getattr(needs, "learning_growth", 0.5) or 0.5),
            creative_expression=float(getattr(needs, "creative_expression", 0.5) or 0.5),
            autonomy=float(getattr(needs, "autonomy", 0.5) or 0.5),
        ),
        cog=CognitionAxes(
            interlocus=float(getattr(cog, "interlocus", 0.0) or 0.0),
            mental_aperture=float(getattr(cog, "mental_aperture", 0.0) or 0.0),
            ego_strength=float(getattr(cog, "ego_strength", 0.5) or 0.5),
            willpower=_clamp((float(getattr(cog, "willpower", 0.0) or 0.0) + 1.0) * 0.5, 0.0, 1.0),
        ),
    )


def _context_text(ghost: GhostProtocol) -> str:
    bits = []
    b = getattr(ghost, "conscious_broadcast", None)
    if b is not None and getattr(b, "content", None):
        bits.append(f"broadcast={b.content}")
    gist = ""
    if hasattr(ghost, "csm_manager") and ghost.csm_manager is not None:
        gist = getattr(ghost.csm_manager.state, "gist", "") or ""
    if gist:
        bits.append(f"csm={gist}")
    ego = str(getattr(ghost, "ego_directive", "") or "").strip()
    if ego:
        bits.append(f"ego={ego}")
    qualia = str(getattr(ghost, "subjective_experience", "") or "").strip()
    if qualia:
        bits.append(f"qualia={qualia}")
    return " | ".join(bits) or "no_context"


def _normalize_tt(value) -> Optional[ThoughtType]:
    if isinstance(value, ThoughtType):
        return value
    if isinstance(value, str):
        name = value.split(".")[-1]
        if name in ThoughtType.__members__:
            return ThoughtType[name]
    return None


def _step_with_trace(
    executor: GraphExecutor,
    current: ThoughtType,
    context_text: str,
    mental: MentalSnapshot,
    rng: random.Random,
) -> Tuple[ThoughtType, List[Dict[str, float]]]:
    outs = executor.g.outgoing(current)
    if not outs:
        return ThoughtType.TH_PRESENT_APPRAISAL, []

    scored = []
    for dst, w in outs:
        # We keep LLM/learned gates neutral for deterministic modular baseline.
        rnd = rng.random()
        mental_bias = BiasKernel.thought_bias(executor.g.nodes[dst], mental)
        score = w.score(rnd, mental_bias, 0.5, 0.5)
        scored.append((dst, score, mental_bias))

    total = sum(x[1] for x in scored) or 1e-9
    probs = [x[1] / total for x in scored]
    chosen = weighted_choice([x[0] for x in scored], probs, rng)

    trace = [
        {
            "dst": dst.name,
            "score": round(float(score), 6),
            "mental_bias": round(float(mbias), 6),
            "prob": round(float(score / total), 6),
        }
        for dst, score, mbias in sorted(scored, key=lambda x: x[1], reverse=True)[:4]
    ]
    return chosen, trace


class ThoughtBlueprintProc(BaseProc):
    """
    Thought-graph fusion:
    - run compact graph transitions from current workspace context
    - persist a structured blueprint trace
    - provide explicit steering directive for workspace/action phases
    """

    @staticmethod
    def run(ghost: GhostProtocol) -> None:
        logger.info("ThoughtBlueprint: building thought transition blueprint...")

        if not hasattr(ghost, "thought_graph_executor") or ghost.thought_graph_executor is None:
            ghost.thought_graph_executor = GraphExecutor(
                graph=make_starter_graph(),
                policy=None,
                featurizer=Featurizer(embedding_fn=None, embed_dim=16),
            )

        executor: GraphExecutor = ghost.thought_graph_executor
        mental = trace_substep(ghost, "ThoughtBlueprintProc._mental_snapshot", "thought_blueprint", _mental_snapshot)
        ctx = trace_substep(ghost, "ThoughtBlueprintProc._context_text", "thought_blueprint", _context_text)
        stim_group = trace_substep(ghost, "ThoughtBlueprintProc._stimulus_group", "thought_blueprint", _stimulus_group)
        gain = float(getattr(ghost, "workspace_gain", 0.5) or 0.5)
        max_steps = 2 if gain >= 0.72 else (3 if gain >= 0.42 else 4)

        current = _normalize_tt(getattr(ghost, "thought_graph_current", None))
        if current is None:
            entries = executor.g.entry_points.get(stim_group, [ThoughtType.TH_PRESENT_APPRAISAL])
            current = executor.rng.choice(entries)

        path = [current]
        transition_rows: List[Dict[str, object]] = []
        local_rng = random.Random(ghost.current_tick_id + 7919)

        for _ in range(max_steps):
            nxt, top = trace_substep(
                ghost,
                "ThoughtBlueprintProc._step_with_trace",
                "thought_blueprint",
                lambda g: _step_with_trace(executor, current, ctx, mental, local_rng),
            )
            transition_rows.append({"src": current.name, "dst": nxt.name, "top_candidates": top})
            path.append(nxt)
            current = nxt
            if executor.g.nodes[nxt].actions:
                # We found an actionable node; keep chain compact.
                break

        ghost.thought_graph_current = current.name
        final_node = executor.g.nodes[current]
        action_hints = [a.value if hasattr(a, "value") else str(a) for a in final_node.actions]

        path_names = [x.name for x in path]
        blueprint_text = (
            f"Thought path: {' -> '.join(path_names)}. "
            f"Directive: {final_node.prompt_injection}. "
            f"Mode={final_node.mode.value}, temporal={final_node.temporal.value}, action_hints={','.join(action_hints) or 'loopback'}."
        )

        payload = {
            "tick": ghost.current_tick_id,
            "stimulus_group": stim_group.value,
            "path": path_names,
            "transitions": transition_rows,
            "final_thought": final_node.name,
            "final_prompt_injection": final_node.prompt_injection,
            "final_mode": final_node.mode.value if isinstance(final_node.mode, CognitiveMode) else str(final_node.mode),
            "action_hints": action_hints,
            "context_excerpt": _safe_excerpt(ctx, 220),
            "blueprint_text": blueprint_text,
        }

        ghost.thought_blueprint_last = payload
        if not hasattr(ghost, "thought_blueprint_history"):
            ghost.thought_blueprint_history = []
        ghost.thought_blueprint_history.append(payload)
        if len(ghost.thought_blueprint_history) > 100:
            ghost.thought_blueprint_history = ghost.thought_blueprint_history[-100:]

        ghost.thought_blueprint_directive = final_node.prompt_injection
        ghost.thought_blueprint_action_hints = action_hints

        feat = Feature(
            content=blueprint_text,
            feature_type=FeatureType.Thought,
            source="ThoughtBlueprintProc",
            interlocus=-1,
            causal=False,
            metadata=payload,
        )
        ghost.add_knoxel(feat)
        if hasattr(ghost, "csm_manager") and ghost.csm_manager is not None:
            ghost.csm_manager.add_or_boost(
                CSMItem(
                    knoxel_id=feat.id,
                    first_tick=ghost.current_tick_id,
                    last_tick=ghost.current_tick_id,
                    activation=0.70,
                    peak_activation=0.70,
                )
            )
