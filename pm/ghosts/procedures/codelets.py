import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from pydantic import BaseModel

from pm.codelets.codelet import CodeletContext, CodeletExecutor, CodeletFamily, CodeletRegistry
from pm.codelets.pathways_detailed import CodeletActivation, get_codelet_pathways
from pm.csm.csm import CSMItem
from pm.data_structures import Feature, FeatureType
from pm.ghosts.knoxel_trace import trace_substep
from pm.ghosts.procedures.base import BaseProc, GhostProtocol
from pm.ghosts.schemas import CodeletRunTrace, CodeletTickTrace
from pm.mental_state_vectors import compute_state_delta, create_empty_ms_vector
from pm.utils.pydantic_utils import basemodel_to_text, pydandic_model_to_dict_jsonable

logger = logging.getLogger(__name__)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _compute_state_boosts(ghost: GhostProtocol) -> Dict[CodeletFamily, float]:
    """Compute family-level activation multipliers from current mental state."""
    boosts: Dict[CodeletFamily, float] = {}

    ms = None
    if getattr(ghost, "current_state", None) and ghost.current_state.latent_mental_state:
        ms = ghost.current_state.latent_mental_state

    if not ms:
        return boosts

    if ms.state_core and ms.state_core.arousal > 0.7:
        boosts[CodeletFamily.RegulationCoping] = 1.5

    if ms.state_needs:
        need_fields = [
            "energy_stability",
            "connection",
            "closeness_need",
            "relevance",
            "learning_growth",
            "creative_expression",
            "autonomy",
        ]
        critical_needs = sum(1 for f in need_fields if getattr(ms.state_needs, f, 0.5) < 0.3)
        if critical_needs >= 2:
            boosts[CodeletFamily.DriversHomeostatis] = 1.8

    if hasattr(ghost, "self_model") and ghost.self_model and ghost.self_model.confidence < 0.4:
        boosts[CodeletFamily.MetaMonitoringIdentity] = 1.6

    if ms.state_core and ms.state_core.valence < -0.5:
        boosts[CodeletFamily.Appraisals] = 1.3

    if ms.state_neurochemical and ms.state_neurochemical.dopamine > 0.7:
        boosts[CodeletFamily.ImaginationSimulation] = 1.4

    profile = dict(getattr(ghost, "partner_expectation_profile", {}) or {})
    disappointment = _clamp(float(profile.get("disappointment_pressure", 0.0) or 0.0), 0.0, 1.0)
    surprise = _clamp(float(profile.get("positive_surprise_bias", 0.0) or 0.0), 0.0, 1.0)

    if disappointment > 0.05:
        boosts[CodeletFamily.RegulationCoping] = max(boosts.get(CodeletFamily.RegulationCoping, 1.0), 1.0 + 1.10 * disappointment)
        boosts[CodeletFamily.Appraisals] = max(boosts.get(CodeletFamily.Appraisals, 1.0), 1.0 + 0.80 * disappointment)
        boosts[CodeletFamily.MetaMonitoringIdentity] = max(
            boosts.get(CodeletFamily.MetaMonitoringIdentity, 1.0),
            1.0 + 0.45 * disappointment,
        )
        # Under elevated expectation pressure, reduce exuberant narrative drift.
        boosts[CodeletFamily.Narratives] = min(boosts.get(CodeletFamily.Narratives, 1.0), 1.0 - 0.25 * disappointment)

    if surprise > 0.05:
        boosts[CodeletFamily.Narratives] = max(boosts.get(CodeletFamily.Narratives, 1.0), 1.0 + 0.20 * surprise)
        boosts[CodeletFamily.ActionTendencies] = max(boosts.get(CodeletFamily.ActionTendencies, 1.0), 1.0 + 0.15 * surprise)

    return boosts


def _build_csm_snapshot(ghost: GhostProtocol, max_items: int = 20) -> str:
    """Robust CSM snapshot serialization used as codelet context."""
    if not hasattr(ghost, "csm_manager") or ghost.csm_manager is None:
        return ""

    items = ghost.csm_manager.get_active_items(min_activation=0.18, limit=max_items)
    lines = []
    for item in items:
        k = ghost.get_knoxel_by_id(item.knoxel_id)
        if not k:
            continue

        content = (getattr(k, "content", "") or "").replace("\n", " ").strip()
        if len(content) > 180:
            content = content[:177].rstrip() + "..."

        source = getattr(k, "source", "") or "-"
        ftype = getattr(k, "feature_type", "")
        lines.append(
            f"[{item.activation:.2f}] id={item.knoxel_id} type={k.__class__.__name__} "
            f"feature_type={ftype} source={source} content={content}"
        )
    return "\n".join(lines)


def _get_ctx_embedding(ghost: GhostProtocol) -> List[float]:
    if ghost.primary_stimulus and getattr(ghost.primary_stimulus, "embedding", None):
        return ghost.primary_stimulus.embedding
    if hasattr(ghost, "current_focus_embedding") and ghost.current_focus_embedding:
        return ghost.current_focus_embedding
    if hasattr(ghost, "csm_manager") and ghost.csm_manager is not None and ghost.csm_manager.state.gist:
        try:
            return ghost.llm.get_embedding(ghost.csm_manager.state.gist)
        except Exception:
            return []
    return []


def _triage_families(
    candidates: List[Tuple[CodeletExecutor, float]],
    state_boosts: Dict[CodeletFamily, float],
    max_families: int = 4,
) -> Set[CodeletFamily]:
    family_scores: Dict[CodeletFamily, float] = {}

    for executor, base_score in candidates:
        families = getattr(executor.signature, "families", []) or []
        for fam in families:
            fam_boost = state_boosts.get(fam, 1.0)
            family_scores[fam] = family_scores.get(fam, 0.0) + (base_score * fam_boost)

    ranked = sorted(family_scores.items(), key=lambda x: (-x[1], x[0].name if hasattr(x[0], "name") else str(x[0])))
    return {fam for fam, _ in ranked[:max_families]}


def _extract_percepts(output_feature: BaseModel) -> List[Tuple[str, BaseModel]]:
    percepts = []
    for field_name in output_feature.__class__.model_fields:
        value = getattr(output_feature, field_name, None)
        if isinstance(value, BaseModel):
            percepts.append((field_name, value))
    return percepts


def _apply_pathway_boosts(
    registry: CodeletRegistry,
    fired_name: str,
    precursor_feature_ids: List[int],
    precursor_dict: Dict[str, List[int]],
) -> List[str]:
    pathways = get_codelet_pathways()
    boosted: List[str] = []
    for _, sequence_list in pathways.items():
        for seq in sequence_list:
            found = False
            for idx, step in enumerate(seq):
                if isinstance(step, CodeletActivation):
                    if found and idx + 1 < len(seq):
                        next_codelet = seq[idx + 1]
                        registry.boost_codelet(next_codelet, factor=step.strength)
                        if next_codelet.name not in precursor_dict:
                            precursor_dict[next_codelet.name] = []
                        precursor_dict[next_codelet.name].extend(precursor_feature_ids)
                        boosted.append(next_codelet.name)
                else:
                    if getattr(step, "name", None) == fired_name:
                        found = True
    return sorted(set(boosted))


def _select_pathway_themes(ghost: GhostProtocol) -> List[str]:
    text_parts: List[str] = []
    if getattr(ghost, "primary_stimulus", None) and ghost.primary_stimulus.content:
        text_parts.append(ghost.primary_stimulus.content.lower())
    if hasattr(ghost, "csm_manager") and ghost.csm_manager is not None and ghost.csm_manager.state.gist:
        text_parts.append(ghost.csm_manager.state.gist.lower())
    if getattr(ghost, "conscious_broadcast", None) and ghost.conscious_broadcast.content:
        text_parts.append(ghost.conscious_broadcast.content.lower())

    text = " ".join(text_parts)
    if not text:
        return []

    theme_keywords: Dict[str, List[str]] = {
        "anxiety": ["worry", "anxious", "fear", "threat", "panic", "unsafe"],
        "curiosity": ["curious", "explore", "question", "learn", "discover", "wonder"],
        "love": ["care", "bond", "closeness", "affection", "together", "trust"],
        "empathy_compassion": ["sorry", "hurt", "support", "understand", "compassion", "empathy"],
        "persistence_goal_pursuit": ["goal", "plan", "progress", "finish", "complete", "step"],
        "wrath": ["angry", "rage", "furious", "offended", "attack", "retaliate"],
        "sloth": ["tired", "fatigue", "later", "delay", "avoid", "procrastinate"],
        "playfulness_creativity": ["joke", "play", "creative", "fun", "improvise", "humor"],
    }

    selected = []
    for theme, keys in theme_keywords.items():
        if any(k in text for k in keys):
            selected.append(theme)
    return sorted(set(selected))


def _seed_pathway_boosts(registry: CodeletRegistry, themes: List[str]) -> List[str]:
    pathways = get_codelet_pathways()
    boosted_codelets: List[str] = []
    for theme in themes:
        for seq in pathways.get(theme, []):
            if not seq:
                continue
            first_step = seq[0]
            if not hasattr(first_step, "name"):
                continue

            factor = 1.15
            if len(seq) > 1 and isinstance(seq[1], CodeletActivation):
                factor = max(1.0, min(2.0, float(seq[1].strength)))

            registry.boost_codelet(first_step, factor=factor)
            boosted_codelets.append(first_step.name)
    return sorted(set(boosted_codelets))


def render_codelet_timeline(trace: CodeletTickTrace, max_runs: int = 6) -> str:
    """Render a compact, human-readable timeline for mental-faculty debugging."""
    lines: List[str] = [
        f"tick={trace.tick} families={','.join(trace.selected_families) or '-'} "
        f"themes={','.join(trace.pathway_themes) or '-'} seeded={len(trace.pathway_seeded)} runs={len(trace.runs)}"
    ]
    for run in trace.runs[:max_runs]:
        mode = "F" if run.output_mode == "fallback" else "S"
        precursor = f"+p{run.precursor_feature_count}" if run.used_precursor_boost else ""
        boosted = f" ->{','.join(run.boosted_followups[:2])}" if run.boosted_followups else ""
        lines.append(
            f"#{run.iteration} {run.codelet}({mode}) score={run.score:.2f} "
            f"percepts={len(run.percepts)} deltas={run.delta_count}{precursor}{boosted}"
        )
    if len(trace.runs) > max_runs:
        lines.append(f"... +{len(trace.runs) - max_runs} more runs")
    return "\n".join(lines)


def _build_meta_reflection_text(trace: CodeletTickTrace) -> str:
    meta_runs = [r for r in trace.runs if "MetaMonitoringIdentity" in r.families]
    if not meta_runs:
        meta_runs = trace.runs[:2]
    else:
        meta_runs = meta_runs[:3]

    run_bits = [
        f"{r.codelet}({r.score:.2f}, d={r.delta_count}, p={r.precursor_feature_count})" for r in meta_runs
    ]
    return (
        f"Internal cognitive review at tick {trace.tick}: "
        f"pathway_themes={','.join(trace.pathway_themes) or 'none'}, "
        f"meta_runs={'; '.join(run_bits) or 'none'}. "
        "Maintaining self-coherence by tracking precursor-influenced thought transitions."
    )


def _should_emit_meta_reflection(ghost: GhostProtocol, trace: CodeletTickTrace) -> bool:
    ms = getattr(getattr(ghost, "current_state", None), "latent_mental_state", None)
    cog = getattr(ms, "state_cognition", None)
    if cog is None:
        return False

    # Internal focus gate
    interlocus = float(getattr(cog, "interlocus", 0.0) or 0.0)
    if interlocus > -0.35:
        return False

    # Meta activity gate
    meta_count = sum(1 for r in trace.runs if "MetaMonitoringIdentity" in r.families)
    return meta_count >= 1


def _maybe_emit_meta_reflection_feature(ghost: GhostProtocol, trace: CodeletTickTrace) -> None:
    if not _should_emit_meta_reflection(ghost, trace):
        return

    text = _build_meta_reflection_text(trace)
    feat = Feature(
        content=text,
        feature_type=FeatureType.MetaInsight,
        source="CodeletTraceReflection",
        interlocus=-1,
        causal=False,
        metadata={
            "trace_tick": trace.tick,
            "pathway_themes": trace.pathway_themes,
            "run_count": len(trace.runs),
        },
    )
    ghost.add_knoxel(feat)

    if hasattr(ghost, "csm_manager") and ghost.csm_manager is not None:
        ghost.csm_manager.add_or_boost(
            CSMItem(
                knoxel_id=feat.id,
                first_tick=ghost.current_tick_id,
                last_tick=ghost.current_tick_id,
                activation=0.65,
                peak_activation=0.65,
            )
        )


class CodeletProc(BaseProc):
    """
    Deep codelet integration procedure.

    Implements:
    - family-level triage before pick
    - pathway-based boosting
    - robust CSM snapshot serialization
    - percept unpacking into feature + appraisal metadata
    - capped iterative loop with precursor tracking
    """

    @staticmethod
    def run(ghost: GhostProtocol) -> None:
        logger.info("Codelets: deep integration run...")

        if not hasattr(ghost, "codelet_registry") or ghost.codelet_registry is None:
            seed_ctx = CodeletContext(
                tick_id=ghost.current_tick_id,
                stimulus=ghost.primary_stimulus,
                llm=ghost.llm,
                story="",
                story_new="",
                csm_snapshot="",
            )
            ghost.codelet_registry = CodeletRegistry.init_from_simple_codelets(seed_ctx)

        current_state = ghost.current_state
        if not current_state:
            return

        # Keep runtime continuity with previous tick's codelet state.
        if ghost.previous_state and getattr(ghost.previous_state, "codelet_state", None):
            try:
                ghost.codelet_registry.apply_codelet_state(ghost.previous_state.codelet_state)
            except Exception:
                logger.exception("Codelets: failed to apply previous codelet state")

        ghost.codelet_registry.decay_step()

        conversation_history = ""
        if hasattr(ghost, "all_narratives") and ghost.all_narratives:
            recent = sorted(ghost.all_narratives, key=lambda n: n.tick_id, reverse=True)[:3]
            conversation_history = "\n".join(n.content for n in recent)

        ctx = CodeletContext(
            tick_id=ghost.current_tick_id,
            stimulus=ghost.primary_stimulus,
            mental_state=current_state.latent_mental_state,
            context_embedding=_get_ctx_embedding(ghost),
            llm=ghost.llm,
            story=conversation_history,
            story_new="",
            csm_snapshot=trace_substep(ghost, "CodeletProc._build_csm_snapshot_init", "codelets", _build_csm_snapshot),
        )

        candidates = trace_substep(ghost, "CodeletProc.pick_candidates_init", "codelets", lambda g: ghost.codelet_registry.pick_candidates(ctx))
        if not candidates:
            ghost.current_state.codelet_state = ghost.codelet_registry.get_codelet_state()
            return

        # Family triage.
        state_boosts = trace_substep(ghost, "CodeletProc._compute_state_boosts", "codelets", _compute_state_boosts)
        pathway_themes = trace_substep(ghost, "CodeletProc._select_pathway_themes", "codelets", _select_pathway_themes)
        pathway_seeded = trace_substep(
            ghost,
            "CodeletProc._seed_pathway_boosts",
            "codelets",
            lambda g: _seed_pathway_boosts(ghost.codelet_registry, pathway_themes),
        )
        selected_families = trace_substep(
            ghost,
            "CodeletProc._triage_families",
            "codelets",
            lambda g: _triage_families(candidates, state_boosts, 4),
        )

        # Structured trace logs for this tick.
        tick_runs: List[CodeletRunTrace] = []

        precursor_dict: Dict[str, List[int]] = {}
        fired_names: Set[str] = set()
        max_iterations = 4
        threshold = 0.55

        for idx in range(max_iterations):
            ctx.csm_snapshot = trace_substep(ghost, "CodeletProc._build_csm_snapshot_iter", "codelets", _build_csm_snapshot)
            raw_candidates = trace_substep(
                ghost, "CodeletProc.pick_candidates_iter", "codelets", lambda g: ghost.codelet_registry.pick_candidates(ctx)
            )

            boosted: List[Tuple[CodeletExecutor, float]] = []
            for executor, base_score in raw_candidates:
                families = getattr(executor.signature, "families", []) or []
                if selected_families and not any(f in selected_families for f in families):
                    continue

                score = float(base_score)
                for fam in families:
                    score *= state_boosts.get(fam, 1.0)

                precursor_count = len(precursor_dict.get(executor.signature.name, []))
                if executor.signature.name in precursor_dict:
                    score += min(0.25, 0.03 * precursor_count)

                boosted.append((executor, min(1.0, score)))

            boosted.sort(key=lambda x: x[1], reverse=True)

            chosen: Optional[Tuple[CodeletExecutor, float]] = None
            for executor, score in boosted:
                if score < threshold:
                    continue
                if executor.signature.name in fired_names:
                    continue
                chosen = (executor, score)
                break

            if chosen is None:
                break

            executor, score = chosen
            fired_names.add(executor.signature.name)

            precursor_count = len(precursor_dict.get(executor.signature.name, []))
            run_log = {
                "iteration": idx + 1,
                "codelet": executor.signature.name,
                "families": [f.name for f in (getattr(executor.signature, "families", []) or [])],
                "score": round(score, 4),
                "percepts": [],
                "feature_ids": [],
                "delta_count": 0,
                "summary_excerpt": "",
                "output_mode": "structured",
                "precursor_feature_count": precursor_count,
                "used_precursor_boost": precursor_count > 0,
                "boosted_followups": [],
            }

            try:
                result_dict = executor.run(ctx) or {}
            except Exception:
                logger.exception("Codelet %s failed", executor.signature.name)
                continue

            first_pass = (result_dict.get("first_pass") or "").strip()
            output_feature = result_dict.get("output_feature")
            explicit_deltas = result_dict.get("mental_state_deltas") or []
            if explicit_deltas and hasattr(ghost, "state_deltas_buffer"):
                for delta in explicit_deltas:
                    ghost.state_deltas_buffer.append(delta)
                run_log["delta_count"] += len(explicit_deltas)

            if not isinstance(output_feature, BaseModel):
                # Fall back to coarse output feature when schema output is unavailable.
                content = first_pass or str(result_dict)
                run_log["summary_excerpt"] = content[:120]
                run_log["output_mode"] = "fallback"
                f = Feature(
                    content=content,
                    feature_type=FeatureType.CodeletOutput,
                    source=executor.signature.name,
                    causal=False,
                    interlocus=-1,
                )
                ghost.add_knoxel(f)
                run_log["feature_ids"].append(f.id)
                if hasattr(ghost, "csm_manager") and ghost.csm_manager is not None:
                    ghost.csm_manager.add_or_boost(
                        CSMItem(
                            knoxel_id=f.id,
                            first_tick=ghost.current_tick_id,
                            last_tick=ghost.current_tick_id,
                            activation=0.75,
                            peak_activation=0.75,
                        )
                    )
                boosted_followups = _apply_pathway_boosts(
                    registry=ghost.codelet_registry,
                    fired_name=executor.signature.name,
                    precursor_feature_ids=[f.id],
                    precursor_dict=precursor_dict,
                )
                run_log["boosted_followups"] = boosted_followups
                executor.runtime.last_fired_tick = ghost.current_tick_id
                executor.runtime.total_runs += 1
                tick_runs.append(CodeletRunTrace.model_validate(run_log))
                continue

            percept_features = _extract_percepts(output_feature)
            salience_values: List[float] = []
            valence_values: List[float] = []
            created_ids: List[int] = []

            for field_name, percept in percept_features:
                salience = float(getattr(percept, "salience", 0.5) or 0.5)
                valence = float(getattr(percept, "valence", 0.0) or 0.0)

                appraisal_vec = create_empty_ms_vector()
                delta_vec = create_empty_ms_vector()

                appraisal_general = getattr(percept, "appraisal_general", None)
                appraisal_social = getattr(percept, "appraisal_social", None)
                if (
                    appraisal_general is not None
                    and appraisal_social is not None
                    and current_state.latent_mental_state is not None
                ):
                    try:
                        tmp_ms = current_state.latent_mental_state.copy(deep=True)
                        tmp_ms.appraisal_general = appraisal_general
                        tmp_ms.appraisal_social = appraisal_social
                        derived_state, derived_delta = compute_state_delta(tmp_ms, relationship_entity_id=None)
                        appraisal_vec = derived_state.to_list()
                        delta_vec = derived_delta.to_list()

                        if hasattr(ghost, "state_deltas_buffer"):
                            # Keep appraisal deltas explicit for StateProc compatibility.
                            pushed = 0
                            ghost.state_deltas_buffer.append(appraisal_general)
                            pushed += 1
                            ghost.state_deltas_buffer.append(appraisal_social)
                            pushed += 1
                            # Also pass other derived substate deltas when present.
                            for sub_name in (
                                "state_neurochemical",
                                "state_core",
                                "state_emotions",
                                "state_needs",
                                "state_cognition",
                            ):
                                sub_delta = getattr(derived_delta, sub_name, None)
                                if sub_delta is not None:
                                    ghost.state_deltas_buffer.append(sub_delta)
                                    pushed += 1
                            run_log["delta_count"] += pushed
                    except Exception:
                        logger.exception(
                            "Codelets: failed state-derivation for percept %s.%s",
                            executor.signature.name,
                            field_name,
                        )

                content = basemodel_to_text(percept)
                feat = Feature(
                    content=content,
                    feature_type=FeatureType.CodeletPercept,
                    source=f"{executor.signature.name}.{field_name}",
                    affective_valence=valence,
                    incentive_salience=salience,
                    interlocus=-1,
                    causal=False,
                    metadata={
                        "codelet": executor.signature.name,
                        "percept_type": percept.__class__.__name__,
                        "raw": pydandic_model_to_dict_jsonable(percept),
                    },
                    mental_state_appraisal=appraisal_vec,
                    mental_state_delta=delta_vec,
                )
                ghost.add_knoxel(feat)
                created_ids.append(feat.id)

                if hasattr(ghost, "csm_manager") and ghost.csm_manager is not None:
                    activation = max(0.20, min(1.0, 0.45 + 0.45 * salience))
                    ghost.csm_manager.add_or_boost(
                        CSMItem(
                            knoxel_id=feat.id,
                            first_tick=ghost.current_tick_id,
                            last_tick=ghost.current_tick_id,
                            activation=activation,
                            peak_activation=activation,
                        )
                    )

                salience_values.append(salience)
                valence_values.append(valence)
                run_log["percepts"].append(percept.__class__.__name__)

            # Aggregate codelet output summary feature.
            mean_salience = float(np.mean(salience_values)) if salience_values else 0.4
            mean_valence = float(np.mean(valence_values)) if valence_values else 0.0
            summary_content = first_pass or basemodel_to_text(output_feature)
            run_log["summary_excerpt"] = summary_content[:120]
            summary_feature = Feature(
                content=summary_content,
                feature_type=FeatureType.CodeletOutput,
                source=executor.signature.name,
                affective_valence=mean_valence,
                incentive_salience=mean_salience,
                interlocus=-1,
                causal=False,
                metadata={
                    "codelet": executor.signature.name,
                    "output_feature": pydandic_model_to_dict_jsonable(output_feature),
                    "percept_ids": created_ids,
                    "selected_families": sorted(f.name for f in selected_families),
                },
            )
            ghost.add_knoxel(summary_feature)
            created_ids.append(summary_feature.id)

            if hasattr(ghost, "csm_manager") and ghost.csm_manager is not None:
                boost_activation = max(0.30, min(1.0, 0.55 + 0.35 * mean_salience))
                ghost.csm_manager.add_or_boost(
                    CSMItem(
                        knoxel_id=summary_feature.id,
                        first_tick=ghost.current_tick_id,
                        last_tick=ghost.current_tick_id,
                        activation=boost_activation,
                        peak_activation=boost_activation,
                    )
                )

            run_log["feature_ids"] = created_ids
            boosted_followups = _apply_pathway_boosts(
                registry=ghost.codelet_registry,
                fired_name=executor.signature.name,
                precursor_feature_ids=created_ids,
                precursor_dict=precursor_dict,
            )
            run_log["boosted_followups"] = boosted_followups

            executor.runtime.last_fired_tick = ghost.current_tick_id
            executor.runtime.total_runs += 1
            tick_runs.append(CodeletRunTrace.model_validate(run_log))

        # Persist registry runtime state for next tick.
        ghost.current_state.codelet_state = ghost.codelet_registry.get_codelet_state()

        tick_trace = CodeletTickTrace(
            tick=ghost.current_tick_id,
            selected_families=sorted(f.name for f in selected_families),
            pathway_themes=pathway_themes,
            pathway_seeded=pathway_seeded,
            runs=tick_runs,
            precursor_links={k: list(v) for k, v in precursor_dict.items()},
        )

        ghost.codelet_trace_last_model = tick_trace
        ghost.codelet_trace_last = tick_trace.model_dump()
        timeline = render_codelet_timeline(tick_trace)
        ghost.codelet_timeline_last = timeline
        if not hasattr(ghost, "codelet_timeline_history"):
            ghost.codelet_timeline_history = []
        ghost.codelet_timeline_history.append({"tick": tick_trace.tick, "timeline": timeline})
        if len(ghost.codelet_timeline_history) > 100:
            ghost.codelet_timeline_history = ghost.codelet_timeline_history[-100:]

        _maybe_emit_meta_reflection_feature(ghost, tick_trace)
        if not hasattr(ghost, "codelet_trace_history"):
            ghost.codelet_trace_history = []
        ghost.codelet_trace_history.append(tick_trace.model_dump())
        if len(ghost.codelet_trace_history) > 100:
            ghost.codelet_trace_history = ghost.codelet_trace_history[-100:]

        logger.info(
            "Codelets: ran %s codelets; families=%s",
            len(tick_trace.runs),
            tick_trace.selected_families,
        )

