import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Optional

from pm.data_structures import KnoxelList, ShellCCQUpdate
from pm.ghosts.procedures.action import ActionSelectionProc
from pm.ghosts.procedures.arbitration import ArbitrationProc
from pm.ghosts.procedures.attention import AttentionProc
from pm.ghosts.procedures.base import BaseProc
from pm.ghosts.procedures.codelets import CodeletProc
from pm.ghosts.procedures.csm import CsmProc
from pm.ghosts.procedures.learning import ExpectationRealityProc
from pm.ghosts.procedures.pam import PamProc
from pm.ghosts.procedures.policy import PolicyDynamics
from pm.ghosts.procedures.qualia import QualiaProc
from pm.ghosts.procedures.reply import ReplyGenerationProc
from pm.ghosts.procedures.persona import PersonaPersistenceProc
from pm.ghosts.procedures.memory_reconsolidation import MemoryReconsolidationProc
from pm.ghosts.procedures.simulations import SimulationProc
from pm.ghosts.procedures.state import StateProc
from pm.ghosts.procedures.thought_blueprint import ThoughtBlueprintProc
from pm.ghosts.procedures.workspace import WorkspaceProc
from pm.ghosts.knoxel_trace import (
    finalize_knoxel_trace_session,
    set_knoxel_trace_phase,
    start_knoxel_trace_session,
    trace_knoxel_flow,
)

logger = logging.getLogger(__name__)


def _safe_excerpt(text: str, max_len: int = 120) -> str:
    t = (text or "").replace("\n", " ").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 3].rstrip() + "..."


@dataclass(frozen=True)
class Phase:
    name: str
    fn: Callable


@dataclass(frozen=True)
class PhaseTrace:
    name: str
    ok: bool
    duration_ms: float
    error: Optional[str] = None


def _run_memory_recon_with_throttle(ghost):
    current_tick = int(getattr(ghost, "current_tick_id", 0) or 0)
    min_interval = int(getattr(ghost, "memory_recon_min_tick_interval", 10) or 10)
    last_tick = int(getattr(ghost, "memory_recon_last_tick", -1_000_000) or -1_000_000)

    if current_tick - last_tick < min_interval:
        payload = {
            "tick": current_tick,
            "ok": True,
            "skipped": True,
            "reason": f"throttled_min_interval={min_interval}",
            "before": {},
            "after": {},
            "delta": {},
            "duration_ms": 0.0,
        }
        ghost.memory_recon_last = payload
        hist = list(getattr(ghost, "memory_recon_history", []) or [])
        hist.append(payload)
        if len(hist) > 100:
            hist = hist[-100:]
        ghost.memory_recon_history = hist
        return payload

    payload = MemoryReconsolidationProc.run(ghost)
    ghost.memory_recon_last_tick = current_tick
    return payload


class BaseProcMain(BaseProc):
    """
    Main modular cognitive orchestrator.

    Runtime contract:
    - Runs phase modules in causal order.
    - Never crashes the outer ghost tick on single-phase failure.
    - Always returns a `ShellCCQUpdate` with best-effort content.
    - Updates ghost state fields (`csm_state`, `ccq_state`, `latent_mental_state`) at end.
    """

    PHASES = (
        Phase(
            "learn_bind",
            trace_knoxel_flow(name="ExpectationRealityProc.bind_previous_outcome", phase="learn_bind")(
                ExpectationRealityProc.bind_previous_outcome
            ),
        ),
        Phase("pam", trace_knoxel_flow(name="PamProc.run", phase="pam")(PamProc.run)),
        Phase("csm", trace_knoxel_flow(name="CsmProc.run", phase="csm")(CsmProc.run)),
        Phase("codelets", trace_knoxel_flow(name="CodeletProc.run", phase="codelets")(CodeletProc.run)),
        Phase("state", trace_knoxel_flow(name="StateProc.run", phase="state")(StateProc.run)),
        Phase("policy", trace_knoxel_flow(name="PolicyDynamics.run", phase="policy")(PolicyDynamics.run)),
        Phase(
            "qualia_attention",
            trace_knoxel_flow(name="QualiaProc.apply_self_model_attention", phase="qualia_attention")(
                QualiaProc.apply_self_model_attention
            ),
        ),
        Phase("attention", trace_knoxel_flow(name="AttentionProc.run", phase="attention")(AttentionProc.run)),
        Phase("simulation", trace_knoxel_flow(name="SimulationProc.run", phase="simulation")(SimulationProc.run)),
        Phase("arbitration", trace_knoxel_flow(name="ArbitrationProc.run", phase="arbitration")(ArbitrationProc.run)),
        Phase(
            "thought_blueprint",
            trace_knoxel_flow(name="ThoughtBlueprintProc.run", phase="thought_blueprint")(ThoughtBlueprintProc.run),
        ),
        Phase("workspace", trace_knoxel_flow(name="WorkspaceProc.run", phase="workspace")(WorkspaceProc.run)),
        Phase(
            "qualia_experience",
            trace_knoxel_flow(name="QualiaProc.generate_subjective_experience", phase="qualia_experience")(
                QualiaProc.generate_subjective_experience
            ),
        ),
        Phase(
            "qualia_model",
            trace_knoxel_flow(name="QualiaProc.update_self_model", phase="qualia_model")(QualiaProc.update_self_model),
        ),
        Phase(
            "qualia_action",
            trace_knoxel_flow(name="QualiaProc.apply_self_model_action", phase="qualia_action")(
                QualiaProc.apply_self_model_action
            ),
        ),
        Phase("action", trace_knoxel_flow(name="ActionSelectionProc.run", phase="action")(ActionSelectionProc.run)),
        Phase("reply", trace_knoxel_flow(name="ReplyGenerationProc.run", phase="reply")(ReplyGenerationProc.run)),
        Phase("persona", trace_knoxel_flow(name="PersonaPersistenceProc.run", phase="persona")(PersonaPersistenceProc.run)),
        Phase(
            "learn_store",
            trace_knoxel_flow(name="ExpectationRealityProc.store_current_expectation", phase="learn_store")(
                ExpectationRealityProc.store_current_expectation
            ),
        ),
        Phase(
            "memory_recon",
            trace_knoxel_flow(name="_run_memory_recon_with_throttle", phase="memory_recon")(
                _run_memory_recon_with_throttle
            ),
        ),
    )

    @staticmethod
    def cognitive_cycle(ghost, shell=None) -> ShellCCQUpdate:
        logger.info("--- STARTING MODULAR COGNITIVE CYCLE Tick %s ---", ghost.current_tick_id)
        BaseProcMain._ensure_runtime_contract(ghost)
        start_knoxel_trace_session(ghost)
        traces = []

        for phase in BaseProcMain.PHASES:
            start = perf_counter()
            set_knoxel_trace_phase(ghost, phase.name)
            try:
                phase.fn(ghost)
                traces.append(PhaseTrace(name=phase.name, ok=True, duration_ms=(perf_counter() - start) * 1000.0))
            except Exception:
                # Best-effort progression: keep other phases alive if one fails.
                logger.exception("Phase '%s' failed; continuing cycle.", phase.name)
                traces.append(
                    PhaseTrace(
                        name=phase.name,
                        ok=False,
                        duration_ms=(perf_counter() - start) * 1000.0,
                        error="phase_failure",
                    )
                )

        # Finalize runtime state and return a real CCQ object.
        ccq = BaseProcMain._finalize_cycle(ghost)
        trace_paths = finalize_knoxel_trace_session(ghost, ccq=ccq)
        BaseProcMain._persist_trace(ghost, traces)
        BaseProcMain._persist_debug_panel(ghost, traces, ccq, trace_paths=trace_paths)
        BaseProcMain._persist_event_trace(ghost, traces)
        logger.info("--- END MODULAR COGNITIVE CYCLE Tick %s ---", ghost.current_tick_id)
        return ccq

    @staticmethod
    def _ensure_runtime_contract(ghost) -> None:
        """
        Normalize optional runtime fields consumed by the modular phases.
        """
        defaults = {
            "input_knoxels": [],
            "state_deltas_buffer": [],
            "broadcast_history": [],
            "current_coalition": [],
            "simulated_reply": None,
            "selected_action_schema": None,
            "reply_blueprint_last": {},
            "reply_blueprint_history": [],
            "persona_pattern_signatures": set(),
            "story_context_embedding_cache": {},
            "story_context_last": {},
            "story_context_history": [],
            "agent_context_embedding_cache": {},
            "agent_context_last": {},
            "agent_context_history": [],
            "reply_agent_last": {},
            "conscious_broadcast": None,
            "enable_knoxel_trace": True,
            "knoxel_trace_output_dir": "data/knoxel_traces",
            "memory_recon_last": {},
            "memory_recon_history": [],
            "memory_recon_last_tick": -1_000_000,
            "memory_recon_min_tick_interval": 10,
        }
        for name, value in defaults.items():
            if not hasattr(ghost, name):
                if isinstance(value, (list, dict, set)):
                    setattr(ghost, name, value.copy())
                else:
                    setattr(ghost, name, value)

    @staticmethod
    def _persist_trace(ghost, traces) -> None:
        trace_rows = [
            {
                "phase": t.name,
                "ok": t.ok,
                "duration_ms": round(t.duration_ms, 3),
                "error": t.error,
            }
            for t in traces
        ]
        ghost.last_cycle_trace = trace_rows
        if not hasattr(ghost, "cycle_traces"):
            ghost.cycle_traces = []
        ghost.cycle_traces.append({"tick": ghost.current_tick_id, "phases": trace_rows})
        if len(ghost.cycle_traces) > 100:
            ghost.cycle_traces = ghost.cycle_traces[-100:]

        summary = ", ".join(
            f"{r['phase']}={'ok' if r['ok'] else 'err'}({r['duration_ms']}ms)" for r in trace_rows
        )
        logger.debug("Cycle trace tick=%s %s", ghost.current_tick_id, summary)

    @staticmethod
    def _persist_debug_panel(ghost, traces, ccq: ShellCCQUpdate, trace_paths: Optional[dict] = None) -> None:
        phase_rows = [
            {
                "name": t.name,
                "ok": t.ok,
                "duration_ms": round(t.duration_ms, 3),
                "error": t.error,
            }
            for t in traces
        ]
        phase_total_ms = round(sum(x["duration_ms"] for x in phase_rows), 3)

        coalition = list(getattr(ghost, "current_coalition", []) or [])
        conscious_candidates = list(getattr(ghost, "conscious_candidates", []) or [])
        broadcast = getattr(ghost, "conscious_broadcast", None)
        codelet_trace = getattr(ghost, "codelet_trace_last", {}) or {}
        codelet_timeline = getattr(ghost, "codelet_timeline_last", "")

        meta_run_count = 0
        for run in codelet_trace.get("runs", []):
            fams = run.get("families", [])
            if "MetaMonitoringIdentity" in fams:
                meta_run_count += 1

        emitted_reflection = False
        all_features = list(getattr(ghost, "all_features", []) or [])
        for f in reversed(all_features):
            if getattr(f, "tick_id", None) != ghost.current_tick_id:
                break
            if getattr(f, "source", "") == "CodeletTraceReflection":
                emitted_reflection = True
                break

        panel = {
            "tick": ghost.current_tick_id,
            "phase": {
                "total_ms": phase_total_ms,
                "ok_count": sum(1 for x in phase_rows if x["ok"]),
                "err_count": sum(1 for x in phase_rows if not x["ok"]),
                "rows": phase_rows,
            },
            "attention_workspace": {
                "workspace_gain": float(getattr(ghost, "workspace_gain", 0.0) or 0.0),
                "coalition_target_count": int(getattr(ghost, "coalition_target_count", 0) or 0),
                "generation_token_cap": int(getattr(ghost, "generation_token_cap", 0) or 0),
                "coalition_size": len(coalition),
                "candidate_size": len(conscious_candidates),
                "broadcast_id": getattr(broadcast, "id", None),
                "broadcast_excerpt": (getattr(broadcast, "content", "") or "")[:120],
            },
            "codelets": {
                "run_count": len(codelet_trace.get("runs", [])),
                "selected_families": list(codelet_trace.get("selected_families", []) or []),
                "pathway_themes": list(codelet_trace.get("pathway_themes", []) or []),
                "pathway_seeded": list(codelet_trace.get("pathway_seeded", []) or []),
                "meta_run_count": meta_run_count,
                "precursor_link_count": len(codelet_trace.get("precursor_links", {}) or {}),
                "timeline": codelet_timeline,
            },
            "arbitration": {
                "winner_lens": (getattr(ghost, "ego_decision_last", {}) or {}).get("winner_lens", ""),
                "runner_up_lens": (getattr(ghost, "ego_decision_last", {}) or {}).get("runner_up_lens", ""),
                "dissonance": float((getattr(ghost, "ego_decision_last", {}) or {}).get("dissonance", 0.0) or 0.0),
                "directive_excerpt": ((getattr(ghost, "ego_directive", "") or "")[:120]),
            },
            "qualia": {
                "inner_voice_excerpt": ((getattr(ghost, "subjective_experience", "") or "")[:120]),
                "reflection_confidence": float((getattr(ghost, "qualia_reflection_last", {}) or {}).get("confidence", 0.0) or 0.0),
                "reflection_tension": float((getattr(ghost, "qualia_reflection_last", {}) or {}).get("tension", 0.0) or 0.0),
                "theory_confidence": float((getattr(ghost, "self_model_update_last", {}) or {}).get("confidence", 0.0) or 0.0),
                "theory_mismatch": float((getattr(ghost, "self_model_update_last", {}) or {}).get("mismatch", 0.0) or 0.0),
                "active_theory_count": len(list(getattr(getattr(ghost, "self_model", None), "active_theories", []) or [])),
            },
            "thought_blueprint": {
                "final_thought": (getattr(ghost, "thought_blueprint_last", {}) or {}).get("final_thought", ""),
                "path_len": len((getattr(ghost, "thought_blueprint_last", {}) or {}).get("path", []) or []),
                "directive_excerpt": _safe_excerpt((getattr(ghost, "thought_blueprint_directive", "") or ""), 120),
                "action_hints": list(getattr(ghost, "thought_blueprint_action_hints", []) or []),
            },
            "learning": {
                "pending_queue_len": len(list(getattr(ghost, "pending_expectation_queue", []) or [])),
                "last_outcome": (getattr(ghost, "last_expectation_outcome", {}) or {}).get("outcome", ""),
                "last_match_score": float((getattr(ghost, "last_expectation_outcome", {}) or {}).get("match_score", 0.0) or 0.0),
                "policy_tendency_count": len(dict(getattr(ghost, "policy_tendencies", {}) or {})),
            },
            "reply_agent": {
                "action_description": str(getattr(getattr(ghost, "selected_action_schema", None), "action_description", "") or ""),
                "reply_len": len(str(getattr(ghost, "simulated_reply", "") or "")),
                "reply_excerpt": _safe_excerpt(str(getattr(ghost, "simulated_reply", "") or ""), 120),
                "has_last_payload": bool(getattr(ghost, "reply_agent_last", {})),
                "blueprint_name": str((getattr(ghost, "reply_blueprint_last", {}) or {}).get("name", "") or ""),
                "blueprint_prefix_excerpt": _safe_excerpt(str((getattr(ghost, "reply_blueprint_last", {}) or {}).get("generation_prefix", "") or ""), 120),
            },
            "story_context": {
                "selected_count": int((getattr(ghost, "story_context_last", {}) or {}).get("selected_count", 0) or 0),
                "focus_excerpt": _safe_excerpt(str((getattr(ghost, "story_context_last", {}) or {}).get("focus_excerpt", "") or ""), 120),
                "history_len": len(list(getattr(ghost, "story_context_history", []) or [])),
            },
            "llm_quality": {
                "event_count": len(list(getattr(ghost, "llm_tool_quality_history", []) or [])),
                "last_phase": ((list(getattr(ghost, "llm_tool_quality_history", []) or [])[-1] if getattr(ghost, "llm_tool_quality_history", []) else {}) or {}).get("phase", ""),
                "last_ok": bool(((list(getattr(ghost, "llm_tool_quality_history", []) or [])[-1] if getattr(ghost, "llm_tool_quality_history", []) else {}) or {}).get("ok", False)),
                "last_fallback_reason": ((list(getattr(ghost, "llm_tool_quality_history", []) or [])[-1] if getattr(ghost, "llm_tool_quality_history", []) else {}) or {}).get("fallback_reason", ""),
            },
            "reflection": {
                "eligible": meta_run_count > 0 and float(getattr(ghost, "workspace_gain", 0.0) or 0.0) > 0.0,
                "emitted": emitted_reflection,
            },
            "ccq": {
                "story_len": len(ccq.as_story or ""),
                "assistant_len": len(ccq.as_assistant or ""),
                "knoxel_count": len(ccq.knoxels.to_list()),
                "knoxel_ids": [k.id for k in ccq.knoxels.to_list()],
            },
            "knoxel_trace": {
                "enabled": bool(getattr(ghost, "enable_knoxel_trace", False)),
                "calls": len((getattr(ghost, "last_knoxel_trace", {}) or {}).get("calls", [])),
                "knoxels": len((getattr(ghost, "last_knoxel_trace", {}) or {}).get("knoxels", {})),
                "json_path": str((trace_paths or {}).get("json", "") or ""),
                "cytoscape_json_path": str((trace_paths or {}).get("cytoscape_json", "") or ""),
                "html_path": str((trace_paths or {}).get("html", "") or ""),
            },
        }

        ghost.runtime_debug_panel = panel
        if not hasattr(ghost, "runtime_debug_panel_history"):
            ghost.runtime_debug_panel_history = []
        ghost.runtime_debug_panel_history.append(panel)
        if len(ghost.runtime_debug_panel_history) > 100:
            ghost.runtime_debug_panel_history = ghost.runtime_debug_panel_history[-100:]

        logger.debug(
            "DebugPanel tick=%s gain=%.2f coalition=%d codelet_runs=%d",
            ghost.current_tick_id,
            panel["attention_workspace"]["workspace_gain"],
            panel["attention_workspace"]["coalition_size"],
            panel["codelets"]["run_count"],
        )

    @staticmethod
    def _persist_event_trace(ghost, traces) -> None:
        """
        Persist per-phase event rows with key decisions for replay/debug workflows.
        """
        panel = getattr(ghost, "runtime_debug_panel", {}) or {}
        attn = panel.get("attention_workspace", {}) or {}
        arb = panel.get("arbitration", {}) or {}
        thought = panel.get("thought_blueprint", {}) or {}
        learn = panel.get("learning", {}) or {}
        llmq = panel.get("llm_quality", {}) or {}

        phase_keys = {
            "attention": {
                "workspace_gain": float(attn.get("workspace_gain", 0.0) or 0.0),
                "coalition_target_count": int(attn.get("coalition_target_count", 0) or 0),
                "generation_token_cap": int(attn.get("generation_token_cap", 0) or 0),
            },
            "arbitration": {
                "winner_lens": str(arb.get("winner_lens", "") or ""),
                "runner_up_lens": str(arb.get("runner_up_lens", "") or ""),
                "dissonance": float(arb.get("dissonance", 0.0) or 0.0),
            },
            "thought_blueprint": {
                "final_thought": str(thought.get("final_thought", "") or ""),
                "path_len": int(thought.get("path_len", 0) or 0),
            },
            "action": {
                "simulated_reply_len": len(str(getattr(ghost, "simulated_reply", "") or "")),
                "ego_directive_excerpt": _safe_excerpt(str(getattr(ghost, "ego_directive", "") or "")),
                "action_description": str(getattr(getattr(ghost, "selected_action_schema", None), "action_description", "") or ""),
            },
            "reply": {
                "simulated_reply_len": len(str(getattr(ghost, "simulated_reply", "") or "")),
                "reply_excerpt": _safe_excerpt(str(getattr(ghost, "simulated_reply", "") or "")),
                "reply_blueprint": str((getattr(ghost, "reply_blueprint_last", {}) or {}).get("name", "") or ""),
                "story_context_selected_count": int((getattr(ghost, "story_context_last", {}) or {}).get("selected_count", 0) or 0),
            },
            "learn_store": {
                "pending_queue_len": int(learn.get("pending_queue_len", 0) or 0),
                "last_match_score": float(learn.get("last_match_score", 0.0) or 0.0),
            },
            "learn_bind": {
                "last_outcome": str(learn.get("last_outcome", "") or ""),
            },
            "qualia_model": {
                "theory_confidence": float((panel.get("qualia", {}) or {}).get("theory_confidence", 0.0) or 0.0),
                "theory_mismatch": float((panel.get("qualia", {}) or {}).get("theory_mismatch", 0.0) or 0.0),
            },
        }

        event_rows = []
        for t in traces:
            row = {
                "tick": int(getattr(ghost, "current_tick_id", 0) or 0),
                "phase": t.name,
                "ok": bool(t.ok),
                "duration_ms": round(float(t.duration_ms), 3),
                "error": t.error,
                "llm_last_phase": str(llmq.get("last_phase", "") or ""),
                "llm_last_ok": bool(llmq.get("last_ok", False)),
                "key_decisions": phase_keys.get(t.name, {}),
            }
            event_rows.append(row)

        ghost.last_cycle_events = event_rows
        if not hasattr(ghost, "cycle_event_history"):
            ghost.cycle_event_history = []
        ghost.cycle_event_history.append({"tick": ghost.current_tick_id, "events": event_rows})
        if len(ghost.cycle_event_history) > 100:
            ghost.cycle_event_history = ghost.cycle_event_history[-100:]

    @staticmethod
    def _finalize_cycle(ghost) -> ShellCCQUpdate:
        knoxels = KnoxelList()
        story = ""

        # Preferred source 1: simulated reply (already action-selected response text).
        if getattr(ghost, "simulated_reply", None):
            story = ghost.simulated_reply

        # Preferred source 2: conscious broadcast content if available.
        broadcast = getattr(ghost, "conscious_broadcast", None)
        if broadcast is not None:
            knoxels.add(broadcast)
            if not story:
                story = broadcast.content or ""

        # Preferred source 3: most recent input feature fallback.
        if len(knoxels.to_list()) == 0:
            for f in reversed(getattr(ghost, "all_features", [])):
                knoxels.add(f)
                if not story:
                    story = f.content or ""
                break

        # Last-resort non-empty story for shells that expect content.
        if not story:
            story = "<No conscious content generated in this tick.>"

        # Persist state for downstream modules/persistence.
        if ghost.current_state:
            if hasattr(ghost, "csm_manager"):
                ghost.current_state.csm_state = ghost.csm_manager.state
            ghost.current_state.ccq_state = {x.id: 1.0 for x in knoxels.to_list()}
            try:
                ghost.current_state.latent_mental_state = ghost._compute_mental_state()
            except Exception:
                logger.exception("Failed to recompute latent mental state; preserving previous state.")

        return ShellCCQUpdate(
            last_causal_id=ghost.max_knoxel_id if getattr(ghost, "all_knoxels", None) else -1,
            current_tick=ghost.current_tick_id,
            knoxels=knoxels,
            as_story=story,
            as_assistant=story,
        )

