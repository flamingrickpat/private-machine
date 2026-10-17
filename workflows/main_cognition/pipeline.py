"""Pure orchestration of the eight cognitive stages.

The parent runs the ordered children over one request and one shared proposal
batch. Each child is a real kernel workflow invoked through ``run_stage``; this
module owns only the order, the data threaded between stages, and the typed
action dispatch records. It contains no prompts and no model calls.
"""
from __future__ import annotations

import hashlib
import json
from typing import Callable

from workflows.main_cognition.dispatch import resolve_prerequisites, resolve_same_cycle
from workflows.main_cognition.schemas import (
    ActionDispatch,
    AppraisalResult,
    CognitiveReport,
    StageReceipt,
)
from workflows.main_cognition.affect_dynamics import REGULATION_REACTIVE, project_affect
from workflows.main_cognition.codelet_schedule import state_signals

STAGE_ORDER = (
    "triage",
    "appraisal",
    "context",
    "codelets",
    "attention",
    "goals",
    "workspace",
    "action_selection",
)

DEFAULT_WORKFLOWS = [
    {"name": "dialogue_story", "description": "Generate one in-character Story reply."},
    {"name": "dialogue_assistant", "description": "Assistant reply with scoped tools."},
    {"name": "self_management", "description": "Start, status, update, or terminate a managed run."},
    {"name": "local_task", "description": "Run one small real local task."},
    {"name": "minecraft", "description": "Run a sustained Minecraft specialist with exclusive body ownership."},
]


def _candidates(request: dict, reports: dict) -> list[dict]:
    """Build the candidate list for attention from retrieved memories."""
    seen: set[int] = set()
    candidates: list[dict] = []
    context = reports.get("context", {})
    memory_by_id = {
        int(item["knoxel_id"]): str(item.get("text", ""))
        for item in request.get("memory_candidates", [])
    }
    for knoxel_id in context.get("candidate_ids", []):
        if knoxel_id in seen:
            continue
        seen.add(knoxel_id)
        candidates.append({
            "knoxel_id": int(knoxel_id),
            "text": memory_by_id.get(int(knoxel_id), ""),
            "kind": "memory",
        })
    for stimulus in request.get("stimuli", []):
        knoxel_id = int(stimulus["knoxel_id"])
        if knoxel_id in seen:
            continue
        seen.add(knoxel_id)
        candidates.append({
            "knoxel_id": knoxel_id,
            "text": str(stimulus.get("text", "")),
            "kind": "stimulus",
        })
    return candidates


# Accepted proposal kinds each downstream owner consumes. A typed nomination
# without a consumer does not establish a cognitive effect, so each owner
# names the kinds it reads. Every kind's consumer is documented in
# schemas.PROPOSAL_CONSUMERS.
GOAL_KINDS = frozenset({"action_tendency", "belief"})
ACTION_KINDS = frozenset({"action_tendency", "regulation", "simulation"})


def proposal_consumers(kind: str) -> list[str]:
    """Return the owners that consume an accepted proposal of this kind.

    The map is the executable form of ``PROPOSAL_CONSUMERS``: every selected,
    non-stale nomination is always framed by the workspace; the goals stage
    reads motives and self-beliefs; action selection reads wants,
    imagination, and ways to stay steady; the belief owner stages subjective
    beliefs. The host records the list on each committed AttentionRecord, so
    a consumer claim is inspectable per nomination.
    """
    consumers = ["workspace"]
    if kind in GOAL_KINDS:
        consumers.append("goals")
    if kind in ACTION_KINDS:
        consumers.append("action_selection")
    if kind == "belief":
        consumers.append("belief_owner")
    return consumers


def inner_proposals(attention: dict, kinds: frozenset[str]) -> list[dict]:
    """Return the accepted nominations of the given kinds, kind-labeled.

    Only selected, non-empty nominations enter a downstream prompt. Stale
    marking happens later at commit, so the stage treats every winner as a
    proposal, never as a command. Consumers state that in their prompts.
    """
    items = []
    for nomination in (attention or {}).get("nominations", []):
        if nomination.get("status") != "selected" or not str(nomination.get("text", "")).strip():
            continue
        if str(nomination.get("attention_kind", "")) not in kinds:
            continue
        items.append({
            "kind": str(nomination.get("attention_kind", "")),
            "reality": str(nomination.get("reality", "")),
            "text": str(nomination.get("text", "")),
            "target_ids": [int(key) for key in nomination.get("target_ids", [])],
        })
    return items


def _candidate_texts(request: dict) -> list[dict]:
    items = [dict(item) for item in request.get("memory_candidates", [])]
    # Attention can select an active goal without retrieving it as a memory.
    # Workspace must receive the same goal records before it validates focus.
    seen = {int(item["knoxel_id"]) for item in items}
    for goal in request.get("active_goal_records", []):
        if int(goal["knoxel_id"]) not in seen:
            items.append(dict(goal))
            seen.add(int(goal["knoxel_id"]))
    for stimulus in request.get("stimuli", []):
        items.append({"knoxel_id": int(stimulus["knoxel_id"]),
                      "text": str(stimulus.get("text", ""))})
    return list({int(item["knoxel_id"]): item for item in items}.values())


def stable_action_id(intent: dict, index: int) -> str:
    """Return a durable identity for one proposed action.

    The same workflow intent receives the same ID across cycles, so a repeated
    start request can be recognised and suppressed. An action without content
    falls back to its list position because it has no stable identity.
    """
    content = str(intent.get("content", "")).strip()
    if not content:
        return f"action-{index}"
    material = json.dumps({
        "workflow_name": str(intent.get("workflow_name", "")),
        "action_type": str(intent.get("action_type", "Reply")),
        "content": content,
        "arguments": intent.get("arguments", {}),
    }, sort_keys=True, default=str)
    return "action-" + hashlib.sha1(material.encode("utf-8")).hexdigest()[:12]


def action_dispatches(actions: list[dict]) -> list[ActionDispatch]:
    """Convert executable choices into dispatch proposals without external effects.

    Quiet choices remain in the original stage receipt. They cannot invoke a
    body or public workflow, even when they carry a valid workflow name.
    """
    dispatches = []
    for index, intent in enumerate(actions):
        if intent.get("action_type") in {"Ignore", "Idle", "Sleep"}:
            continue
        dispatches.append(ActionDispatch(
            action_id=str(intent.get("action_id") or stable_action_id(intent, index)),
            workflow_name=str(intent.get("workflow_name", "")),
            action_type=str(intent.get("action_type", "Reply")),
            content=str(intent.get("content", "")),
            arguments=dict(intent.get("arguments", {})),
            depends_on=list(intent.get("depends_on", [])),
            requires_completed=list(intent.get("requires_completed", [])),
            presentation=dict(intent.get("presentation", {})),
        ))
    return dispatches


def run_cycle(
    request: dict,
    run_stage: Callable[[str, dict], dict],
    action_receipts: dict[str, str] | None = None,
) -> CognitiveReport:
    """Run all eight stages in order and return the typed parent report.

    ``run_stage(name, payload)`` returns the parsed stage report as a mapping.
    Stage payloads are cumulative: every earlier stage's report is available to
    every later one. ``action_receipts`` maps already-known action IDs to terminal
    states so real success prerequisites can gate dispatch.
    """
    reports: dict[str, dict] = {}
    receipts: list[StageReceipt] = []
    base = dict(request)

    def call(name: str, extra: dict) -> dict:
        payload = {**base, **extra}
        report = run_stage(name, payload)
        if not isinstance(report, dict):
            raise TypeError(f"stage {name} returned {type(report).__name__}, not a mapping")
        reports[name] = report
        receipts.append(StageReceipt(
            name=name,
            status="done",
            summary=str(report.get("summary", "")),
        ))
        return report

    call("triage", {})
    call("appraisal", {"triage": reports["triage"]})

    # Project the affect step once. Later stages read the same projection, and
    # the parent report carries it so the module commits the same state. The
    # projection also maps accepted affect to one regulation mode.
    appraisal_result = _appraisal_from(reports.get("appraisal"))
    user_ids = {int(item["knoxel_id"]) for item in request.get("stimuli", [])
                if item.get("source") == "user_input"}
    partner = (request.get("relationship_entity_id")
               if appraisal_result is not None and user_ids.intersection(appraisal_result.evidence_ids) else None)
    projection = project_affect(
        request.get("mental_state_latent_vec") or None,
        appraisal_result,
        elapsed_seconds=float(request.get("mental_state_age_seconds", 0.0) or 0.0),
        relationship_entity_id=partner,
        personality=request.get("personality_vec") or None,
        expectation_error=float(getattr(appraisal_result, "expectation_error", 0.0) or 0.0),
    )
    affect_block = {
        "summary": projection.summary,
        "regulation": projection.regulation,
        "attention_focus": projection.behavior.get("attention_focus", "neutral"),
        "attention_bias": {
            "valence_focus": round(float(projection.attention_bias[0]), 4),
            "salience": round(float(projection.attention_bias[1]), 4),
        },
        "behavior": projection.behavior,
        "emotions": projection.state.state_emotions.model_dump(),
        "relationship": projection.state.state_relationship.model_dump() if projection.state.state_relationship else None,
    }
    base["state_summary"] = projection.summary
    base["affect"] = affect_block
    reports["appraisal"] = {**reports["appraisal"], "affect": affect_block}

    call("context", {"triage": reports["triage"], "appraisal": reports["appraisal"]})
    # Bind model selection to original evidence. Generated paraphrases cannot
    # replace source text or become an independently remembered event.
    memory_by_id = {int(item["knoxel_id"]): item for item in request.get("memory_candidates", [])}
    selected_memories = []
    for key in reports["context"].get("candidate_ids", []):
        if int(key) not in memory_by_id:
            raise ValueError(f"Context selected memory outside the supplied snapshot: {key}")
        if memory_by_id[int(key)] not in selected_memories:
            selected_memories.append(memory_by_id[int(key)])
    reports["context"]["memories"] = [item["text"] for item in selected_memories]
    # Codelets read the accepted affect projection, not the previous state, so
    # an emotion that this event just caused can already activate them.
    # Codelet and scoring calls omit the public grounding block. Each is one
    # small question for a small model; the workspace and later stages still
    # receive that block before anything can reach public speech.
    call("codelets", {
        "public_context": None,
        "triage": reports["triage"],
        "appraisal": reports["appraisal"],
        "context": reports["context"],
        "state_signals": state_signals(projection.state),
    })
    call("attention", {
        "public_context": None,
        "triage": reports["triage"],
        "appraisal": reports["appraisal"],
        "context": reports["context"],
        "codelets": reports["codelets"],
        "candidates": _candidates(request, reports),
    })
    call("goals", {
        "appraisal": reports["appraisal"],
        "context": reports["context"],
        "current_goals": request.get("active_goals", []),
        "inner_proposals": inner_proposals(reports["attention"], GOAL_KINDS),
        "mental_proposals": request.get("mental_proposals", []),
    })
    call("workspace", {
        "public_context": None,
        "attention": reports["attention"],
        "goals": reports["goals"],
        "candidate_texts": _candidate_texts(request),
    })
    # Every accepted thought currently yields to action selection. Bounded
    # continuation must own the next step before this stage can be deferred.
    call("action_selection", {
        "workspace": reports["workspace"],
        "goals": reports["goals"],
        "context": reports["context"],
        "capabilities": request.get("capabilities", []),
        "available_workflows": request.get("available_workflows") or DEFAULT_WORKFLOWS,
        "inner_proposals": inner_proposals(reports["attention"], ACTION_KINDS),
    })

    # Game danger preempts slow regulation. The workspace stage validates its
    # own mode first; this override only fires after a valid report exists.
    if projection.regulation == REGULATION_REACTIVE:
        workspace_report = reports["workspace"]
        workspace_report["control_mode"] = "reactive"
        reason = str(workspace_report.get("frame_reason", "")).strip()
        workspace_report["frame_reason"] = (
            f"{reason} Danger preempts contemplation.".strip()
            if "preempt" not in reason.lower() else reason
        )

    dispatches = action_dispatches(reports["action_selection"].get("actions", []))
    dispatches = resolve_same_cycle(dispatches)
    dispatches = resolve_prerequisites(dispatches, dict(action_receipts or {}))

    workspace = reports["workspace"]
    attention = reports["attention"]
    origins = {
        str(int(item.get("knoxel_id"))): str(item.get("kind", ""))
        for item in request.get("stimuli", [])
    }
    return CognitiveReport(
        status="done",
        summary=f"ran {len(receipts)} stages, dispatched "
                f"{sum(1 for d in dispatches if d.status == 'dispatched')} actions",
        receipts=receipts,
        dispatches=dispatches,
        focus_ids=[int(item) for item in workspace.get("focus_ids", [])],
        winning_attention_ids=[int(item) for item in attention.get("selected_ids", [])],
        control_mode=str(workspace.get("control_mode", "reactive")),
        frame_reason=str(workspace.get("frame_reason", "")),
        subjective_experience=str(workspace.get("subjective_experience", "")),
        appraisal=_appraisal_from(reports.get("appraisal")),
        goal_changes=[dict(change) for change in reports.get("goals", {}).get("changes", [])],
        event_origins=origins,
        mental_state_latent_vec=projection.state.to_list(),
        affect_trace=projection.trace,
        affect_summary=projection.summary,
        regulation=projection.regulation,
        selected_memories=selected_memories,
        memory_trace={**request.get("memory_trace", {}),
                      "context_selected_ids": [item["knoxel_id"] for item in selected_memories]},
        belief_updates=reports["context"].get("belief_updates", []),
        nominations=[dict(item) for item in attention.get("nominations", [])],
        codelet_trace={key: value for key, value in reports["codelets"].items() if key != "proposals"}
        | {"competition": attention.get("competition", {})},
        focus_decision=attention.get("competition", {}).get("focus", {}),
        action_selection=dict(reports["action_selection"]),
        wait_request=next(({"seconds": item.get("arguments", {}).get("timeout_seconds",
                                request.get("wait_default_seconds", 1800)),
                            "reason": item.get("content") or "Wait for a new stimulus."}
                          for item in reports["action_selection"].get("actions", [])
                          if item.get("action_type") == "Sleep"), None),
    )


def _appraisal_from(report: dict | None) -> AppraisalResult | None:
    """Return a typed appraisal from a stage report, or None when absent."""
    if not report:
        return None
    return AppraisalResult.model_validate(report)


__all__ = ["ACTION_KINDS", "DEFAULT_WORKFLOWS", "GOAL_KINDS", "STAGE_ORDER", "action_dispatches",
           "inner_proposals", "proposal_consumers", "run_cycle", "stable_action_id"]
