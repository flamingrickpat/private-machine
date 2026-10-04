"""Pure orchestration of the seven cognitive stages.

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
    CognitiveReport,
    StageReceipt,
)

STAGE_ORDER = (
    "triage",
    "appraisal",
    "context",
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
    {"name": "minecraft", "description": "Observe and act in the Minecraft world once."},
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


def _candidate_texts(request: dict) -> list[dict]:
    items = [dict(item) for item in request.get("memory_candidates", [])]
    for stimulus in request.get("stimuli", []):
        items.append({"knoxel_id": int(stimulus["knoxel_id"]),
                      "text": str(stimulus.get("text", ""))})
    return items


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


def run_cycle(
    request: dict,
    run_stage: Callable[[str, dict], dict],
    action_receipts: dict[str, str] | None = None,
) -> CognitiveReport:
    """Run all seven stages in order and return the typed parent report.

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
    call("context", {"triage": reports["triage"], "appraisal": reports["appraisal"]})
    call("attention", {
        "triage": reports["triage"],
        "appraisal": reports["appraisal"],
        "context": reports["context"],
        "candidates": _candidates(request, reports),
    })
    call("goals", {
        "appraisal": reports["appraisal"],
        "context": reports["context"],
        "current_goals": request.get("active_goals", []),
    })
    call("workspace", {
        "attention": reports["attention"],
        "goals": reports["goals"],
        "candidate_texts": _candidate_texts(request),
    })
    call("action_selection", {
        "workspace": reports["workspace"],
        "goals": reports["goals"],
        "capabilities": request.get("capabilities", []),
        "available_workflows": request.get("available_workflows") or DEFAULT_WORKFLOWS,
    })

    dispatches: list[ActionDispatch] = []
    for index, intent in enumerate(reports["action_selection"].get("actions", [])):
        dispatches.append(ActionDispatch(
            action_id=str(intent.get("action_id") or stable_action_id(intent, index)),
            workflow_name=str(intent.get("workflow_name", "")),
            action_type=str(intent.get("action_type", "Reply")),
            content=str(intent.get("content", "")),
            arguments=dict(intent.get("arguments", {})),
            depends_on=list(intent.get("depends_on", [])),
            requires_completed=list(intent.get("requires_completed", [])),
        ))
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
    )


def _appraisal_from(report: dict | None):
    """Return a typed appraisal from a stage report, or None when absent."""
    if not report:
        return None
    from workflows.main_cognition.schemas import AppraisalResult
    try:
        return AppraisalResult.model_validate(report)
    except Exception:
        return None


__all__ = ["DEFAULT_WORKFLOWS", "STAGE_ORDER", "run_cycle", "stable_action_id"]
