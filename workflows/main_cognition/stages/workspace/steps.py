"""Workspace role.

Entry point: ``run(context)``. Builds the workspace messages from the request,
its attention result, its goals result, and the candidate texts, validates the
model JSON into :class:`WorkspaceResult`, rejects focus ids outside the supplied
candidates (and outside the attention selection when one was made), rejects an
unknown control mode, and writes ``outbox/report.yaml``. The message builder and
the guards are pure and tested without a kernel, model, or MemoryStore.
"""
from __future__ import annotations

import json
from pathlib import Path

from workflows.main_cognition.schemas import WorkspaceResult
from workflows.main_cognition.stage_support import (
    StageOutputError, complete_json, load_yaml, read_request, write_report,
)

WORKFLOW_DIR = Path(__file__).resolve().parent

CONTROL_MODES = ("reactive", "reflective", "blended")


def load_prompts(path: Path | None = None) -> dict:
    return load_yaml(path or WORKFLOW_DIR / "prompts.yaml")


def iter_candidates(candidate_texts) -> list[tuple[int, str]]:
    """Yield ``(knoxel_id, text)`` pairs from a mapping or a list."""
    pairs: list[tuple[int, str]] = []
    if isinstance(candidate_texts, dict):
        for key, value in candidate_texts.items():
            if isinstance(value, dict):
                pairs.append((int(value.get("knoxel_id", key)), str(value.get("text", ""))))
            else:
                pairs.append((int(key), str(value)))
    else:
        for item in candidate_texts or []:
            pairs.append((int(item["knoxel_id"]), str(item.get("text", ""))))
    return pairs


def candidate_ids(candidate_texts) -> set[int]:
    return {knoxel_id for knoxel_id, _ in iter_candidates(candidate_texts)}


def allowed_focus_ids(request: dict) -> set[int]:
    """Candidate ids, narrowed to the attention selection when one exists."""
    allowed = candidate_ids(request.get("candidate_texts", []))
    attention = request.get("attention") or {}
    selected = {int(knoxel_id) for knoxel_id in attention.get("selected_ids", [])}
    if selected:
        allowed &= selected
    return allowed


def render_candidates(candidate_texts) -> str:
    lines = [
        f"- id={knoxel_id} text={text!r}"
        for knoxel_id, text in iter_candidates(candidate_texts)
    ]
    return "\n".join(lines) or "(none)"


def render_block(value) -> str:
    if value is None:
        return "(none)"
    return json.dumps(value, indent=2, sort_keys=False, default=str)


def ensure_valid_result(result: WorkspaceResult, allowed: set[int]) -> None:
    """Reject focus ids outside ``allowed`` and unknown control modes."""
    allowed = {int(knoxel_id) for knoxel_id in allowed}
    for focus_id in result.focus_ids:
        if int(focus_id) not in allowed:
            raise StageOutputError(
                f"Workspace focused on id {focus_id} that was not an allowed candidate."
            )
    if result.control_mode not in CONTROL_MODES:
        raise StageOutputError(
            f"Workspace chose control_mode {result.control_mode!r}; "
            f"expected one of {', '.join(CONTROL_MODES)}."
        )


ACCEPTED_LABELS = {
    "observed": "happening now",
    "remembered": "memory",
    "internal": "my own feeling or thought",
    "inferred": "guess",
    "imagined": "possibility",
}


def render_accepted(attention: dict | None) -> str:
    """List the winning nominations with their reality labels.

    Older attention reports have no status field; they render as ``(none)``.
    """
    lines = []
    for item in (attention or {}).get("nominations", []):
        if item.get("status") != "selected" or not str(item.get("text", "")).strip():
            continue
        label = ACCEPTED_LABELS.get(str(item.get("reality", "")), str(item.get("reality", "")))
        lines.append(f"- ({label}) {item['text']}")
    return "\n".join(lines) or "(none)"


def compact_attention(attention: dict | None) -> dict:
    """Keep the workspace prompt short: ids, status, and scores only."""
    attention = attention or {}
    return {
        "selected_ids": attention.get("selected_ids", []),
        "nominations": [{key: item.get(key) for key in
                         ("nomination_id", "target_ids", "status", "score", "reality")}
                        for item in attention.get("nominations", [])],
    }


def build_messages(request: dict, prompts: dict) -> list[dict]:
    companion_name = request.get("companion_name", "Spamuel")
    system = prompts["system_prompt"].format(companion_name=companion_name)
    user = prompts["user_template"].format(
        companion_name=companion_name,
        user_name=request.get("user_name", "Jesse"),
        state_summary=request.get("state_summary") or "(no recorded state update)",
        affect=render_block(request.get("affect")),
        active_goals=", ".join(request.get("active_goals", [])) or "(none)",
        accepted=render_accepted(request.get("attention")),
        attention=render_block(compact_attention(request.get("attention"))),
        goals=render_block(request.get("goals")),
        candidates=render_candidates(request.get("candidate_texts", [])),
    )
    user += "\nAllowed focus ids: " + json.dumps(sorted(allowed_focus_ids(request)))
    return [
        {"role": "system", "content": system.strip()},
        {"role": "user", "content": user.strip()},
    ]


def run(context) -> dict:
    request = read_request(context.runtime.environment)
    prompts = load_prompts()
    messages = build_messages(request, prompts)
    allowed = allowed_focus_ids(request)
    # No invalid frame reaches commit. Offer one correction for a model that
    # selected an ID outside the accepted attention boundary.
    for attempt in range(2):
        result = complete_json(context, messages, WorkspaceResult, temperature=0.7, enable_thinking=False)
        try:
            ensure_valid_result(result, allowed)
            break
        except StageOutputError:
            if attempt == 1:
                raise
            messages.extend([
                {"role": "assistant", "content": result.model_dump_json()},
                {"role": "user", "content": "This frame uses an invalid focus ID or control mode. "
                 f"Use only focus IDs from {sorted(allowed)} and control modes from {CONTROL_MODES}."},
            ])
    write_report(context.runtime.environment, result)
    return {
        "status": "done",
        "summary": (
            f"assembled a {result.control_mode} frame over "
            f"{len(result.focus_ids)} focus ids"
        ),
    }


__all__ = [
    "ACCEPTED_LABELS",
    "CONTROL_MODES",
    "compact_attention",
    "render_accepted",
    "allowed_focus_ids",
    "build_messages",
    "candidate_ids",
    "ensure_valid_result",
    "iter_candidates",
    "load_prompts",
    "render_block",
    "render_candidates",
    "run",
]
