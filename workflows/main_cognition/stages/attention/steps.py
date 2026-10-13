"""Attention role.

Entry point: ``run(context)``. Builds the attention messages, validates the
model JSON into :class:`AttentionResult`, rejects citations of knoxel ids that
were not supplied as candidates, and writes ``outbox/report.yaml``. The message
builder and the id guard are pure and tested without a kernel, model, or MemoryStore.
"""
from __future__ import annotations

import json
from pathlib import Path

from workflows.main_cognition.schemas import AttentionResult
from workflows.main_cognition.stage_support import (
    StageOutputError, complete_json, load_yaml, read_request, write_report,
)

WORKFLOW_DIR = Path(__file__).resolve().parent


def load_prompts(path: Path | None = None) -> dict:
    return load_yaml(path or WORKFLOW_DIR / "prompts.yaml")


def candidate_ids(candidates: list[dict]) -> set[int]:
    return {int(candidate["knoxel_id"]) for candidate in candidates}


def render_candidates(candidates: list[dict]) -> str:
    lines = []
    for candidate in candidates:
        lines.append(
            f"- id={candidate.get('knoxel_id')} kind={candidate.get('kind', '')} "
            f"text={candidate.get('text', '')!r}"
        )
    return "\n".join(lines) or "(none)"


def render_block(value) -> str:
    if value is None:
        return "(none)"
    return json.dumps(value, indent=2, sort_keys=False, default=str)


def ensure_candidate_ids(result: AttentionResult, allowed: set[int]) -> None:
    """Reject any id that was not in the supplied candidate list."""
    allowed = {int(candidate_id) for candidate_id in allowed}
    for nomination in result.nominations:
        for target_id in nomination.target_ids:
            if int(target_id) not in allowed:
                raise StageOutputError(
                    f"Attention nominated target id {target_id} that was not a candidate."
                )
    for selected_id in result.selected_ids:
        if int(selected_id) not in allowed:
            raise StageOutputError(
                f"Attention selected id {selected_id} that was not a candidate."
            )


def build_messages(request: dict, prompts: dict) -> list[dict]:
    candidates = list(request.get("candidates", []))
    system = prompts["system_prompt"].format(
        companion_name=request.get("companion_name", "Spamuel"),
    )
    user = prompts["user_template"].format(
        companion_name=request.get("companion_name", "Spamuel"),
        user_name=request.get("user_name", "Jesse"),
        state_summary=request.get("state_summary") or "(no recorded state update)",
        affect=render_block(request.get("affect")),
        active_goals=", ".join(request.get("active_goals", [])) or "(none)",
        triage=render_block(request.get("triage")),
        appraisal=render_block(request.get("appraisal")),
        context=render_block(request.get("context")),
        candidates=render_candidates(candidates),
    )
    return [
        {"role": "system", "content": system.strip()},
        {"role": "user", "content": user.strip()},
    ]


def run(context) -> dict:
    request = read_request(context.runtime.environment)
    prompts = load_prompts()
    messages = build_messages(request, prompts)
    result = complete_json(context, messages, AttentionResult, temperature=0.2)
    ensure_candidate_ids(result, candidate_ids(list(request.get("candidates", []))))
    write_report(context.runtime.environment, result)
    return {
        "status": "done",
        "summary": (
            f"nominated {len(result.nominations)} and selected "
            f"{len(result.selected_ids)} candidates"
        ),
    }


__all__ = [
    "build_messages",
    "candidate_ids",
    "ensure_candidate_ids",
    "load_prompts",
    "render_block",
    "render_candidates",
    "run",
]
