"""Appraisal role.

Entry point: ``run(context)``. Builds the appraisal messages from the request
and its triage result, validates the model JSON into :class:`AppraisalResult`,
and writes ``outbox/report.yaml``. The message builder is pure and tested
without a kernel, model, or MemoryStore.
"""
from __future__ import annotations

from pathlib import Path
from pm.constants import Sampler
from pm.context_render.turns import Role, Turn, format_content

from workflows.main_cognition.schemas import AppraisalResult
from workflows.main_cognition.stage_support import (
    complete_json, load_yaml, read_request, write_report,
)

WORKFLOW_DIR = Path(__file__).resolve().parent


def load_prompts(path: Path | None = None) -> dict:
    return load_yaml(path or WORKFLOW_DIR / "prompts.yaml")


def render_triage(request: dict) -> str:
    """Render each triage item with its reasoning and the matching stimulus text."""
    stimuli = {
        item.get("knoxel_id"): item.get("text", "")
        for item in request.get("stimuli", [])
    }
    triage = request.get("triage") or {}
    lines = []
    for item in triage.get("items", []):
        knoxel_id = item.get("knoxel_id")
        lines.append(
            f"- id={knoxel_id} category={item.get('category')} "
            f"input_kind={item.get('input_kind')} "
            f"urgency={item.get('urgency')} relevance={item.get('relevance')} "
            f"text={stimuli.get(knoxel_id, '')!r} "
            f"reasoning={item.get('reasoning', '')!r}"
        )
    return "\n".join(lines) or "(none)"


def build_messages(request: dict, prompts: dict) -> list[Turn]:
    """Build typed turns from the selected triage and current stage input."""
    companion_name = request.get("companion_name", "Spamuel")
    system = prompts["system_prompt"].format(companion_name=companion_name)
    user = format_content(prompts["user_template"],
        companion_name=companion_name,
        user_name=request.get("user_name", "Jesse"),
        state_summary=request.get("state_summary") or "(no recorded state update)",
        active_goals=", ".join(request.get("active_goals", [])) or "(none)",
        triage=render_triage(request),
    )
    return [
        Turn(Role.System, system.strip()),
        Turn(Role.User, user.strip()),
    ]


def run(context) -> dict:
    request = read_request(context.runtime.environment)
    prompts = load_prompts()
    messages = build_messages(request, prompts)
    result = complete_json(context, messages, AppraisalResult, temperature=Sampler.TempInstruct, enable_thinking=False)
    write_report(context.runtime.environment, result)
    return {
        "status": "done",
        "summary": (
            f"appraised {len(result.evidence_ids)} evidence ids; "
            f"feeling: {result.feeling}"
        ),
    }


__all__ = ["build_messages", "load_prompts", "render_triage", "run"]
