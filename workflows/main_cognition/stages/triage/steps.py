"""Triage role.

Entry point: ``run(context)``. Builds the triage messages, validates the model
JSON into :class:`TriageResult`, and writes ``outbox/report.yaml``. The message
builder is pure and tested without a kernel, model, or Ghost.
"""
from __future__ import annotations

from pathlib import Path

from workflows.main_cognition.schemas import Stimulus, TriageResult
from workflows.main_cognition.stage_support import (
    complete_json, load_yaml, read_request, write_report,
)

WORKFLOW_DIR = Path(__file__).resolve().parent


def load_prompts(path: Path | None = None) -> dict:
    return load_yaml(path or WORKFLOW_DIR / "prompts.yaml")


def render_stimuli(stimuli: list[Stimulus]) -> str:
    lines = []
    for stimulus in stimuli:
        lines.append(f"- id={stimulus.knoxel_id} source={stimulus.source} "
                     f"kind={stimulus.kind} text={stimulus.text!r}")
    return "\n".join(lines) or "(none)"


def build_messages(request: dict, prompts: dict) -> list[dict]:
    stimuli = [Stimulus.model_validate(item) for item in request.get("stimuli", [])]
    system = prompts["system_prompt"].format(
        companion_name=request.get("companion_name", "Spamuel"),
    )
    user = prompts["user_template"].format(
        companion_name=request.get("companion_name", "Spamuel"),
        user_name=request.get("user_name", "Jesse"),
        state_summary=request.get("state_summary") or "(no recorded state update)",
        active_goals=", ".join(request.get("active_goals", [])) or "(none)",
        stimuli=render_stimuli(stimuli),
    )
    return [
        {"role": "system", "content": system.strip()},
        {"role": "user", "content": user.strip()},
    ]


def run(context) -> dict:
    request = read_request(context.runtime.environment)
    prompts = load_prompts()
    messages = build_messages(request, prompts)
    result = complete_json(context, messages, TriageResult, temperature=0.1)
    write_report(context.runtime.environment, result)
    return {
        "status": "done",
        "summary": f"triaged {len(result.items)} stimuli",
    }


__all__ = ["build_messages", "load_prompts", "render_stimuli", "run"]
