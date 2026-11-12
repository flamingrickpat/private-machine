"""Propose private questions without allocating attention or executing thought.

See :mod:`workflows.inner_thought.host` for the private subsystem boundary.
The host supplies frozen evidence. This child generates at most three typed
proposals, including a valid empty batch. Its output grants no GWT, action,
evidence, or public-speech authority. The host retains the stage receipt.
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict
from pm.constants import Sampler
from pm.context_render.turns import Role, Turn
from pm_workflows.python_role import RoleContext

from workflows.inner_thought.operations import OPERATIONS
from workflows.inner_thought.pipeline import citable_ids
from workflows.inner_thought.schemas import InnerRequest, NominationBatch
from workflows.main_cognition.model_json import complete_json, load_yaml, read_request, write_report

WORKFLOW_DIR = Path(__file__).resolve().parent


class NominationPrompts(BaseModel):
    """Required texts of the nomination child prompt file."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    system_prompt: str
    user_template: str


def load_prompts(path: Path | None = None) -> NominationPrompts:
    """Read required prompt texts without replacing a missing configuration."""
    return NominationPrompts.model_validate(load_yaml(path or WORKFLOW_DIR / "prompts.yaml"))


def build_messages(request: InnerRequest, prompts: NominationPrompts) -> list[Turn]:
    """Present frozen sources and independent judgments before nomination."""
    operations = [f"{operation.name}: {operation.description}" for operation in OPERATIONS.values()]
    system = prompts.system_prompt.format(companion_name=request.companion_name)
    user = prompts.user_template.format(
        request=request.model_dump_json(),
        operations="\n".join(operations),
        schema=json.dumps(NominationBatch.model_json_schema(), ensure_ascii=False),
    )
    return [Turn(Role.System, system.strip()), Turn(Role.User, user.strip())]


def validate_nominations(request: InnerRequest, result: NominationBatch) -> NominationBatch:
    """Keep proposal citations inside supplied evidence and delivered operations."""
    known = citable_ids(request)
    for nomination in result.nominations:
        if nomination.suggested_operation not in OPERATIONS:
            raise ValueError(f"A nomination suggests an undelivered operation: {nomination.suggested_operation}")
        invalid = set(nomination.evidence_ids) - known
        if invalid:
            raise ValueError(f"A nomination cites evidence outside the frozen request: {sorted(invalid)}")
    return result


def run(context: RoleContext) -> dict[str, str]:
    """Generate a bounded batch; the caller separately accepts attention."""
    assert context.runtime is not None
    request = InnerRequest.model_validate(read_request(context.runtime.environment))
    messages = build_messages(request, load_prompts())

    def validate(payload: dict[str, object], raw: str) -> NominationBatch:
        """Keep the original evidence contract during structured-output repairs."""
        return validate_nominations(request, NominationBatch.model_validate(payload))

    result = complete_json(context, messages, NominationBatch, temperature=Sampler.TempInstruct,
                           enable_thinking=False, validate=validate)
    write_report(context.runtime.environment, result)
    return {"status": "done", "summary": f"proposed {len(result.nominations)} private questions: {result.reason}"}
