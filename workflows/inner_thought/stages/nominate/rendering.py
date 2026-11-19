"""Propose private questions without allocating attention or executing thought.

See :mod:`workflows.inner_thought.host` for the private subsystem boundary.
The host supplies frozen evidence. This child generates at most three typed
proposals, including a valid empty batch. Its output grants no GWT, action,
evidence, or public-speech authority. The host retains the stage receipt.
"""
from __future__ import annotations

from pm.llm.generation import GenerationSettings

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict
from pm.constants import Sampler
from pm_workflows.python_role import RoleContext

from workflows.inner_thought.operations import OPERATIONS
from workflows.inner_thought.pipeline import citable_ids
from workflows.inner_thought.schemas import PrivateRoleResult
from workflows.inner_thought.schemas import InnerRequest, NominationBatch
from pm.runs.role_io import load_yaml, read_request, write_report
from pm.llm.role_model import role_model
from pm.llm.structured import parse_response, JsonOutputError
from pydantic import ValidationError
from pm.context_render.prompt import RenderSettings, RenderedPrompt, RendererCapture, RendererInput
from workflows.inner_thought.prompt_selection import PrivatePromptInput, private_frame, private_source_blocks, select_private_evidence

WORKFLOW_DIR = Path(__file__).resolve().parent


class NominationPrompts(BaseModel):
    """Required texts of the nomination child prompt file."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    system_prompt: str
    user_template: str


def load_prompts(path: Path | None = None) -> NominationPrompts:
    """Read required prompt texts without replacing a missing configuration."""
    return load_yaml(path or WORKFLOW_DIR / "prompts.yaml", NominationPrompts)


@dataclass(frozen=True, slots=True)
class NominationRenderInput:
    request: InnerRequest
    prompts: NominationPrompts
    correction: str = ""


def render_nomination(request: InnerRequest, prompts: NominationPrompts, settings: RenderSettings,
                      correction: str = "") -> RenderedPrompt:
    operations = "\n".join(f"{operation.name}: {operation.description}" for operation in OPERATIONS.values())
    required = prompts.user_template.format(request=private_frame(request), operations=operations,
        schema=json.dumps(NominationBatch.model_json_schema(), ensure_ascii=False))
    return select_private_evidence(PrivatePromptInput("private nomination",
        prompts.system_prompt.format(companion_name=request.companion_name), required,
        private_source_blocks(request), correction), settings)


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


def run(context: RoleContext) -> PrivateRoleResult:
    """Generate a bounded batch; the caller separately accepts attention."""
    assert context.runtime is not None
    request = read_request(context.runtime.environment, InnerRequest)
    session = role_model(context.runtime)
    settings = session.render_settings()
    prompts = load_prompts()
    correction = ""
    attempt = 0
    while True:
        render_input = NominationRenderInput(request, prompts, correction)
        with RendererCapture(session.capture_settings("private.nominate"),
                RendererInput("private.nominate", render_input, settings)) as capture:
            rendered = render_nomination(request, prompts, settings, correction)
            capture.save_output(rendered)
        raw = session.complete(rendered.turns, settings=GenerationSettings(temperature=Sampler.TempInstruct, enable_thinking=False))
        try:
            result = parse_response(raw, NominationBatch)
            known = {item.record_id for item in rendered.evidence if item.included}
            if request.main_frame.frame_id is not None:
                known.add(request.main_frame.frame_id)
            validate_nominations(request, result)
            for nomination in result.nominations:
                if set(nomination.evidence_ids) - known:
                    raise ValueError("A nomination cites evidence not selected for this prompt.")
        except (ValidationError, JsonOutputError, ValueError) as error:
            attempt += 1
            correction = str(error)
            context.runtime.environment.write_text(f"scratch/nomination-rejected-{attempt:03}.txt", raw + "\n" + correction)
            continue
        break
    write_report(context.runtime.environment, result)
    return {"status": "done", "summary": f"proposed {len(result.nominations)} private questions: {result.reason}"}
