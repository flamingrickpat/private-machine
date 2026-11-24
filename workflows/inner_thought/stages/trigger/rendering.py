"""Private trigger role.

Entry point: ``run(context)``. Reads the fired triggers and the private
candidates the host derived, lets the model pick one operation, and writes the
validated ``TriggerChoice`` to ``outbox/report.yaml``. The message builder is
pure and tested without a kernel, model, or MemoryStore.
"""
from __future__ import annotations

from pm.llm.generation import GenerationSettings

from pathlib import Path
import json
from dataclasses import dataclass
from typing import Annotated, Literal
from pydantic import RootModel, BaseModel, ConfigDict, Field

from pm.constants import Sampler
from pm_workflows.python_role import RoleContext

from workflows.inner_thought.operations import OPERATIONS
from workflows.inner_thought.schemas import PrivateRoleResult
from workflows.inner_thought.schemas import InnerRequest, TriggerChoice
from workflows.inner_thought.schemas import PrivateAttentionContext, ThoughtAttentionDecision
from workflows.cognitive_control.attention import validate_private_attention
from pm.runs.role_io import load_yaml, read_request, write_report
from pm.llm.role_model import role_model
from pm.llm.structured import parse_response, JsonOutputError
from pydantic import ValidationError
from pm.context_render.prompt import RenderSettings, RenderedPrompt, RendererCapture, RendererInput
from workflows.inner_thought.prompt_selection import PrivatePromptInput, private_frame, private_source_blocks, select_private_evidence

WORKFLOW_DIR = Path(__file__).resolve().parent


class TriggerPrompts(BaseModel):
    """Required prompt texts for legacy triggers and separate private attention."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    system_prompt: str
    user_template: str
    attention_system_prompt: str
    attention_user_template: str


class AttentionStageRequest(BaseModel):
    """Frozen private sources beside the shared owner's actual competing context."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    stage_kind: Literal["private_attention"] = "private_attention"
    request: InnerRequest
    attention: PrivateAttentionContext


def load_prompts(path: Path | None = None) -> TriggerPrompts:
    """Read both selection contracts without defaulting missing prompt texts."""
    return load_yaml(path or WORKFLOW_DIR / "prompts.yaml", TriggerPrompts)


def _render_triggers(request: InnerRequest) -> str:
    lines: list[str] = []
    for candidate in request.trigger_candidates:
        allowed = ", ".join(name for name in candidate.operations if name in OPERATIONS) or "none"
        lines.append(f"- {candidate.name} (urgency {candidate.urgency:.2f}): {candidate.cause}. "
                     f"Allowed operations: {allowed}.")
    return "\n".join(lines) or "(none)"


@dataclass(frozen=True, slots=True)
class TriggerRenderInput:
    request: InnerRequest
    prompts: TriggerPrompts
    correction: str = ""


def render_trigger(request: InnerRequest, prompts: TriggerPrompts, settings: RenderSettings,
                   correction: str = "") -> RenderedPrompt:
    required = private_frame(request, settings) + "\n\nFired triggers:\n" + _render_triggers(request)
    required += f"\nReflection review due: {request.reflection.review_due}"
    return select_private_evidence(PrivatePromptInput("private trigger",
        prompts.system_prompt.format(companion_name=request.companion_name), required,
        private_source_blocks(request), correction), settings)


def render_private_attention(request: AttentionStageRequest, prompts: TriggerPrompts,
                             settings: RenderSettings, correction: str = "") -> RenderedPrompt:
    attention = request.attention
    lines = [private_frame(request.request, settings), "Shared attention: " + attention.attention_summary,
        "Blocked reason: " + attention.blocked_reason,
        f"Pending input IDs: {attention.pending_input_ids}; active intentions: {attention.active_intention_ids}"]
    focus = attention.accepted_focus
    if focus is not None:
        lines.append(f"Accepted focus: {focus.accept}; instruction: {focus.instruction}; "
            f"promote {focus.promote_ids}; reduce {focus.reduce_ids}; strength {focus.strength:.3f}; reason {focus.reason}")
    required_ids: set[int] = set()
    for offered in attention.candidates:
        nomination = offered.nomination
        judgments = nomination.judgments
        required_ids.update(nomination.evidence_ids)
        lines.append(f"Nomination {offered.nomination_id}: {nomination.text}\nQuestion: {nomination.question}\n"
            f"Reality: {nomination.reality}; evidence {nomination.evidence_ids}; original roots {offered.root_source_ids}; "
            f"operation {nomination.suggested_operation}.\nJudgments: interestingness {judgments.interestingness:.3f}; "
            f"character relevance {judgments.character_relevance:.3f}; plausibility {judgments.plausibility:.3f}; "
            f"consequence {judgments.consequence:.3f}; desirability {judgments.desirability:.3f}; "
            f"action suitability {judgments.action_suitability:.3f}. {judgments.reason}")
    for parked in attention.parked:
        required_ids.update(parked.available_evidence_ids)
        lines.append(f"Parked receipt {parked.receipt_id}: {parked.question}\nStop: {parked.stop_reason}; "
            f"deadline elapsed {parked.deadline_elapsed}; available evidence {parked.available_evidence_ids}")
    for displaced in attention.displaced:
        lines.append(f"Displaced receipt {displaced.receipt_id}: {displaced.question}\n"
            f"Reason: {displaced.displaced_reason}; earlier operations {displaced.earlier_operation_count}")
    lines.append("Delivered operations:\n" + "\n".join(f"{item.name}: {item.description}" for item in OPERATIONS.values()))
    lines.append("Output schema:\n" + json.dumps(ThoughtAttentionDecision.model_json_schema()))
    return select_private_evidence(PrivatePromptInput("private attention",
        prompts.attention_system_prompt.format(companion_name=request.request.companion_name), "\n\n".join(lines),
        private_source_blocks(request.request, required_ids=frozenset(required_ids)), correction), settings)


@dataclass(frozen=True, slots=True)
class AttentionRenderInput:
    request: AttentionStageRequest
    prompts: TriggerPrompts
    correction: str = ""


class TriggerEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    stage_kind: Literal["linear_trigger"] = "linear_trigger"
    request: InnerRequest


class TriggerStageRequest(RootModel[Annotated[AttentionStageRequest | TriggerEnvelope, Field(discriminator="stage_kind")]]):
    pass


def run(context: RoleContext) -> PrivateRoleResult:
    """Choose one of the deterministic trigger candidates, or none."""
    assert context.runtime is not None
    request = read_request(context.runtime.environment, TriggerStageRequest).root
    prompts = load_prompts()
    session = role_model(context.runtime)
    settings = session.render_settings()
    correction = ""
    attempt = 0
    if isinstance(request, AttentionStageRequest):
        selection = request
        while True:
            render_input = AttentionRenderInput(selection, prompts, correction)
            with RendererCapture(session.capture_settings("private.attention"),
                    RendererInput("private.attention", render_input, settings)) as capture:
                rendered = render_private_attention(selection, prompts, settings, correction)
                capture.save_output(rendered)
            raw = session.complete(rendered.turns, settings=GenerationSettings(temperature=Sampler.TempInstruct, enable_thinking=False))
            try:
                decision = validate_private_attention(selection.attention, parse_response(raw, ThoughtAttentionDecision))
            except (ValidationError, JsonOutputError, ValueError) as error:
                attempt += 1
                correction = str(error)
                context.runtime.environment.write_text(f"scratch/attention-rejected-{attempt:03}.txt", raw + "\n" + correction)
                continue
            break
        write_report(context.runtime.environment, decision)
        return {"status": "done", "summary": f"private attention {decision.disposition}: {decision.reason}"}
    parsed = request.request
    while True:
        render_input = TriggerRenderInput(parsed, prompts, correction)
        with RendererCapture(session.capture_settings("private.trigger"),
                RendererInput("private.trigger", render_input, settings)) as capture:
            rendered = render_trigger(parsed, prompts, settings, correction)
            capture.save_output(rendered)
        raw = session.complete(rendered.turns, settings=GenerationSettings(temperature=Sampler.TempInstruct, enable_thinking=False))
        try:
            result = parse_response(raw, TriggerChoice)
            known = {item.record_id for item in rendered.evidence if item.included}
            if set(result.candidate_ids) - known:
                raise ValueError("Trigger candidate_ids must name evidence selected for this prompt.")
        except (ValidationError, JsonOutputError, ValueError) as error:
            attempt += 1
            correction = str(error)
            context.runtime.environment.write_text(f"scratch/trigger-rejected-{attempt:03}.txt", raw + "\n" + correction)
            continue
        break
    write_report(context.runtime.environment, result)
    return {"status": "done", "summary": f"chose operation {result.operation or 'none'} from trigger {result.trigger or 'none'}"}


__all__ = ["AttentionStageRequest", "render_private_attention", "render_trigger", "load_prompts", "run"]
