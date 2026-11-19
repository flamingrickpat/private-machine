"""Private integration role.

Entry point: ``run(context)``. Reads the performed thought and proposes its
accepted consequences as a validated ``IntegrationResult``. The message
builder is pure and tested without a kernel, model, or MemoryStore.

A request that carries ``focus_advice`` ends deliberate mental work. Only then
does the prompt offer the optional ``focus`` field. A request that carries
``perspective_revision`` ends a perspective inquiry; only then does the prompt
offer the optional entity-model revision. Spontaneous integrations see neither.
See ``workflows/inner_thought/host.py`` for the loop overview.
"""
from __future__ import annotations

from pm.llm.generation import GenerationSettings

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from pm.constants import Sampler
from pm_workflows.python_role import RoleContext

from workflows.inner_thought.perspective import PerspectiveRevisionRequest
from workflows.inner_thought.schemas import PrivateRoleResult
from workflows.inner_thought.schemas import (
    FocusAdviceRequest, GraphOutcomeRules, InnerRequest, IntegrationResult, ThoughtResult, TriggerChoice,
)
from workflows.inner_thought.pipeline import (
    validate_graph_outcome_rules, validate_integration,
)
from pm.runs.role_io import load_yaml, read_request, write_report
from pm.llm.role_model import role_model
from pm.llm.structured import parse_response, JsonOutputError
from pydantic import ValidationError
from pm.context_render.prompt import RenderSettings, RenderedPrompt, RendererCapture, RendererInput
from workflows.inner_thought.prompt_selection import PrivatePromptInput, private_frame, private_source_blocks, select_private_evidence
from workflows.inner_thought.request_views import OperationSuggestion

WORKFLOW_DIR = Path(__file__).resolve().parent


class IntegrationPrompts(BaseModel):
    """The prompt texts of the integrate stage, read from ``prompts.yaml``."""

    model_config = ConfigDict(extra="forbid")
    system_prompt: str = Field(description="System turn. It states the rules and the JSON schema of every integration.")
    user_template: str = Field(description="User turn with the performed thought, its evidence, goals, and expectations.")
    focus_advice_template: str = Field(description="Appended to the user turn only when the thought ends deliberate mental work.")
    perspective_revision_template: str = Field(
        description="Appended to the user turn only at the outcome of a perspective inquiry.")


def load_prompts(path: Path | None = None) -> IntegrationPrompts:
    """Read and validate the stage prompts. A missing or unknown key raises."""
    return load_yaml(path or WORKFLOW_DIR / "prompts.yaml", IntegrationPrompts)


class IntegrationEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    request: InnerRequest
    trigger: TriggerChoice = Field(default_factory=TriggerChoice)
    thought: ThoughtResult = Field(default_factory=ThoughtResult)
    operation: OperationSuggestion | None = None
    focus_advice: FocusAdviceRequest | None = None
    perspective_revision: PerspectiveRevisionRequest | None = None
    graph_outcome_rules: GraphOutcomeRules | None = None


@dataclass(frozen=True, slots=True)
class IntegrationRenderInput:
    request: IntegrationEnvelope
    prompts: IntegrationPrompts
    correction: str = ""


def render_integration(request: IntegrationEnvelope, prompts: IntegrationPrompts, settings: RenderSettings,
                       correction: str = "") -> RenderedPrompt:
    thought = request.thought
    required = (private_frame(request.request) + "\n\nPerformed private thought:\n" + thought.thought +
        "\nReality: " + thought.reality + "\nEvidence summary: " + thought.evidence_summary +
        "\nInterpretation: " + thought.interpretation + f"\nCited evidence: {thought.evidence_ids}" +
        "\nOperation: " + (request.operation.name if request.operation is not None else request.trigger.operation))
    pinned = set(thought.evidence_ids)
    advice = request.focus_advice
    if advice is not None:
        pinned.update(advice.citable_ids)
        required += "\n\n" + prompts.focus_advice_template.format(companion_name=request.request.companion_name,
            instruction=advice.instruction, citable_ids=", ".join(str(key) for key in advice.citable_ids))
    perspective = request.perspective_revision
    if perspective is not None:
        pinned.update(perspective.citable_source_ids)
        claims = "\n".join(f"[{item.claim_id}; {item.claim_key}] {item.statement}; sources {item.source_ids}"
            for item in perspective.existing_claims)
        required += "\n\n" + prompts.perspective_revision_template.format(subject=perspective.subject_name,
            citable_source_ids=", ".join(str(key) for key in perspective.citable_source_ids), existing_claims=claims)
    return select_private_evidence(PrivatePromptInput("private integration",
        prompts.system_prompt.format(companion_name=request.request.companion_name), required,
        private_source_blocks(request.request, required_ids=frozenset(pinned)), correction), settings)


def run(context: RoleContext) -> PrivateRoleResult:
    assert context.runtime is not None
    request = read_request(context.runtime.environment, IntegrationEnvelope)
    prompts = load_prompts()
    session = role_model(context.runtime)
    settings = session.render_settings()
    correction = ""
    attempt = 0
    while True:
        render_input = IntegrationRenderInput(request, prompts, correction)
        with RendererCapture(session.capture_settings("private.integrate"),
                RendererInput("private.integrate", render_input, settings)) as capture:
            rendered = render_integration(request, prompts, settings, correction)
            capture.save_output(rendered)
        raw = session.complete(rendered.turns, settings=GenerationSettings(temperature=Sampler.TempInstruct, enable_thinking=False))
        known = {item.record_id for item in rendered.evidence if item.included}
        if request.request.main_frame.frame_id is not None:
            known.add(request.request.main_frame.frame_id)
        source_texts = {item.knoxel_id: item.text for item in request.request.candidates if item.knoxel_id in known}
        try:
            result = parse_response(raw, IntegrationResult)
            validate_integration(result, known, request.request, focus_requested=request.focus_advice is not None,
                perspective=request.perspective_revision,
                source_texts=source_texts if request.perspective_revision is not None else None)
            if request.graph_outcome_rules is not None:
                validate_graph_outcome_rules(result, request.graph_outcome_rules, known)
        except (ValidationError, JsonOutputError, ValueError) as error:
            attempt += 1
            correction = str(error)
            context.runtime.environment.write_text(f"scratch/integration-rejected-{attempt:03}.txt", raw + "\n" + correction)
            continue
        break
    write_report(context.runtime.environment, result)
    return {"status": "done", "summary": result.summary or "integrated the private thought"}
