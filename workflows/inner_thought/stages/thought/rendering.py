"""Private thought role.

Entry point: ``run(context)``. Performs the chosen operation over the frozen
private candidates and writes the validated ``ThoughtResult``. The message
builder is pure and tested without a kernel, model, or MemoryStore.
"""
from __future__ import annotations

from pm.llm.generation import GenerationSettings

from pathlib import Path
from typing import Annotated, Literal
import json
from dataclasses import dataclass
from workflows.inner_thought.rendering import inspection as render_inspection

from pm.constants import Sampler
from pm_workflows.python_role import RoleContext

from workflows.inner_thought.operations import validate_operation
from workflows.inner_thought.schemas import PrivateRoleResult
from workflows.inner_thought.schemas import InnerRequest, ThoughtResult, TriggerChoice
from pm.runs.role_io import load_yaml, read_request, write_report
from pm.llm.role_model import role_model
from pm.llm.structured import parse_response, JsonOutputError
from pydantic import BaseModel, ConfigDict, RootModel, ValidationError, Field
from pm.context_render.prompt import RenderSettings, RenderedPrompt, RendererCapture, RendererInput
from workflows.inner_thought.prompt_selection import PrivatePromptInput, PrivateEvidenceBlock, private_frame, private_source_blocks, select_private_evidence
from workflows.inner_thought.request_views import OperationSuggestion

WORKFLOW_DIR = Path(__file__).resolve().parent


class ThoughtPrompts(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    system_prompt: str
    user_template: str


def load_prompts(path: Path | None = None) -> ThoughtPrompts:
    return load_yaml(path or WORKFLOW_DIR / "prompts.yaml", ThoughtPrompts)


class ThoughtEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    stage_kind: Literal["linear_thought"] = "linear_thought"
    request: InnerRequest
    trigger: TriggerChoice = Field(default_factory=TriggerChoice)
    operation: OperationSuggestion | None = None


@dataclass(frozen=True, slots=True)
class ThoughtRenderInput:
    request: ThoughtEnvelope
    prompts: ThoughtPrompts
    correction: str = ""


def render_thought(request: ThoughtEnvelope, prompts: ThoughtPrompts, settings: RenderSettings,
                   correction: str = "") -> RenderedPrompt:
    trigger = request.trigger
    operation = validate_operation(request.operation.name if request.operation is not None else trigger.operation or "none")
    system = prompts.system_prompt.format(companion_name=request.request.companion_name,
        operation_name=operation.name, operation_group=operation.group, operation_mode=operation.mode,
        operation_focus=operation.temporal_focus, operation_description=operation.description)
    required = private_frame(request.request) + "\nTrigger reason: " + trigger.reason
    required += "\nAvailable next operations:\n" + "\n".join(item.name for item in request.request.operations)
    blocks = list(private_source_blocks(request.request, required_ids=frozenset(trigger.candidate_ids)))
    if request.request.inspection is not None:
        blocks.append(PrivateEvidenceBlock("Implementation inspection", render_inspection(request.request.inspection), 12))
    return select_private_evidence(PrivatePromptInput("private thought", system, required, tuple(blocks), correction), settings)


def render_graph_thought(request: GraphThoughtRequest, settings: RenderSettings,
                         correction: str = "") -> RenderedPrompt:
    from workflows.inner_thought.graph_execution import graph_thought_instruction
    instruction_input = GraphThoughtRequest(
        stage_kind=request.stage_kind,
        request=request.request,
        state=request.state,
        grant=request.grant,
        origin=request.origin,
        nodes=request.nodes,
        earlier_artifacts=request.earlier_artifacts,
        recall=request.recall,
        reflection_evidence=None,
        inspection_evidence=None,
        deliberate_budget=request.deliberate_budget,
        perspective_evidence=request.perspective_evidence,
        activity_evidence=request.activity_evidence,
        original_source_ids=request.original_source_ids,
    )
    system = graph_thought_instruction(instruction_input)
    state = request.state
    origin = request.origin
    required = (private_frame(request.request) + f"\n\nQuestion: {origin.question}\nOrigin: {origin.text}\n"
        f"Episode {state.episode_id}; sequence {state.sequence}; status {state.status}; stalled steps {state.stalled_steps}; "
        f"pending generated nodes {state.pending_generation_ids}; frontier {state.frontier_ids}.\n"
        f"Observer {request.grant.observer_entity_id}; reality {request.grant.scope.reality_id}; frame {request.grant.scope.frame}.\n"
        "Output schema:\n" + json.dumps(InquiryStep.model_json_schema()))
    pinned = set(origin.evidence_ids)
    blocks = list(private_source_blocks(request.request, required_ids=frozenset(pinned)))
    required_nodes = set(state.frontier_ids) | set(state.pending_generation_ids)
    if state.node_ids:
        required_nodes.add(state.node_ids[-1])
    for node in reversed(request.nodes):
        blocks.append(PrivateEvidenceBlock(f"Accepted graph node {node.record_id}",
            f"Operation {node.operation}; reality {node.reality}; parents {node.parent_record_ids}; "
            f"sources {node.source_event_ids}\n{node.text}", 20, (node.record_id,), node.record_id in required_nodes))
    for node in reversed(request.earlier_artifacts):
        blocks.append(PrivateEvidenceBlock(f"Earlier private artifact {node.record_id}; not corroboration",
            f"Operation {node.operation}; reality {node.reality}; sources {node.source_event_ids}\n{node.text}",
            3, (node.record_id,)))
    if request.recall is not None:
        for match in request.recall.matches:
            blocks.append(PrivateEvidenceBlock(f"Recalled evidence {match.knoxel_id}", match.text,
                10 + match.score, (match.knoxel_id,)))
    if request.reflection_evidence is not None:
        blocks.append(PrivateEvidenceBlock("Selected reflection evidence", request.reflection_evidence, 9))
    if request.inspection_evidence is not None:
        blocks.append(PrivateEvidenceBlock("Implementation inspection", request.inspection_evidence, 9))
    return select_private_evidence(PrivatePromptInput("private inquiry operation", system, required,
        tuple(blocks), correction), settings)


from workflows.inner_thought.graph_execution import GraphThoughtRequest, validate_graph_step
from workflows.inner_thought.schemas import InquiryStep


@dataclass(frozen=True, slots=True)
class GraphRenderInput:
    request: GraphThoughtRequest
    correction: str = ""


class ThoughtStageRequest(RootModel[Annotated[GraphThoughtRequest | ThoughtEnvelope, Field(discriminator="stage_kind")]]):
    pass


def run(context: RoleContext) -> PrivateRoleResult:
    assert context.runtime is not None
    request = read_request(context.runtime.environment, ThoughtStageRequest).root
    session = role_model(context.runtime)
    settings = session.render_settings()
    correction = ""
    attempt = 0
    if isinstance(request, GraphThoughtRequest):
        parsed_graph = request
        while True:
            render_input = GraphRenderInput(parsed_graph, correction)
            with RendererCapture(session.capture_settings("private.graph"),
                    RendererInput("private.graph", render_input, settings)) as capture:
                rendered = render_graph_thought(parsed_graph, settings, correction)
                capture.save_output(rendered)
            raw = session.complete(rendered.turns, settings=GenerationSettings(temperature=Sampler.TempReasoning))
            try:
                graph_result = validate_graph_step(parsed_graph, parse_response(raw, InquiryStep))
                known = {item.record_id for item in rendered.evidence if item.included}
                if parsed_graph.request.main_frame.frame_id is not None:
                    known.add(parsed_graph.request.main_frame.frame_id)
                cited = set(graph_result.evidence_ids)
                for alternative in graph_result.alternatives:
                    cited.update(alternative.evidence_ids)
                if cited - known:
                    raise ValueError("Inquiry citations must name evidence selected for this prompt.")
            except (ValidationError, JsonOutputError, ValueError) as error:
                attempt += 1
                correction = str(error)
                context.runtime.environment.write_text(f"scratch/inquiry-rejected-{attempt:03}.txt", raw + "\n" + correction)
                continue
            break
        write_report(context.runtime.environment, graph_result)
        return {"status": "done", "summary": graph_result.reason}
    parsed = request
    prompts = load_prompts()
    while True:
        render_input = ThoughtRenderInput(parsed, prompts, correction)
        with RendererCapture(session.capture_settings("private.thought"),
                RendererInput("private.thought", render_input, settings)) as capture:
            rendered = render_thought(parsed, prompts, settings, correction)
            capture.save_output(rendered)
        raw = session.complete(rendered.turns, settings=GenerationSettings(temperature=Sampler.TempReasoning))
        try:
            result = parse_response(raw, ThoughtResult)
            known = {item.record_id for item in rendered.evidence if item.included}
            if set(result.evidence_ids) - known:
                raise ValueError("Thought citations must name evidence selected for this prompt.")
        except (ValidationError, JsonOutputError, ValueError) as error:
            attempt += 1
            correction = str(error)
            context.runtime.environment.write_text(f"scratch/thought-rejected-{attempt:03}.txt", raw + "\n" + correction)
            continue
        break
    write_report(context.runtime.environment, result)
    return {"status": "done", "summary": f"performed thought with reality {result.reality}"}
