"""Non-kernel orchestration for standalone Story dialogue.

This module builds the generation tool, opens a read-only MemoryStore environment,
invokes the real kernel runner, reads the typed report, and (on request)
commits the accepted dialogue. It also runs the post-generation evaluator.

Generation and evaluation are separate phases on purpose: every generation for
a batch finishes before the evaluator model is activated. The evaluator reads
saved reports and never rewrites a candidate.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from uuid import uuid4
from pathlib import Path
from pydantic import BaseModel, JsonValue

from pm.constants import Sampler
from pm.utils import yaml_util

from pm.memory.store import MemoryStore
from pm.records.memory_records import Feature
from pm.records.enums import FeatureType, InterlocusType
from pm.llm.activation import ChatClient
from pm.llm.session import ModelSession
from pm.llm.generation import GenerationSettings
from pm.llm.stage_model import LazyStageModel
from pm.runs.role_runtime import PMWorkflowRuntime
from pm.context_render.prompt import (RenderSettings, RenderedPrompt, RendererInput, RendererCapture,
    RendererCaptureOptions, RendererCaptureSettings, require_rendered_prompt)
from pm.context_render.correction import render_regeneration_feedback
from pm.context_render.turns import Role, Turn
from threading import Event
from pm.runs.workflow_runner import WorkflowRequest, WorkflowResult, WorkflowRunner, WorkflowStatus
from workflows.dialogue_story.schemas import StoryQualityScore, StoryReport, StoryRequest, StorySituation, EvaluationPrompts
from workflows.public_speech.speech_text import reject_classifier_report
from pm.memory.bash_view import MemoryBashEnvironment

WORKFLOW_DIR = Path(__file__).resolve().parent
MANIFEST = WORKFLOW_DIR / "dialogue_story.workflow.md"
PROJECT_ROOT = WORKFLOW_DIR.parent.parent


def load_situation(path: str | Path) -> StorySituation:
    return StorySituation.model_validate(yaml_util.safe_load(Path(path).read_text(encoding="utf-8")))


def load_evaluation_prompts(path: str | Path) -> EvaluationPrompts:
    return EvaluationPrompts.model_validate(yaml_util.safe_load(Path(path).read_text(encoding="utf-8")))


def build_story_session(client: ChatClient, cancel: Event) -> ModelSession:
    model = LazyStageModel(client.connection, client.activation)
    session = ModelSession(model, cancel)
    session.render_settings()
    return session


def make_environment(ghost: MemoryStore) -> MemoryBashEnvironment:
    """Return a read-only working environment over the current store state."""
    view = ghost.snapshot()
    return MemoryBashEnvironment(view.all_knoxels, view.gwt_turns, "read")


def generate_story(
    ghost: MemoryStore,
    environment: MemoryBashEnvironment,
    client: ChatClient,
    request: StoryRequest,
    runtime_root: Path,
    run_id: str,
    *,
    resume: bool = False,
) -> tuple[WorkflowResult, StoryReport]:
    """Run the Story workflow once and return its result and typed report.

    ``resume`` continues a parked kernel rather than replaying generation. A
    run that did not complete raises RuntimeError with its kernel summary;
    no partial report is returned as an answer.
    """
    cancel = threading.Event()
    runtime = PMWorkflowRuntime(
        environment=environment,
        stop_event=cancel,
        tools={},
        model_session=build_story_session(client, cancel),
        propagate_python_errors=True,
    )
    runner = WorkflowRunner(MANIFEST, runtime_root, namespace="dialogue_story", base_dir=PROJECT_ROOT)
    result = runner.invoke(WorkflowRequest(run_id, request.model_dump(mode="json")), runtime, resume=resume)
    if result.status is not WorkflowStatus.Completed:
        raise RuntimeError(f"Story run {run_id} did not complete: {result.kernel_summary}")
    report = StoryReport.model_validate(result.report)
    return result, report


class StoryCommitReceipt(BaseModel):
    """Identify the accepted standalone Story utterance in its saved checkpoint."""

    commit_id: str
    knoxel_id: int


def commit_accepted_dialogue(ghost: MemoryStore, candidate: str, companion_name: str,
                            commit_id: str) -> StoryCommitReceipt:
    """Commit one accepted reply as public dialogue memory.

    Called only for an accepted candidate, after generation. Trials never call
    this, so independent trials start from the same initial state.
    """
    reject_classifier_report(candidate)
    batch = ghost.begin_batch(commit_id, new_tick=True)
    feature = batch.create(Feature(
        content=candidate,
        feature_type=FeatureType.Dialogue,
        interlocus=InterlocusType.Public,
        causal=True,
        override_source=companion_name,
        metadata={"memory_kind": "utterance", "factual_promotion": False},
    ))
    receipt = ghost.commit(batch)
    return StoryCommitReceipt(commit_id=receipt.commit_id, knoxel_id=feature.id)


# --------------------------------------------------------------- evaluation

@dataclass(frozen=True, slots=True)
class StoryEvaluationInput:
    companion_name: str
    current_user_message: str
    candidate: str
    prompts: EvaluationPrompts
    prior_dialogue: str = ""
    memories: str = ""
    problem: str = ""
    rejected_answer: str = ""


def render_story_evaluation(input: StoryEvaluationInput, settings: RenderSettings) -> RenderedPrompt:
    user = input.prompts.user_template.format(companion_name=input.companion_name,
        current_user_message=input.current_user_message, candidate=input.candidate,
        prior_dialogue=input.prior_dialogue or "(none)", memories=input.memories or "(none)")
    turns = (Turn(Role.System, input.prompts.system_prompt), Turn(Role.User, user))
    if input.problem:
        feedback = render_regeneration_feedback("story evaluation", input.problem, settings)
        turns = (*turns, Turn(Role.User, feedback.text))
        settings = settings.without_correction_reserve()
    return require_rendered_prompt("story evaluation", turns, settings)


async def evaluate_candidate(
    evaluator: ChatClient,
    *,
    companion_name: str,
    current_user_message: str,
    candidate: str,
    evaluator_prompt: EvaluationPrompts,
    prior_dialogue: str = "",
    memories: str = "",
    capture_options: RendererCaptureOptions = RendererCaptureOptions(),
) -> StoryQualityScore:
    """Score one saved candidate. Invalid output is an evaluation error."""
    from pm.llm.structured import extract_json, JsonOutputError
    from pm.llm.workflow_sessions import serving_context
    from pydantic import ValidationError
    capacity, _source = await serving_context(evaluator.connection, Event())
    output_tokens = evaluator.connection.completion_token_limit
    if output_tokens is None:
        raise ValueError("Story evaluation requires a configured output allowance.")
    settings = RenderSettings(context_window=capacity, input_fraction=evaluator.connection.context_input_fraction,
                              reserved_output_tokens=output_tokens)
    problem = ""
    rejected_answer = ""
    while True:
        input = StoryEvaluationInput(companion_name, current_user_message, candidate, evaluator_prompt,
                                     prior_dialogue, memories, problem, rejected_answer)
        capture_settings = RendererCaptureSettings(capture_options.directory / "story_evaluation" / uuid4().hex,
            capture_options.capture_input, capture_options.capture_output)
        with RendererCapture(capture_settings, RendererInput("story_evaluation", input, settings)) as capture:
            prompt = render_story_evaluation(input, settings)
            capture.save_output(prompt)
        raw = (await evaluator.complete(prompt.turns, settings=GenerationSettings(temperature=Sampler.TempReasoning))).text
        try:
            return StoryQualityScore.model_validate(extract_json(raw))
        except (JsonOutputError, ValidationError) as error:
            problem = str(error)
            rejected_answer = raw



def save_json(path: Path, payload: JsonValue | BaseModel | tuple[BaseModel, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, BaseModel):
        encoded = payload.model_dump(mode="json")
    elif isinstance(payload, tuple):
        encoded = [item.model_dump(mode="json") for item in payload]
    else:
        encoded = payload
    path.write_text(json.dumps(encoded, indent=2, ensure_ascii=False), encoding="utf-8")
