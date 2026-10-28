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
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pm.constants import Sampler
from pm.utils import yaml_util
from pm_workflows.runtime import WorkflowRuntime

from pm.memory.store import MemoryStore
from pm.records.memory_records import Feature
from pm.records.enums import FeatureType, InterlocusType
from pm.llm.activation import ChatClient
from pm.llm.completion import AsyncTextCompletion
from pm.llm.prompt_encoding import TurnInput
from collections.abc import Sequence
from threading import Event
from pm.runs.workflow_runner import WorkflowRequest, WorkflowResult, WorkflowRunner, WorkflowStatus
from workflows.dialogue_story.schemas import StoryQualityScore, StoryReport
from pm.memory.bash_view import MemoryBashEnvironment

WORKFLOW_DIR = Path(__file__).resolve().parent
MANIFEST = WORKFLOW_DIR / "dialogue_story.workflow.md"
PROJECT_ROOT = WORKFLOW_DIR.parent.parent


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Load one YAML prompt or evaluator file from disk."""
    return yaml_util.safe_load(Path(path).read_text(encoding="utf-8"))


def build_story_tools(client: ChatClient) -> dict[str, AsyncTextCompletion]:
    """Bind generation and repair measurement to the same active client.

    Empty output uses the shared bounded retry. Cancellation reaches both
    inference and measurement. Programming and provider errors propagate.
    """

    async def story_complete(messages: Sequence[TurnInput], **generation: object) -> str:
        from pm.llm.stage_model import complete_with_retry
        return await complete_with_retry(client, messages, generation)

    async def measure(messages: Sequence[TurnInput], *, cancel: threading.Event | None = None,
                      **generation: object) -> dict[str, object]:
        from pm.llm.context_window import measure_context
        with client.activation.use(client.connection):
            return await measure_context(client.connection, messages,
                                         cancel if cancel is not None else threading.Event(), **generation)

    story_complete.measure_context = measure
    return {"story_complete": story_complete, "public_complete": story_complete}


def make_environment(ghost: MemoryStore) -> MemoryBashEnvironment:
    """Return a read-only working environment over the current store state."""
    view = ghost.snapshot()
    return MemoryBashEnvironment(view.serialize_knoxels(), view.serialize_gwt(), "read")


def generate_story(
    ghost: MemoryStore,
    environment: MemoryBashEnvironment,
    client: ChatClient,
    request_payload: dict[str, Any],
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
    runtime = WorkflowRuntime(
        environment=environment,
        stop_event=threading.Event(),
        tools=build_story_tools(client),
        propagate_python_errors=True,
    )
    runner = WorkflowRunner(MANIFEST, runtime_root, namespace="dialogue_story", base_dir=PROJECT_ROOT)
    result = runner.invoke(WorkflowRequest(run_id, request_payload), runtime, resume=resume)
    if result.status is not WorkflowStatus.Completed:
        raise RuntimeError(f"Story run {run_id} did not complete: {result.kernel_summary}")
    report = StoryReport.model_validate(result.report)
    return result, report


def commit_accepted_dialogue(ghost: MemoryStore, candidate: str, companion_name: str,
                            commit_id: str) -> dict[str, Any]:
    """Commit one accepted reply as public dialogue memory.

    Called only for an accepted candidate, after generation. Trials never call
    this, so independent trials start from the same initial state.
    """
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
    return {"commit_id": receipt.commit_id, "knoxel_id": feature.id}


# --------------------------------------------------------------- evaluation

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> dict:
    """Extract the first JSON object from evaluator text. Malformed raises."""
    match = _JSON_OBJECT.search(text or "")
    if match is None:
        raise ValueError("The evaluator returned no JSON object.")
    return json.loads(match.group(0))


async def evaluate_candidate(
    evaluator: ChatClient,
    *,
    companion_name: str,
    current_user_message: str,
    candidate: str,
    evaluator_prompt: dict,
    prior_dialogue: str = "",
    memories: str = "",
) -> StoryQualityScore:
    """Score one saved candidate. Invalid output is an evaluation error."""
    user = evaluator_prompt["user_template"].format(
        companion_name=companion_name,
        current_user_message=current_user_message,
        candidate=candidate,
        prior_dialogue=prior_dialogue or "(none)",
        memories=memories or "(none)",
    )
    from pm.llm.structured import complete_json_async
    from pm.llm.context_window import measure_context

    async def complete(turns: Sequence[TurnInput], **generation: object) -> str:
        """Return the evaluator's final text, never its reasoning."""
        return (await evaluator.complete(turns, **generation)).text

    async def measure(turns: Sequence[TurnInput], *, cancel: Event | None = None,
                      **generation: object) -> dict[str, object]:
        return await measure_context(evaluator.connection, turns, cancel or Event(), **generation)

    return await complete_json_async(complete,
        [{"role": "system", "content": evaluator_prompt["system_prompt"]}, {"role": "user", "content": user}],
        StoryQualityScore, measure=measure, temperature=Sampler.TempReasoning)



def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


__all__ = [
    "MANIFEST",
    "PROJECT_ROOT",
    "build_story_tools",
    "commit_accepted_dialogue",
    "evaluate_candidate",
    "extract_json",
    "generate_story",
    "make_environment",
    "read_yaml",
    "save_json",
]
