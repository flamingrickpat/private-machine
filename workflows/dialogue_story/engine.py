"""Non-kernel orchestration for standalone Story dialogue.

This module builds the generation tool, opens a read-only Ghost environment,
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

import yaml
from pm_workflows.runtime import WorkflowRuntime

from pm_core.model.knoxel_common import Feature
from pm_core.model.knoxel_enums import FeatureType, InterlocusType
from runtime.model_activation import ChatClient
from runtime.workflow_runner import WorkflowRequest, WorkflowResult, WorkflowRunner, WorkflowStatus
from workflows.dialogue_story.schemas import StoryQualityScore, StoryReport
from working_target import GhostBashEnvironment

WORKFLOW_DIR = Path(__file__).resolve().parent
MANIFEST = WORKFLOW_DIR / "dialogue_story.workflow.md"
PROJECT_ROOT = WORKFLOW_DIR.parent.parent


def read_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def build_story_tools(client: ChatClient) -> dict:
    """The scoped generation tool. Async so the sync role owns the event loop."""

    async def story_complete(messages: list[dict], **generation) -> str:
        result = await client.complete(messages, **generation)
        return result.choices[0].message.content or ""

    return {"story_complete": story_complete}


def make_environment(ghost) -> GhostBashEnvironment:
    view = ghost.snapshot()
    return GhostBashEnvironment(view.serialize_knoxels(), view.serialize_gwt(), "read")


def generate_story(
    ghost,
    environment: GhostBashEnvironment,
    client: ChatClient,
    request_payload: dict,
    runtime_root: Path,
    run_id: str,
    *,
    resume: bool = False,
) -> tuple[WorkflowResult, StoryReport]:
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


def commit_accepted_dialogue(ghost, candidate: str, companion_name: str, commit_id: str) -> dict:
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
    result = await evaluator.complete(
        [
            {"role": "system", "content": evaluator_prompt["system_prompt"]},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        max_tokens=4096,
    )
    text = result.choices[0].message.content or ""
    return StoryQualityScore.model_validate(extract_json(text))


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
