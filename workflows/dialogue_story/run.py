"""Run standalone Story dialogue without module threads.

    python -m workflows.dialogue_story.run accept
    python -m workflows.dialogue_story.run trials --trials 10
    python -m workflows.dialogue_story.run show

``accept`` runs the fixed situation once, with the one agentic review, and
commits an accepted reply to a MemoryStore checkpoint. ``trials`` runs N independent
generations from the same initial MemoryStore state, then activates the evaluator
over the saved artifacts and reports the quality gate. ``show`` prints the
exact three messages without contacting a model.
"""
from __future__ import annotations

from pm import vclock

import argparse
import asyncio
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from pm.memory.embeddings import EmbeddingService
from pm.memory.store import MemoryStore
from pm.llm.activation import ChatClient, ModelActivation
from pm.llm.config import LlmPreset, load_config, resolve_connection
from pm.llm.prompt_encoding import OpenAIChatEncoder
from workflows.dialogue_story import standalone as engine
from workflows.dialogue_story.context import plan_story_request
from workflows.dialogue_story.schemas import StoryQualityScore, StoryReport, StoryRequest
from workflows.dialogue_story.steps import build_story_messages, load_prompts, render_prior_dialogue

DEFAULT_SEED = engine.PROJECT_ROOT / "test_data" / "personas" / "spamuel" / "canon.yaml"
DEFAULT_SITUATION = engine.PROJECT_ROOT / "test_data" / "situations" / "story_dialogue" / "situation.yaml"
REQUIRED_GOOD = 7


def _config_path() -> str:
    if os.environ.get("PM_MODEL_CONFIG"):
        return os.environ["PM_MODEL_CONFIG"]
    if Path("config.local.yaml").is_file():
        return "config.local.yaml"
    return "config.example.yaml"


def _run_root(runtime_root: Path) -> Path:
    stamp = vclock.now_aware().strftime("%Y%m%dT%H%M%SZ")
    return Path(runtime_root) / "w4" / f"{stamp}-{uuid4().hex[:8]}"


def _plan_request(ghost: MemoryStore, embedding: EmbeddingService,
                  situation: dict[str, Any]) -> StoryRequest:
    """Plan one Story request synchronously for a command-line run."""
    return asyncio.run(plan_story_request(ghost, embedding, situation))


def command_show(args: argparse.Namespace) -> int:
    """Print the exact Story messages without any model call."""
    ghost = MemoryStore.from_yaml(args.seed)
    embedding = EmbeddingService(load_config(_config_path()).embedding, live_test=True)
    request = _plan_request(ghost, embedding, engine.read_yaml(args.situation))
    prompts = load_prompts()
    for index, message in enumerate(build_story_messages(request, prompts), start=1):
        print(f"===== message {index}: {message.role.value} =====")
        print(message.content.text)
        print()
    return 0


def command_accept(args: argparse.Namespace) -> int:
    """Generate, evaluate, and commit one accepted Story reply."""
    config = load_config(_config_path())
    owner = ModelActivation()
    default = resolve_connection(config)
    asyncio.run(owner.activate(default, live_test=args.live_test))
    client = ChatClient(default, owner, live_test=args.live_test)
    embedding = EmbeddingService(config.embedding, live_test=args.live_test)

    run_root = _run_root(args.runtime_root)
    ghost = MemoryStore.from_yaml(args.seed, checkpoint=run_root / "ghost.pickle")
    situation = engine.read_yaml(args.situation)
    request = _plan_request(ghost, embedding, situation)
    prompts = load_prompts()

    engine.save_json(run_root / "request.json", request.model_dump(mode="json"))
    # The neutral turns cannot enter JSON directly. Encode them into the
    # OpenAI-compatible wire form so the saved evidence shows the exact
    # request payload that a provider would receive.
    engine.save_json(
        run_root / "messages.json",
        OpenAIChatEncoder().encode(build_story_messages(request, prompts))["messages"],
    )

    run_id = f"w4-accept-{uuid4().hex[:8]}"
    result, report = engine.generate_story(
        ghost, engine.make_environment(ghost), client, request.model_dump(mode="json"),
        run_root, run_id,
    )
    engine.save_json(run_root / "accept" / "report.json", report.model_dump(mode="json"))
    engine.save_json(run_root / "accept" / "kernel-summary.json", result.kernel_summary)

    commit = None
    if report.accepted:
        commit = engine.commit_accepted_dialogue(
            ghost, report.candidate, request.companion_name, run_id,
        )
        engine.save_json(run_root / "accept" / "ghost-commit.json", commit)

    print(f"run root: {run_root}")
    print(f"accepted: {report.accepted}")
    print(f"candidate: {report.candidate!r}")
    if commit is not None:
        print(f"committed knoxel {commit['knoxel_id']} under checkpoint {run_root / 'ghost.pickle'}")
    if not report.accepted:
        return 2
    return 0


def command_trials(args: argparse.Namespace) -> int:
    """Generate several independent candidates and retain each evaluated report."""
    config = load_config(_config_path())
    owner = ModelActivation()
    default = resolve_connection(config)
    asyncio.run(owner.activate(default, live_test=args.live_test))
    client = ChatClient(default, owner, live_test=args.live_test)
    embedding = EmbeddingService(config.embedding, live_test=args.live_test)

    run_root = _run_root(args.runtime_root)
    ghost = MemoryStore.from_yaml(args.seed)
    situation = engine.read_yaml(args.situation)
    request = _plan_request(ghost, embedding, situation)
    environment = engine.make_environment(ghost)

    engine.save_json(run_root / "request.json", request.model_dump(mode="json"))
    reports: list[StoryReport] = []
    for trial in range(args.trials):
        run_id = f"w4-trial-{trial:02d}-{uuid4().hex[:6]}"
        result, report = engine.generate_story(
            ghost, environment, client, request.model_dump(mode="json"), run_root, run_id,
        )
        reports.append(report)
        engine.save_json(
            run_root / "trials" / f"trial-{trial:02d}" / "report.json",
            report.model_dump(mode="json"),
        )
        engine.save_json(
            run_root / "trials" / f"trial-{trial:02d}" / "kernel-summary.json",
            result.kernel_summary,
        )
        print(f"trial {trial}: accepted={report.accepted} candidate={report.candidate!r}")

    # Generation is finished for the batch. Only now activate the evaluator.
    evaluator_connection = resolve_connection(config, preset=LlmPreset.UnitTestEvaluator)
    asyncio.run(owner.activate(evaluator_connection, live_test=args.live_test))
    evaluator = ChatClient(evaluator_connection, owner, live_test=args.live_test)
    evaluator_prompt = engine.read_yaml(engine.WORKFLOW_DIR / "evaluator.yaml")

    evaluations: list[dict] = []
    prior = render_prior_dialogue(request.prior_dialogue)
    memories = "\n".join(request.memories)
    for trial, report in enumerate(reports):
        score = asyncio.run(engine.evaluate_candidate(
            evaluator,
            companion_name=request.companion_name,
            current_user_message=request.current_user_message,
            candidate=report.candidate,
            evaluator_prompt=evaluator_prompt,
            prior_dialogue=prior,
            memories=memories,
        ))
        entry = {"trial": trial, "accepted": report.accepted, **score.model_dump()}
        evaluations.append(entry)
        engine.save_json(run_root / "trials" / f"trial-{trial:02d}" / "evaluation.json", entry)

    good = sum(1 for entry in evaluations if entry["scale"] == 1)
    summary = {
        "trials": args.trials,
        "good": good,
        "borderline": sum(1 for entry in evaluations if entry["scale"] == 0),
        "poor": sum(1 for entry in evaluations if entry["scale"] == -1),
        "required_good": REQUIRED_GOOD,
        "passed": good >= REQUIRED_GOOD,
        "accepted": sum(1 for report in reports if report.accepted),
        "evaluator_model": evaluator_connection.model,
        "evaluator_shares_backend": config.evaluator_shares_backend(),
        "rubric_version": evaluator_prompt["version"],
    }
    engine.save_json(run_root / "evaluations.json", evaluations)
    engine.save_json(run_root / "summary.json", summary)
    print(f"run root: {run_root}")
    print(f"quality gate: {summary['passed']} ({good}/{args.trials} good, rubric {summary['rubric_version']})")
    return 0 if summary["passed"] else 3


def main(argv: list[str] | None = None) -> int:
    """Run one standalone Story subcommand."""
    parser = argparse.ArgumentParser(description="Standalone Story dialogue")
    parser.add_argument("--seed", default=str(DEFAULT_SEED))
    parser.add_argument("--situation", default=str(DEFAULT_SITUATION))
    parser.add_argument("--runtime-root", default=".runtime")
    parser.add_argument("--live-test", action=argparse.BooleanOptionalAction, default=True,
                        help="Fail fast on an unavailable endpoint instead of retrying forever.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("show", help="print the exact three messages without a model")
    sub.add_parser("accept", help="run once and commit an accepted reply")
    trials = sub.add_parser("trials", help="run N independent trials and evaluate them")
    trials.add_argument("--trials", type=int, default=10)
    args = parser.parse_args(argv)

    started = time.monotonic()
    if args.command == "show":
        code = command_show(args)
    elif args.command == "accept":
        code = command_accept(args)
    else:
        code = command_trials(args)
    print(f"elapsed: {time.monotonic() - started:.1f}s")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
