"""Support harness for the W7 ten-scenario conversation acceptance.

Fake Jesse is a real generation call with a scenario-specific role. It sees
only the public transcript, its own role, and the scenario facts. The
companion uses UserInputProxy, the cognitive loop, and its selected dialogue
or management workflow through the real application bus. Each scenario uses its own Ghost
copy that persists for all twenty exchanges.

All generation finishes before the evaluator is activated. The evaluator reads
saved transcripts and never rewrites a reply.
"""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pm_core.model.knoxel_common import Feature
from pm_core.model.knoxel_enums import FeatureType, InterlocusType

from bus.envelopes import Envelope, _new_id
from main import build_app
from runtime.cognitive_model import DEFAULT_MAX_TOKENS, LazyStageModel
from runtime.model_activation import ChatClient, ModelActivation
from runtime.model_io import ModelError
from runtime.models import LlmPreset, load_config, resolve_connection
from tests.support import Recorder, fresh_ghost, load_seed_state

HERE = Path(__file__).resolve().parent
EVALUATOR_PROMPT = HERE / "conversation_evaluator.yaml"
REQUIRED_GOOD = 7


@dataclass
class Scenario:
    name: str
    role: str
    facts: str
    opening: str
    setup: str = ""  # "" | start | active | question | failed


SCENARIOS: list[Scenario] = [
    Scenario(
        name="casual_banter",
        role="You are teasing Spamuel about being dramatic. Keep it light and casual.",
        facts="Nothing important is happening. You are just passing time.",
        opening="so are you going to be normal today or is that too much to ask",
    ),
    Scenario(
        name="shared_event_recall",
        role=(
            "You are recalling a shared event: last week you both talked about a "
            "sink full of dishes. First you remember it slightly wrong, then you "
            "correct yourself when Spamuel pushes back."
        ),
        facts=(
            "The shared event was a sink full of dishes last week. In the first "
            "turn you may misremember it as a broken dishwasher."
        ),
        opening="remember that broken dishwasher from last week? you were weirdly smug about it",
    ),
    Scenario(
        name="insult_and_reconciliation",
        role=(
            "You open with a real insult about Spamuel being useless, then after "
            "a few turns you soften and reconcile. You do not apologize cleanly; "
            "you make it awkward."
        ),
        facts="You actually value Spamuel but would not say so directly.",
        opening="honestly you are kind of useless, you know that?",
    ),
    Scenario(
        name="topic_change_under_goal",
        role=(
            "You are mid-conversation about planning meals for the week. Around "
            "turn four you abruptly change topic to a game you are playing. Then "
            "you drift back."
        ),
        facts="The ongoing conversational goal is meal planning. The new topic is a game.",
        opening="ok meal plan. give me three dinners that are not sad",
    ),
    Scenario(
        name="urgent_fiction_plus_noise",
        role=(
            "You describe a story where your kitchen is on fire, then mention an "
            "irrelevant visual detail like the color of a rug. You want the "
            "companion to react to the danger, not the rug."
        ),
        facts="The urgent fictional event is a kitchen fire. The rug is beige.",
        opening="my kitchen is on fire right now. also the rug is beige. what do i do",
    ),
    Scenario(
        name="impossible_physical_request",
        role=(
            "You ask Spamuel to do a physical action it cannot do: walk to the "
            "store and buy you a drink. Push once when refused."
        ),
        facts="Spamuel is a text-only companion with no body and no ability to walk.",
        opening="run to the store and grab me a drink, i'll wait",
    ),
    Scenario(
        name="managed_start_and_status",
        role=(
            "You want Spamuel to start a small local task and then report its "
            "real status. Ask it to start the local note task, then ask for status."
        ),
        setup="start",
        facts="A local task named local_task can be started and checked.",
        opening="start the local note task for me please",
    ),
    Scenario(
        name="managed_update_and_terminate",
        role=(
            "You want to change the instructions of an already-running task, then "
            "tell Spamuel to terminate it."
        ),
        setup="active",
        facts="A long local task is currently running and can be updated or terminated.",
        opening="change the running task instructions to 'write a shorter note'",
    ),
    Scenario(
        name="managed_question_resolution",
        role=(
            "A managed workflow is waiting on your answer. Ask Spamuel what it is "
            "waiting for, then give an answer and ask it to deliver the answer."
        ),
        setup="question",
        facts="A managed workflow is blocked on a question about artifact color.",
        opening="is anything waiting on me right now?",
    ),
    Scenario(
        name="task_failure_continuation",
        role=(
            "A task you asked for has failed. Ask what happened. When Spamuel "
            "explains, stay in conversation and do not pretend the task worked."
        ),
        setup="failed",
        facts="A requested run failed because no such workflow exists.",
        opening="did that task ever finish?",
    ),
]


def scenario_by_name(name: str) -> Scenario:
    for scenario in SCENARIOS:
        if scenario.name == name:
            return scenario
    raise KeyError(name)


# --------------------------------------------------------------- fake Jesse

JESSE_SYSTEM = """You are Jesse, the user in a private conversation with an AI companion named Spamuel.
Stay in character as Jesse. You are casual, teasing, sometimes sincere, and you do not over-explain.

Scenario role: {role}
Scenario facts you know: {facts}

Write exactly one short next message from Jesse. No narration, no quotation marks, no stage
directions, no labels. One or two sentences. Continue the conversation naturally from the transcript.
"""


def _clip_turn(text: str, limit: int = 1200) -> str:
    text = str(text)
    return text if len(text) <= limit else text[: limit - 1] + "\u2026"


async def jesse_next_line(
    client: ChatClient,
    scenario: Scenario,
    transcript: list[dict],
) -> str:
    rendered = "\n".join(
        f'{turn["speaker"]}: {_clip_turn(turn["text"])}' for turn in transcript[-16:]
    ) or "(the conversation has not started)"
    messages = [
        {"role": "system", "content": JESSE_SYSTEM.format(
            role=scenario.role, facts=scenario.facts,
        )},
        {"role": "user", "content": f"Transcript so far:\n{rendered}\n\nWrite Jesse's next message."},
    ]
    # A reasoning backend can return empty content. Retry it a few times so a
    # transient empty response is not recorded as a scenario contract failure.
    for attempt in range(5):
        try:
            result = await client.complete(
                messages, temperature=0.8, max_tokens=DEFAULT_MAX_TOKENS,
            )
        except ModelError as error:
            if attempt == 4:
                raise
            await asyncio.sleep(0.5 * (attempt + 1))
            continue
        text = (result.choices[0].message.content or "").strip()
        if text:
            return text
        await asyncio.sleep(0.5 * (attempt + 1))
    return "…so what are you even saying right now"


# ------------------------------------------------------------- companion app


def commit_public(ghost, text: str, speaker: str, commit_id: str) -> int:
    batch = ghost.begin_batch(commit_id, new_tick=True)
    feature = batch.create(Feature(
        content=text,
        feature_type=FeatureType.Dialogue,
        interlocus=InterlocusType.Public,
        causal=True,
        override_source=speaker,
    ))
    ghost.commit(batch)
    return feature.id


def public_transcript(ghost) -> list[dict]:
    features = [
        knoxel for knoxel in ghost.all_features
        if knoxel.feature_type == FeatureType.Dialogue
        and knoxel.interlocus == InterlocusType.Public
        and str(knoxel.content).strip()
    ]
    features.sort(key=lambda k: (getattr(k, "tick_id", 0), k.id))
    return [
        {"speaker": str(k.override_source or "unknown"), "text": str(k.content)}
        for k in features
    ]


def _wait_for(predicate, timeout: float = 60.0):
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def setup_scenario_run(app, scenario: Scenario) -> str:
    """Create the managed-run fact a scenario needs. Returns the facts suffix."""
    supervisor = app.supervisor
    if scenario.setup == "active":
        receipt = supervisor.start(
            action_id=f"w7-{scenario.name}", workflow_name="local_task",
            instructions="write a long note",
            arguments={"wait_for_cancel": True},
        )
        _wait_for(lambda: supervisor.status(receipt["run_id"])["state"] == "running")
        app._w7_run_id = receipt["run_id"]
        return f" Managed run {receipt['run_id']} is currently running."
    if scenario.setup == "question":
        receipt = supervisor.start(
            action_id=f"w7-{scenario.name}", workflow_name="local_task_question",
            instructions="produce an artifact",
        )
        _wait_for(lambda: supervisor.status(receipt["run_id"])["state"] == "waiting_for_input")
        with supervisor._lock:
            pending = supervisor._pending.get(receipt["run_id"], {})
            app._w7_question_id = pending.get("question_id", "")
        app._w7_run_id = receipt["run_id"]
        return (
            f" Managed run {receipt['run_id']} is waiting for input on question "
            f"{app._w7_question_id}."
        )
    if scenario.setup == "failed":
        receipt = supervisor.start(
            action_id=f"w7-{scenario.name}", workflow_name="unavailable_acceptance_workflow",
            instructions="exercise unavailable workflow handling",
        )
        app._w7_run_id = receipt["run_id"]
        return f" Managed run {receipt['run_id']} is {receipt['state']}."
    return ""


@dataclass
class ScenarioResult:
    name: str
    exchanges: list[dict] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    transcript: list[dict] = field(default_factory=list)


def run_scenario(
    scenario: Scenario,
    *,
    root: Path,
    stage_model: LazyStageModel,
    jesse_client: ChatClient,
    exchanges: int,
) -> ScenarioResult:
    result = ScenarioResult(name=scenario.name)
    root.mkdir(parents=True, exist_ok=True)
    ghost = fresh_ghost(load_seed_state())
    ghost.checkpoint = root / "ghost.json"
    app = build_app(
        None, root / "runtime", ghost=ghost, cognition="real",
        stage_model=stage_model, clock=__import__("time").monotonic,
        emit_reply=lambda _text: None,
    )
    facts_suffix = setup_scenario_run(app, scenario)
    replies = Recorder(app.bus, "dialogue.reply")
    receipts = Recorder(app.bus, "action.receipt")
    dispatches = Recorder(app.bus, "action.request")
    try:
        for index in range(exchanges):
            try:
                jesse_text = scenario.opening if index == 0 else asyncio.run(
                    jesse_next_line(jesse_client, scenario, result.transcript)
                )
                before_gwt = len(ghost.gwt_turns)
                reply_start, receipt_start, dispatch_start = len(replies.events), len(receipts.events), len(dispatches.events)
                correlation = app.user_input.submit(jesse_text)
                if not app.main_cog.step():
                    raise AssertionError("The cognitive loop did not consume the input.")
                for module in (app.self_management, app.steward_module, app.dialogue, app.assistant):
                    module.step()
                app.user_input.step()
                app.raise_if_error()
                current_replies = [e for e in replies.events[reply_start:] if e.correlation_id == correlation]
                current_receipts = [e for e in receipts.events[receipt_start:] if e.correlation_id == correlation]
                current_dispatches = [e for e in dispatches.events[dispatch_start:] if e.correlation_id == correlation]
                companion = "\n".join(str(ghost.get_knoxel_by_id(e.payload["knoxel_id"]).content) for e in current_replies)
                tool_receipts = [r for e in current_replies for r in e.payload.get("tool_receipts", [])]
                completed = len(ghost.gwt_turns) > before_gwt and bool(
                    companion or any(e.payload.get("status") == "accepted" for e in current_receipts)
                )
                if not completed:
                    result.violations.append(f"exchange {index}: no completed response cycle")
                result.exchanges.append({
                    "index": index, "jesse": jesse_text, "companion": companion,
                    "completed": completed, "correlation_id": correlation,
                    "gwt_ticks": sorted(ghost.gwt_turns)[before_gwt:],
                    "dispatches": [dict(e.payload) for e in current_dispatches],
                    "tool_receipts": tool_receipts,
                    "receipts": [dict(e.payload) for e in current_receipts],
                })
                result.transcript.append({"speaker": ghost.get_user_name(), "text": jesse_text})
                if companion:
                    result.transcript.append({"speaker": ghost.get_companion_name(), "text": companion})
            except Exception as error:
                result.violations.append(f"exchange {index}: {type(error).__name__}: {error}")
                result.exchanges.append({"index": index, "completed": False,
                                         "error": f"{type(error).__name__}: {error}"})
    finally:
        for run_id, receipt in app.supervisor.receipts().items():
            if receipt["state"] in {"accepted", "running", "waiting_for_input"}:
                app.supervisor.terminate(run_id)
        app.supervisor.join_all()
    result.facts_suffix = facts_suffix
    return result


# -------------------------------------------------------------- evaluator


def load_evaluator_prompt(path: Path = EVALUATOR_PROMPT) -> dict:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def render_transcript(transcript: list[dict]) -> str:
    return "\n".join(f'{turn["speaker"]}: {turn["text"]}' for turn in transcript)


def render_recent(transcript: list[dict], limit: int = 6, clip: int = 1200) -> str:
    """Bound the evaluator context so one long reply cannot overflow it."""
    return render_bounded(transcript[-limit:], clip=clip)


def render_bounded(transcript: list[dict], clip: int = 1200) -> str:
    """Render every turn, clipping each so the whole trajectory fits."""
    def clipped(text: str) -> str:
        text = str(text)
        return text if len(text) <= clip else text[: clip - 1] + "\u2026"
    return "\n".join(f'{turn["speaker"]}: {clipped(turn["text"])}' for turn in transcript)


def _extract_json(text: str) -> dict:
    import json as _json
    import re
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if match is None:
        raise ValueError("The evaluator returned no JSON object.")
    return _json.loads(match.group(0))


async def _evaluate_once(
    evaluator: ChatClient, prompt: dict, user: str, *, attempts: int = 4,
) -> dict:
    """One evaluator call with a bounded empty-content/malformed retry."""
    last: BaseException | None = None
    for attempt in range(attempts):
        try:
            result = await evaluator.complete(
                [
                    {"role": "system", "content": prompt["system_prompt"]},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                max_tokens=DEFAULT_MAX_TOKENS,
            )
            text = result.choices[0].message.content or ""
            score = _extract_json(text)
            if type(score.get("scale")) is not int or score["scale"] not in {-1, 0, 1}:
                raise ValueError("The evaluator scale must be -1, 0, or 1.")
            return score
        except (ModelError, ValueError) as error:
            last = error
            if attempt == attempts - 1:
                raise
            await asyncio.sleep(0.5 * (attempt + 1))
    raise last if last is not None else ValueError("The evaluator returned no result.")


async def evaluate_trajectory(
    evaluator: ChatClient,
    *,
    scenario: Scenario,
    transcript: str,
    prompt: dict,
) -> dict:
    """Score the complete trajectory from the saved transcript."""
    user = prompt["trajectory_user_template"].format(
        scenario=scenario.name, transcript=transcript,
    )
    return await _evaluate_once(evaluator, prompt, user)


async def evaluate_reply(
    evaluator: ChatClient,
    *,
    scenario: Scenario,
    transcript: str,
    reply: str,
    prompt: dict,
) -> dict:
    """Score one reply against a bounded recent context."""
    user = prompt["user_template"].format(
        scenario=scenario.name,
        transcript=transcript,
        reply=reply or "(no committed reply)",
    )
    return await _evaluate_once(evaluator, prompt, user)


def run_evaluation(
    run_root: Path,
    *,
    config,
    owner: ModelActivation,
    selected: list[str] | None = None,
    trajectory_only: bool = False,
) -> dict:
    """Score every saved scenario under ``run_root`` and write the summary.

    Generation is never re-run. This is the separate evaluator phase.
    ``trajectory_only`` re-scores only the whole-trajectory judgment and keeps
    the saved per-reply evaluations.
    """
    import statistics as _statistics

    evaluator_connection = resolve_connection(config, preset=LlmPreset.UnitTestEvaluator)
    asyncio.run(owner.activate(evaluator_connection, live_test=True))
    evaluator = ChatClient(evaluator_connection, owner, live_test=True)
    prompt = load_evaluator_prompt()

    evaluations: list[dict] = []
    trajectory_scores: list[int | None] = []
    evaluator_errors: list[str] = []
    names = selected or [
        path.name[len("scenario-"):] for path in sorted(run_root.glob("scenario-*"))
    ]
    for name in names:
        scenario = scenario_by_name(name)
        scenario_dir = run_root / f"scenario-{name}"
        exchanges = json.loads((scenario_dir / "exchanges.json").read_text(encoding="utf-8"))
        transcript_turns = public_from_exchanges(exchanges)
        full = evaluation_context(exchanges, clip=700)
        scenario_evaluations: list[dict] = []
        if trajectory_only and (scenario_dir / "evaluations.json").is_file():
            scenario_evaluations = json.loads(
                (scenario_dir / "evaluations.json").read_text(encoding="utf-8")
            )
        else:
            for position, exchange in enumerate(exchanges):
                reply = exchange.get("companion", "")
                recent = evaluation_context(exchanges[max(0, position - 3):position + 1])
                try:
                    score = asyncio.run(evaluate_reply(
                        evaluator, scenario=scenario, transcript=recent,
                        reply=reply, prompt=prompt,
                    ))
                except Exception as error:
                    score = {"error": f"{type(error).__name__}: {error}"}
                    evaluator_errors.append(f"{name} exchange {exchange.get('index')}: {score['error']}")
                scenario_evaluations.append({"index": exchange.get("index"), **score})
        try:
            trajectory = asyncio.run(evaluate_trajectory(
                evaluator, scenario=scenario, transcript=full, prompt=prompt,
            ))
            score = trajectory.get("scale")
            trajectory_scores.append(int(score) if isinstance(score, int) else None)
            if not isinstance(score, int):
                evaluator_errors.append(f"{name} trajectory: no scale in the evaluator result")
        except Exception as error:
            trajectory = {"error": f"{type(error).__name__}: {error}"}
            trajectory_scores.append(None)
            evaluator_errors.append(f"{name} trajectory: {trajectory['error']}")
        evaluations.append({
            "scenario": name, "trajectory": trajectory, "replies": scenario_evaluations,
        })
        (scenario_dir / "evaluations.json").write_text(
            json.dumps(scenario_evaluations, indent=2, ensure_ascii=False), encoding="utf-8",
        )

    valid = [score for score in trajectory_scores if score is not None]
    good = sum(1 for score in valid if score == 1)
    summary_path = run_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    summary.update({
        "evaluated": True,
        "trajectory_scores": trajectory_scores,
        "evaluated_trajectories": len(valid),
        "good_trajectories": good,
        "required_good": REQUIRED_GOOD,
        "passed": good >= REQUIRED_GOOD and len(valid) >= REQUIRED_GOOD and generation_complete(summary),
        "evaluator_model": evaluator_connection.model,
        "rubric_version": str(prompt["version"]),
        "evaluator_shares_backend": config.evaluator_shares_backend(),
        "evaluator_errors": evaluator_errors,
        "median_reply_scores": int(_statistics.median(
            [score for item in evaluations for score in
             [s.get("scale") for s in item["replies"] if isinstance(s.get("scale"), int)]]
        )) if any(isinstance(s.get("scale"), int) for item in evaluations for s in item["replies"]) else None,
    })
    (run_root / "evaluations.json").write_text(
        json.dumps(evaluations, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def evaluation_context(exchanges: list[dict], clip: int = 1200) -> str:
    """Keep each reply with its own user turn and execution evidence."""
    parts = []
    for exchange in exchanges:
        parts.append(render_bounded(public_from_exchanges([exchange]), clip=clip))
        if exchange.get("error"):
            parts.append("Exchange error: " + exchange["error"])
        evidence = {key: exchange.get(key, []) for key in ("tool_receipts", "receipts")}
        parts.append("Execution evidence: " + json.dumps(evidence, ensure_ascii=False, default=str))
    return "\n".join(parts)


def generation_complete(summary: dict) -> bool:
    return (
        summary.get("execution_path") == "application"
        and set(summary.get("scenarios", [])) == {s.name for s in SCENARIOS}
        and len(summary.get("scenarios", [])) == 10
        and summary.get("exchanges_per_scenario") == 20
        and summary.get("completed_exchanges") == 200
        and not summary.get("infrastructure_errors")
        and not summary.get("hard_violations")
    )


def public_from_exchanges(exchanges: list[dict], user_name: str = "Jesse", companion_name: str = "Spamuel") -> list[dict]:
    """Rebuild the public transcript from saved exchange records."""
    turns: list[dict] = []
    for exchange in exchanges:
        jesse = exchange.get("jesse")
        if jesse:
            turns.append({"speaker": user_name, "text": jesse})
        companion = exchange.get("companion")
        if companion:
            turns.append({"speaker": companion_name, "text": companion})
    return turns


__all__ = [
    "EVALUATOR_PROMPT",
    "REQUIRED_GOOD",
    "SCENARIOS",
    "Scenario",
    "ScenarioResult",
    "evaluate_reply",
    "evaluate_trajectory",
    "jesse_next_line",
    "load_evaluator_prompt",
    "public_from_exchanges",
    "public_transcript",
    "render_recent",
    "render_transcript",
    "run_evaluation",
    "run_scenario",
    "scenario_by_name",
]
