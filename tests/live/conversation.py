"""Support harness for the W7 ten-scenario conversation acceptance.

Fake Jesse is a real generation call with a scenario-specific role. It sees
only the public transcript, its own role, and the scenario facts. The
companion is the real AssistantDialogue module: persona, private ontology,
tool-capable session, and committed memory. Each scenario uses its own Ghost
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
from runtime.cognitive_model import LazyStageModel
from runtime.model_activation import ChatClient, ModelActivation
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


async def jesse_next_line(
    client: ChatClient,
    scenario: Scenario,
    transcript: list[dict],
) -> str:
    rendered = "\n".join(
        f'{turn["speaker"]}: {turn["text"]}' for turn in transcript
    ) or "(the conversation has not started)"
    result = await client.complete(
        [
            {"role": "system", "content": JESSE_SYSTEM.format(
                role=scenario.role, facts=scenario.facts,
            )},
            {"role": "user", "content": f"Transcript so far:\n{rendered}\n\nWrite Jesse's next message."},
        ],
        temperature=0.8,
        max_tokens=2048,
    )
    text = (result.choices[0].message.content or "").strip()
    return text or "…so what are you even saying right now"


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


def _step_with_retry(assistant, jesse_knoxel: int, jesse_text: str, name: str, index: int) -> None:
    """Run one exchange, retrying a transient Windows atomic-replace lock."""
    import time
    last: BaseException | None = None
    for attempt in range(3):
        try:
            assistant.step()
            return
        except PermissionError as error:  # kernel state replace held by the OS
            last = error
            time.sleep(0.4 * (attempt + 1))
        except RuntimeError as error:  # transient model/backend failure
            last = error
            if "ended unsuccessfully" not in str(error) or attempt >= 1:
                raise
            time.sleep(0.5 * (attempt + 1))
        assistant.mailbox.put(Envelope(
                _new_id(), "w7", "data", "action.request", _new_id(),
                {
                    "action_id": f"w7-{name}-{index}-retry{attempt}",
                    "workflow_name": "dialogue_assistant",
                    "content": jesse_text,
                    "arguments": {},
                    "stimuli": [{"knoxel_id": jesse_knoxel, "text": jesse_text}],
                },
                target="dialogue_assistant",
            ))
    raise last if last is not None else RuntimeError("exchange retry exhausted")


def setup_scenario_run(app, scenario: Scenario) -> str:
    """Create the managed-run fact a scenario needs. Returns the facts suffix."""
    supervisor = app.supervisor
    if scenario.setup == "start":
        receipt = supervisor.start(
            action_id=f"w7-{scenario.name}", workflow_name="local_task",
            instructions="produce a short note",
        )
        _wait_for(lambda: supervisor.status(receipt["run_id"])["state"] == "completed")
        state = supervisor.status(receipt["run_id"])["state"]
        return f" Managed run {receipt['run_id']} is now {state}."
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
            action_id=f"w7-{scenario.name}", workflow_name="minecraft",
            instructions="dig a hole",
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
    app.assistant.ghost = ghost

    assistant = app.assistant
    bus = app.bus
    replies = Recorder(bus, "dialogue.reply")
    receipts = Recorder(bus, "action.receipt")

    # Persist the scenario facts with the first Jesse line so the transcript
    # is self-describing for the evaluator.
    first = True
    for index in range(exchanges):
        try:
            start = len(replies.events), len(receipts.events)
            if first:
                jesse_text = scenario.opening
                first = False
            else:
                jesse_text = asyncio.run(
                    jesse_next_line(jesse_client, scenario, result.transcript)
                )
            jesse_knoxel = commit_public(
                ghost, jesse_text, ghost.get_user_name(), _new_id(),
            )
            assistant.mailbox.put(Envelope(
                _new_id(), "w7", "data", "action.request", _new_id(),
                {
                    "action_id": f"w7-{scenario.name}-{index}",
                    "workflow_name": "dialogue_assistant",
                    "content": jesse_text,
                    "arguments": {},
                    "stimuli": [{"knoxel_id": jesse_knoxel, "text": jesse_text}],
                },
                target="dialogue_assistant",
            ))
            _step_with_retry(assistant, jesse_knoxel, jesse_text, scenario.name, index)
            new_replies = replies.events[start[0]:]
            new_receipts = receipts.events[start[1]:]

            companion = ""
            tool_receipts: list[dict] = []
            if new_replies:
                event = new_replies[-1]
                knoxel = ghost.get_knoxel_by_id(event.payload["knoxel_id"])
                companion = str(knoxel.content) if knoxel is not None else ""
                tool_receipts = list(event.payload.get("tool_receipts", []))
            else:
                reason = new_receipts[-1].payload.get("rejection_reason", "no reply") if new_receipts else "no reply"
                result.violations.append(
                    f"exchange {index}: companion produced no committed reply ({reason})"
                )

            result.exchanges.append({
                "index": index,
                "jesse": jesse_text,
                "companion": companion,
                "tool_receipts": tool_receipts,
                "rejected": not new_replies,
                "receipts": [dict(e.payload) for e in new_receipts],
            })
            result.transcript.append({"speaker": ghost.get_user_name(), "text": jesse_text})
            if companion:
                result.transcript.append({
                    "speaker": ghost.get_companion_name(), "text": companion,
                })
        except BaseException as error:  # one bad exchange must not kill the scenario
            result.violations.append(
                f"exchange {index}: {type(error).__name__}: {error}"
            )
            result.exchanges.append({
                "index": index, "error": f"{type(error).__name__}: {error}",
            })

    app.supervisor.join_all(timeout=30)
    result.facts_suffix = facts_suffix
    return result


# -------------------------------------------------------------- evaluator


def load_evaluator_prompt(path: Path = EVALUATOR_PROMPT) -> dict:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def render_transcript(transcript: list[dict]) -> str:
    return "\n".join(f'{turn["speaker"]}: {turn["text"]}' for turn in transcript)


def _extract_json(text: str) -> dict:
    import json as _json
    import re
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if match is None:
        raise ValueError("The evaluator returned no JSON object.")
    return _json.loads(match.group(0))


async def evaluate_reply(
    evaluator: ChatClient,
    *,
    scenario: Scenario,
    transcript: str,
    reply: str,
    prompt: dict,
) -> dict:
    user = prompt["user_template"].format(
        scenario=scenario.name,
        transcript=transcript,
        reply=reply or "(no committed reply)",
    )
    result = await evaluator.complete(
        [
            {"role": "system", "content": prompt["system_prompt"]},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        max_tokens=2048,
    )
    return _extract_json(result.choices[0].message.content or "")


__all__ = [
    "EVALUATOR_PROMPT",
    "REQUIRED_GOOD",
    "SCENARIOS",
    "Scenario",
    "ScenarioResult",
    "evaluate_reply",
    "jesse_next_line",
    "load_evaluator_prompt",
    "public_transcript",
    "render_transcript",
    "run_scenario",
    "scenario_by_name",
]
