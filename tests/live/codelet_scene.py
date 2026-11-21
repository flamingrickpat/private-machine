"""Run one conversation on an isolated copy of Spamuel and save its codelet trace.

Start with `python -m tests.live.codelet_scene --help`. Each invocation runs the
assembled application with the configured models. Use the same scene and source
checkpoint for two catalogs to inspect their public replies and private percepts.
"""
from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from time import monotonic, sleep
from uuid import uuid4

from pydantic import BaseModel, Field

from pm import vclock
from pm.constants import CODELET_BUDGET_MAX
from pm.llm.activation import ModelActivation
from pm.llm.config import load_config, resolve_connection
from pm.llm.mcp_config import ToolCallConfig
from pm.llm.prompt_log import log_scope
from pm.llm.stage_model import LazyStageModel
from pm.llm.workflow_sessions import WorkflowModels
from pm.memory.store import MemoryStore
from pm.records.codelet_records import CodeletPercept
from pm.records.global_workspace import GwtTurnRecord
from pm.utils.operating_system import OperatingSystem
from workflows.companion import build_app
from workflows.main_cognition.codelet_registry import DEFAULT_DEFINITIONS, LoadedDefinitions, load_definitions
from workflows.public_speech.publication import exchange_record


ROOT = Path(__file__).resolve().parents[2]
CANON = ROOT / "test_data/personas/spamuel/canon.yaml"
SCENES: dict[str, str] = {  # pm-lint: allow PM004 -- synthetic user inputs indexed by scenario ID
    "good-news": "I got the pottery studio key! Six months on the waiting list and it's really mine. Can I tell you about the first thing I want to make?",
    "grief": "I went to return my sister's spare key today and remembered all over again that she's dead. Can you sit with me for a bit? I don't want a fix.",
    "quiet": "Can we just sit here while I finish this row of knitting? I like having you around even when neither of us has much to say.",
    "invitation": "We're both adults. I liked flirting with you last night. I'm asking you on an actual date this time. Do you want that too?",
    "friends": "I like talking with you. Just so we're clear, when I said coffee I meant as friends. Do you still want to come?",
    "desire": "I like being desired, and I'm flirting with you on purpose. We're both adults. Do you enjoy that kind of attention from me?",
    "withdrawal": "I know I asked you to take the lead when we flirted. I've changed my mind. Stop the roleplay. Can we talk about music?",
    "repair": "I promised you I would keep your rough draft private. I showed it to Lee. That was my doing. I've asked Lee to delete it. What do you want me to do about my copy?",
    "shared-work": "I laid the red blocks along the roof like you suggested. The curve finally reads as a spiral! Which bit would you like to design next?",
    "task": "I'm sorting the oak planks into stacks of 64. If I have 65, how many go in the second stack?",
}


class PerceptReadout(BaseModel):
    """One committed private impression and whether attention admitted it."""

    record_id: int
    codelet: str
    content: str
    causal: bool


class SceneReadout(BaseModel):
    """Public speech and main cognition have separate clocks and evidence."""

    scene: str
    message: str
    input_id: int
    reply: str | None = None
    reply_seconds: float | None = None
    cycle_seconds: float | None = None
    turn: GwtTurnRecord | None = None
    percepts: list[PerceptReadout] = Field(default_factory=list)

    def save(self, path: Path) -> None:
        """Retain each observed commitment even if its counterpart never arrives."""
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")  # pm-lint: allow PM007 -- filesystem evidence


def read_percepts(store: MemoryStore, turn: GwtTurnRecord, catalog: LoadedDefinitions) -> list[PerceptReadout]:
    """Check the catalog and budget used by the committed turn, then follow its record IDs."""
    trace = turn.trace
    if trace.definitions_digest != catalog.digest or not trace.codelet_runs:
        raise AssertionError(f"Tick {turn.tick_id} lacks runs from catalog {catalog.digest}.")
    expected_runs = min(trace.codelet_budget, len(catalog.definitions.codelets))
    if len(trace.codelet_runs) != expected_runs or trace.codelet_budget > CODELET_BUDGET_MAX:
        raise AssertionError(f"Tick {turn.tick_id} ran {len(trace.codelet_runs)} codelets with budget "
                             f"{trace.codelet_budget}. Expected {expected_runs}, maximum {CODELET_BUDGET_MAX}.")
    percepts: list[PerceptReadout] = []
    for record_id in trace.percept_ids:
        record = store.all_knoxels[record_id]
        if not isinstance(record, CodeletPercept):
            raise AssertionError(f"Trace percept {record_id} has type {type(record).__name__}.")
        percepts.append(PerceptReadout(record_id=record_id, codelet=record.codelet_name,
                                      content=record.content, causal=record.causal))
    return percepts


@contextmanager
def scene_runtime(clock: vclock.VirtualClock, prompt_directory: Path) -> Iterator[None]:
    """Give the application and its threads one clock and prompt directory for their lifetime."""
    environment = OperatingSystem.environment
    previous_clock = vclock.active()
    previous_log = environment.get("PM_PROMPT_LOG_DIR")
    previous_cache = environment.get("PM_COMPLETION_CACHE")
    try:
        vclock.install(clock)
        environment["PM_PROMPT_LOG_DIR"] = str(prompt_directory)
        environment["PM_COMPLETION_CACHE"] = "0"
        with log_scope(prompt_directory):
            yield
    finally:
        vclock.install(previous_clock)
        if previous_log is None:
            environment.pop("PM_PROMPT_LOG_DIR", None)
        else:
            environment["PM_PROMPT_LOG_DIR"] = previous_log
        if previous_cache is None:
            environment.pop("PM_COMPLETION_CACHE", None)
        else:
            environment["PM_COMPLETION_CACHE"] = previous_cache


def run_scene(scene: str, config: Path, source: Path, catalog: Path) -> Path:
    """Wait up to thirty minutes for a committed public reply and the main turn for that input."""
    out = ROOT / ".runtime/codelet-expansion/scenes" / f"{scene}-{uuid4().hex[:8]}"
    out.mkdir(parents=True)
    print(f"Evidence: {out}", flush=True)
    store = MemoryStore.load(source, checkpoint=out / "ghost.pickle")
    # Resume one minute after the checkpoint, even if the fixture is months old on disk.
    opening = max(record.timestamp_world_begin for record in store.all_knoxels.values()) + timedelta(minutes=1)
    clock = vclock.VirtualClock(opening)
    definitions = load_definitions(catalog)
    message = SCENES[scene]
    settings = load_config(config)
    activation = ModelActivation()
    model = LazyStageModel(resolve_connection(settings), activation, live_test=True)
    models = WorkflowModels(settings, activation, live_test=True)
    with scene_runtime(clock, out / "prompts"):
        app = build_app(None, out / "runtime", ghost=store, cognition="real", config_path=str(config),
                        stage_model=model, workflow_models=models,
                        codelet_definitions=str(catalog.resolve()), thought_period=3600,
                        record_lifecycle=False, world_poll_seconds=None, enable_minecraft=False,
                        tool_call=ToolCallConfig())
        try:
            start = monotonic()
            correlation = app.user_input.submit(message)
            source_record = exchange_record(store, correlation)
            if source_record is None:
                raise AssertionError(f"Input {correlation} was not committed.")
            input_id = source_record.id
            receipt = SceneReadout(scene=scene, message=message, input_id=input_id)
            path = out / "scene.json"
            receipt.save(path)
            app.run()
            while monotonic() - start < 1800:
                clock.set(opening + timedelta(seconds=monotonic() - start))
                app.raise_if_error()
                with store.lock:
                    if receipt.reply is None:
                        source_record = store.snapshot({input_id}, capture=False).get(input_id)
                        reply_id = source_record.metadata.get("public_reply_id")
                        if isinstance(reply_id, int):
                            receipt.reply = store.all_knoxels[reply_id].content
                            receipt.reply_seconds = monotonic() - start
                            receipt.save(path)
                    if receipt.turn is None:
                        for turn in store.gwt_turns.values():
                            if turn.slot_id == "main" and input_id in turn.stimulus_ids:
                                receipt.turn = turn
                                receipt.cycle_seconds = monotonic() - start
                                receipt.percepts = read_percepts(store, turn, definitions)
                                receipt.save(path)
                                break
                    if receipt.turn is not None and receipt.reply is not None:
                        return path
                sleep(0.5)
            raise TimeoutError(f"No paired public reply and main cycle for input {input_id} within 1800 s. "
                               f"Reply observed: {receipt.reply is not None}. Cycle observed: {receipt.turn is not None}. "
                               f"See {path}.")
        finally:
            app.shutdown()
            app.join()
            app.raise_if_error()
            # Speech can commit after the last main-cycle checkpoint.
            store.save()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, choices=tuple(SCENES))
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--source", type=Path, default=CANON, help="Synthetic Spamuel seed or checkpoint to copy.")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_DEFINITIONS)
    args = parser.parse_args()
    print(run_scene(args.scene, args.config, args.source, args.catalog), flush=True)


if __name__ == "__main__":
    main()
