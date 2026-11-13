"""One deterministic driving session over the real application modules.

``build_app`` assembles the real modules and subscribes them to a real bus, but
it never starts them. This module starts nothing either. Instead it calls
``BusModule.step(block=False)`` in an order the caller states, and advances a
``VirtualClock`` when a timer tick is wanted. The result is the production
module graph, the production store, and the production kernels running in a
single thread with no wall-clock scheduling.

The only substituted collaborator is the model, which
:mod:`tests.replay.model` records or replays.

Reading map: :func:`open_runtime` builds the session, :class:`ScenarioRuntime`
exposes the modules and the deterministic step helpers, and
:class:`PublicReply` retains what the companion actually said.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from pm import vclock
from pm.bus.envelopes import Envelope
from pm.host.application import Application, BusModule
from pm.llm.config import load_config, resolve_connection
from pm.llm.mcp_config import ToolCallConfig
from pm.llm.workflow_sessions import WorkflowModels
from pm.memory.store import MemoryStore
from pm.memory.read_scope import ReadScope
from workflows.cognitive_control.host import CognitiveControl
from workflows.companion import build_app
from workflows.input_triage.host import InputTriage
from workflows.inner_thought.host import PrivateThoughtLoop
from workflows.public_speech.user_input import UserInputProxy

from tests.replay.model import ScriptedModel, open_scripted_model


ROOT = Path(__file__).resolve().parents[2]

#: The virtual interval between two private-thought timer ticks. A scenario
#: advances the clock by this much when it wants one private cycle. The value
#: only has to be larger than any other module's tick in the scenario.
PRIVATE_TICK_SECONDS = 3600.0


class PublicReply(BaseModel):
    """One public utterance the companion emitted during the session."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(description="Exact text handed to the public reply sink.")
    order: int = Field(ge=0, description="Zero-based order of emission within the session.")


class SteppedModule(BaseModel):
    """One module step and whether it found work."""

    model_config = ConfigDict(extra="forbid")

    module: str = Field(description="Module id that was stepped.")
    did_work: bool = Field(description="True when the step ran a batch or a timer tick.")
    virtual_seconds: float = Field(description="Virtual seconds elapsed when the step ran.")


@dataclass(slots=True)
class ScenarioRuntime:
    """The real application, driven one module step at a time.

    ``store`` is the live memory store every module reads and writes. ``app``
    holds the modules but was never started, so nothing here depends on thread
    scheduling. ``events`` retains every envelope the bus delivered during the
    session, in publication order.
    """

    scenario: str
    directory: Path
    clock: vclock.VirtualClock
    store: MemoryStore
    app: Application
    model: ScriptedModel
    control: CognitiveControl
    triage: InputTriage
    private: PrivateThoughtLoop
    input: UserInputProxy
    events: list[Envelope] = field(default_factory=lambda: [])
    replies: list[PublicReply] = field(default_factory=lambda: [])
    steps: list[SteppedModule] = field(default_factory=lambda: [])

    # ------------------------------------------------------------- driving

    def advance(self, seconds: float) -> None:
        """Move virtual time forward. A due timer becomes runnable."""
        if seconds < 0:
            raise ValueError("Virtual time cannot move backward.")
        self.clock.set(self.clock.utc() + timedelta(seconds=seconds))

    def step(self, module: BusModule, seconds: float = 0.0) -> SteppedModule:
        """Advance time and run at most one batch or timer tick of ``module``."""
        if seconds:
            self.advance(seconds)
        did_work = module.step(block=False)
        stepped = SteppedModule(module=module.module_id, did_work=did_work,
                                virtual_seconds=self.clock.monotonic())
        self.steps.append(stepped)
        self.app.raise_if_error()
        return stepped

    def step_named(self, module_id: str, seconds: float = 0.0) -> SteppedModule:
        """Step one module by its bus id, so a scenario reads as module names."""
        return self.step(self.module(module_id), seconds)

    def module(self, module_id: str) -> BusModule:
        """Return the assembled module with this id, or fail loudly."""
        matches = [module for module in self.app.modules if module.module_id == module_id]
        if len(matches) != 1:
            raise LookupError(f"Expected exactly one module named {module_id!r}, found {len(matches)}.")
        return matches[0]

    def private_cycle(self) -> SteppedModule:
        """Run one private-thought timer tick, which is one inquiry invocation."""
        return self.step(self.private, PRIVATE_TICK_SECONDS)

    def settle(self, order: Sequence[str], *, limit: int = 40) -> int:
        """Step ``order`` round-robin until one full pass finds no work.

        A bounded loop, not an open one: a module that keeps producing work is a
        defect worth seeing, so the limit raises instead of spinning.
        """
        for pass_index in range(limit):
            worked = False
            for module_id in order:
                if self.step_named(module_id).did_work:
                    worked = True
            if not worked:
                return pass_index
        raise RuntimeError(f"Modules {list(order)} still had work after {limit} passes.")

    def submit(self, text: str) -> str:
        """Commit one public user message through the real input proxy.

        Outside input arrives on the console thread in the running application,
        so it collects no module's memory view. The harness enters its own
        non-collecting read scope to keep that property when a scenario
        interleaves the submit inside a module step.
        """
        with ReadScope() as reads:
            reads.collecting = False
            return self.input.submit(text)

    def envelopes(self, topic: str) -> tuple[Envelope, ...]:
        """Return every captured envelope of one topic in publication order."""
        return tuple(item for item in self.events if item.topic == topic)

    def close(self) -> None:
        """Stop the assembled modules without joining threads that never ran."""
        self.app.shutdown()

    # ----------------------------------------------------------- inspection

    def provider_requests(self) -> int:
        """Return real provider requests this session made."""
        return self.model.request_count


def open_runtime(scenario: str, store: MemoryStore, directory: Path, *, mode: str,
                 clock_start: datetime) -> ScenarioRuntime:
    """Assemble the real application over ``store`` with a scripted model.

    The virtual clock is installed process-wide because every record timestamp
    and every opportunity age reads it. A replay must therefore start at the
    same instant as its recording, which is why the manifest retains it.
    """
    if mode not in {"record", "replay"}:
        raise ValueError(f"Unknown replay mode {mode!r}.")
    config = load_config(ROOT / "config.yaml")
    connection = resolve_connection(config)
    model = open_scripted_model(connection, directory, "record" if mode == "record" else "replay")
    workflow_models = WorkflowModels(config, model.owner, live_test=True)
    directory.mkdir(parents=True, exist_ok=True)
    clock = vclock.VirtualClock(clock_start)
    vclock.install(clock)
    replies: list[PublicReply] = []

    def collect_reply(text: str) -> None:
        """Retain public speech in emission order without printing it."""
        replies.append(PublicReply(text=text, order=len(replies)))

    app = build_app(None, directory / "runtime", ghost=store, cognition="real",
                    stage_model=model, workflow_models=workflow_models,
                    config_path=str(ROOT / "config.yaml"), thought_period=PRIVATE_TICK_SECONDS,
                    record_lifecycle=False, enable_minecraft=False, world_poll_seconds=None,
                    tool_call=ToolCallConfig(), emit_reply=collect_reply)
    runtime = ScenarioRuntime(
        scenario=scenario,
        directory=directory,
        clock=clock,
        store=store,
        app=app,
        model=model,
        control=next(module for module in app.modules if isinstance(module, CognitiveControl)),
        triage=next(module for module in app.modules if isinstance(module, InputTriage)),
        private=next(module for module in app.modules if isinstance(module, PrivateThoughtLoop)),
        input=next(module for module in app.modules if isinstance(module, UserInputProxy)),
        replies=replies,
    )

    def observe(envelope: Envelope) -> None:
        """Retain every published envelope without altering routing."""
        runtime.events.append(envelope)

    app.bus.subscribe("replay_evidence", lambda envelope: True, observe)
    return runtime


__all__ = ["PRIVATE_TICK_SECONDS", "PublicReply", "ScenarioRuntime", "SteppedModule", "open_runtime"]
