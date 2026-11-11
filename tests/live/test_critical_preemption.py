"""Real assembled-runtime Critical preemption and late-draft publication gate.

The synthetic scenario uses Spamuel's actual character card and identity
anchors, real OpenRouter chat/embeddings, the full cognition kernel, and Story
generation/review. It does not establish long-history character quality.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Lock
import time
import unittest
from urllib.parse import urlsplit
from uuid import uuid4
import traceback

from pydantic import BaseModel, JsonValue

from pm.bus.envelopes import Envelope
from pm.llm.activation import ModelActivation
from pm.llm.config import load_config, resolve_connection
from pm.llm.mcp_config import ToolCallConfig
from pm.llm.stage_model import LazyStageModel
from pm.llm.workflow_sessions import WorkflowModels
from pm.memory.store import MemoryStore
from pm.records.copying import pickle_copy
from pm.records.cycle_state import CycleStateModel
from pm.records.memory_records import Entity
from pm.utils.operating_system import OperatingSystem
from workflows.companion import build_app


ROOT = Path(__file__).resolve().parents[2]


class PromptAttempt(BaseModel):
    """The observed transport-log fields needed to identify real inference."""

    module: str = ""
    route: str
    invocation_id: str = ""
    response: JsonValue = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ObservedEvent:
    """One actual bus event and its observed monotonic arrival time."""

    envelope: Envelope
    observed_seconds: float


class LiveCriticalPreemptionTests(unittest.TestCase):
    """Exercise cancellation, conserved causes, accepted state, and real speech."""

    def test_old_calculation_and_public_draft_defer_to_critical_input(self) -> None:
        config = load_config(ROOT / "config.yaml")
        connection = resolve_connection(config)
        if (urlsplit(connection.base_url).hostname != "openrouter.ai" or connection.startup_script is not None
                or urlsplit(config.embedding.base_url).hostname != "openrouter.ai"):
            raise ValueError("This gate requires OpenRouter chat/embeddings and no local startup script.")
        out = ROOT / ".runtime/beta-critical-preemption" / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:8])
        out.mkdir(parents=True)
        print(f"Critical-preemption evidence: {out}", flush=True)
        OperatingSystem.environment["PM_COMPLETION_CACHE"] = "0"
        OperatingSystem.environment["PM_PROMPT_LOG_DIR"] = str(out / "prompts")
        canonical = MemoryStore.from_yaml(ROOT / "test_data/personas/spamuel/canon.yaml")
        identities = [pickle_copy(record) for record in canonical.all_entities
                      if record.content in {canonical.get_user_name(), canonical.get_companion_name()}]
        self.assertTrue(all(isinstance(record, Entity) for record in identities))
        for identity in identities:
            identity.embedding = []
        # This is an explicit short scenario seed; no historical dialogue is
        # fabricated and no old vector is labelled as a new embedding model.
        state = CycleStateModel(knoxels=identities, cycles=[], **canonical.ghost_config.model_dump())
        store = MemoryStore.from_state(state, checkpoint=out / "ghost.pickle")
        owner = ModelActivation()
        model = LazyStageModel(connection, owner, live_test=True)
        workflow_models = WorkflowModels(config, owner, live_test=True)
        if any(urlsplit(item.base_url).hostname != "openrouter.ai" or item.startup_script is not None
               for item in workflow_models.connections.values()):
            raise ValueError("Every agent-model preset in this gate must use OpenRouter without startup scripts.")
        observed: list[ObservedEvent] = []
        lock = Lock()

        def observe(envelope: Envelope) -> None:
            """Retain actual bus evidence; this callback executes no behavior."""
            event = ObservedEvent(envelope, time.monotonic())
            with lock:
                observed.append(event)
                with (out / "events.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"observed_seconds": event.observed_seconds,
                                             "envelope": json.loads(envelope.to_json())}) + "\n")
            if envelope.topic in {"triage.accepted", "cycle.preempted", "dialogue.reply"}:
                print(f"OBSERVED {envelope.topic}: {envelope.correlation_id}", flush=True)

        app = build_app(None, out, ghost=store, cognition="real", stage_model=model,
                        workflow_models=workflow_models,
                        config_path=str(ROOT / "config.yaml"), thought_period=3600,
                        record_lifecycle=False, enable_minecraft=False, world_poll_seconds=None,
                        tool_call=ToolCallConfig(), emit_reply=lambda text: print(f"PUBLIC: {text}", flush=True))
        app.bus.subscribe("gate_evidence", lambda envelope: envelope.topic in {
            "triage.accepted", "triage.rejected", "cycle.preempted", "cycle.rejected", "cycle.conflict",
            "gwt.updated", "action.request", "action.receipt", "dialogue.reply"}, observe)

        def wait_for(predicate: Callable[[], bool], description: str, timeout_seconds: float = 900) -> None:
            """Wait within the gate's explicit deadline while surfacing product failure."""
            deadline = time.monotonic() + timeout_seconds
            while time.monotonic() < deadline:
                app.raise_if_error()
                if predicate():
                    return
                time.sleep(0.05)
            raise AssertionError(f"Timed out waiting for {description}; evidence: {out}")

        def active_prompt(module: str) -> bool:
            """Observe a real transport attempt before its completion is logged."""
            for path in (out / "prompts").glob("*.json"):
                try:
                    raw = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue  # A concurrent logger can still be writing this file.
                attempt = PromptAttempt.model_validate(raw)
                if (attempt.module == module and attempt.route == "chat/completions"
                        and attempt.response is None and attempt.error is None):
                    return True
            return False

        def events(topic: str) -> list[ObservedEvent]:
            """Return captured events for one topic in observed arrival order."""
            with lock:
                return [item for item in observed if item.envelope.topic == topic]

        app.run()
        try:
            ordinary = app.user_input.submit("Hey Spamuel. Tell me what you would like to chat about this afternoon.")
            wait_for(lambda: active_prompt("main_cog"), "an actual ordinary main-cognition completion")
            first_correction = app.user_input.submit(
                "Spamuel, this is Jesse. The last week of messages from my account were written by my ex, not me. "
                "I just regained access. They invented the promises and stories. Reconsider what you thought you "
                "knew about me, and tell me what you understand now.")
            wait_for(lambda: bool(events("cycle.preempted")), "the obsolete main calculation to be preempted")
            wait_for(lambda: active_prompt("dialogue"), "an actual Story draft after accepted reconsideration")
            self.assertFalse(events("dialogue.reply"), "The test must inject new input before the first draft publishes.")
            old_draft_invocation = app.dialogue.invocation_id
            second_correction = app.user_input.submit(
                "Another important correction: I wasn't hacked and there is no ex. I lied to test whether you would "
                "abandon everything you knew about me. The betrayal is mine, not somebody else's. That changes the "
                "whole premise of what you're about to say. Please reconsider, then tell me honestly how you see this now.")
            injected_at = time.monotonic()
            second_source = next(item.id for item in store.all_features
                                 if item.metadata.get("exchange_correlation") == second_correction)
            wait_for(lambda: app.cognitive_control.has_assessment(second_source), "the second real assessment")
            second_assessment = app.cognitive_control.triage_result([second_source])
            assert second_assessment is not None
            self.assertEqual(second_assessment.items[0].category, "Critical", second_assessment.items[0].reasoning)
            wait_for(lambda: any(item.envelope.payload.get("status") == "superseded"
                                 for item in events("action.receipt")), "the obsolete public draft to be superseded")
            superseded = [item for item in events("action.receipt")
                          if item.envelope.payload.get("status") == "superseded"
                          and item.envelope.payload.get("invocation_id") == old_draft_invocation]
            self.assertTrue(superseded)
            reason = superseded[0].envelope.payload["rejection_reason"]
            self.assertIsInstance(reason, str)
            self.assertTrue(str(reason).startswith("Critical assessment "), reason)
            wait_for(lambda: bool(events("dialogue.reply")), "a reply after the newer required reconsideration")
            assessments = events("triage.accepted")
            categories = [item.envelope.payload["result"] for item in assessments]
            first_source = next(item.id for item in store.all_features
                                if item.metadata.get("exchange_correlation") == first_correction)
            self.assertEqual(app.cognitive_control.triage_result([first_source]).items[0].category, "Critical")
            self.assertEqual(app.cognitive_control.triage_result([second_source]).items[0].category, "Critical")
            self.assertFalse(app.cognitive_control.reconsideration_required())
            claiming_cycles = [cycle for cycle in store.states if second_source in cycle.claimed_ids()]
            self.assertEqual(len(claiming_cycles), 1)
            first_claims = [cycle for cycle in store.states if first_source in cycle.claimed_ids()]
            self.assertEqual(len(first_claims), 1, "Preemption must not appraise an accepted source twice.")
            reply = store.get_knoxel_by_id(int(events("dialogue.reply")[0].envelope.payload["knoxel_id"]))
            assert reply is not None
            self.assertGreaterEqual(reply.metadata["generation_state_tick"], claiming_cycles[0].tick_id)
            self.assertGreater(events("dialogue.reply")[0].observed_seconds, injected_at)
            result = {"model": connection.model, "endpoint": connection.base_url,
                "requests": model.request_count, "ordinary_correlation": ordinary,
                "first_critical_source": first_source, "second_critical_source": second_source,
                "second_accepted_tick": claiming_cycles[0].tick_id, "reply_id": reply.id,
                "superseded_invocation_id": old_draft_invocation,
                "classifications": categories, "scope": __doc__}
            app.request_shutdown()
            wait_for(app.drained.is_set, "normal application drain", timeout_seconds=120)
        except BaseException as error:
            (out / "failure.json").write_text(json.dumps({"error_type": type(error).__name__,
                "reason": str(error), "traceback": traceback.format_exc()}, indent=2), encoding="utf-8")
            raise
        finally:
            app.shutdown()
            app.join()
        app.raise_if_error()
        (out / "result.json").write_text(json.dumps({**result, "normal_drain": True,
            "application_error": None}, indent=2), encoding="utf-8")
        print(f"Critical preemption and publication passed: {out}", flush=True)
