"""Two ordinary messages during actual main calculation on OpenRouter.

This narrow engineering gate establishes neutral guidance through one real cold start.
It establishes overlapping selection, source accounting, and actual speech.
It does not establish sustained character quality or private inquiry continuity.
"""
from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Lock
import time
import traceback
import unittest
from urllib.parse import urlsplit
from uuid import uuid4

from pm.bus.envelopes import Envelope
from pm.llm.activation import ModelActivation
from pm.llm.config import load_config, resolve_connection
from pm.llm.mcp_config import ToolCallConfig
from pm.llm.stage_model import LazyStageModel
from pm.llm.workflow_sessions import WorkflowModels
from pm.memory.store import MemoryStore
from pm.records.copying import pickle_copy
from pm.records.cycle_state import CycleStateModel
from pm.records.base import RecordBase
from pm.utils.operating_system import OperatingSystem
from tests.live.test_critical_preemption import ObservedEvent, PromptAttempt
from workflows.companion import build_app
from workflows.interaction.records import interaction_receipts
from workflows.main_cognition.schemas import ActionRequest
from workflows.cognitive_control.host import CognitiveControl
from workflows.public_speech.user_input import UserInputProxy


ROOT = Path(__file__).resolve().parents[2]


class LiveAsyncInteractionTests(unittest.TestCase):
    """Use actual kernels and provider receipts, without simulation rating agents."""

    def test_two_ordinary_inputs_respond_before_background_main_acceptance(self) -> None:
        config = load_config(ROOT / "config.yaml")
        connection = resolve_connection(config)
        if (urlsplit(connection.base_url).hostname != "openrouter.ai" or connection.startup_script is not None
                or urlsplit(config.embedding.base_url).hostname != "openrouter.ai"):
            raise ValueError("This gate requires OpenRouter chat/embeddings and no local startup script.")
        out = ROOT / ".runtime/beta-b03/live" / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:8])
        out.mkdir(parents=True)
        print(f"Async interaction evidence: {out}", flush=True)
        OperatingSystem.environment["PM_COMPLETION_CACHE"] = "0"
        OperatingSystem.environment["PM_PROMPT_LOG_DIR"] = str(out / "prompts")
        canonical = MemoryStore.from_yaml(ROOT / "test_data/personas/spamuel/canon.yaml")
        identities: list[RecordBase] = [pickle_copy(record) for record in canonical.all_entities
                                       if record.content in {canonical.get_user_name(), canonical.get_companion_name()}]
        for identity in identities:
            identity.embedding = []
        seed = CycleStateModel(knoxels=identities, cycles=[], **canonical.ghost_config.model_dump())
        store = MemoryStore.from_state(seed, checkpoint=out / "ghost.pickle")
        owner = ModelActivation()
        model = LazyStageModel(connection, owner, live_test=True)
        workflow_models = WorkflowModels(config, owner, live_test=True)
        if any(urlsplit(item.base_url).hostname != "openrouter.ai" or item.startup_script is not None
               for item in workflow_models.connections.values()):
            raise ValueError("Every agent preset must use OpenRouter without a startup script.")
        observed: list[ObservedEvent] = []
        lock = Lock()

        def observe(envelope: Envelope) -> None:
            """Capture actual bus timing without altering scheduling or behavior."""
            event = ObservedEvent(envelope, time.monotonic())
            with lock:
                observed.append(event)
                with (out / "events.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"observed_seconds": event.observed_seconds,
                                             "envelope": json.loads(envelope.to_json())}) + "\n")
            if envelope.topic in {"triage.accepted", "interaction.accepted", "gwt.updated", "dialogue.reply"}:
                print(f"OBSERVED {envelope.topic}: {envelope.correlation_id}", flush=True)

        app = build_app(None, out, ghost=store, cognition="real", stage_model=model,
            workflow_models=workflow_models, config_path=str(ROOT / "config.yaml"), thought_period=3600,
            record_lifecycle=False, enable_minecraft=False, world_poll_seconds=None,
            tool_call=ToolCallConfig(), emit_reply=lambda text: print(f"PUBLIC: {text}", flush=True))
        app.bus.subscribe("b03_evidence", lambda envelope: envelope.topic in {
            "triage.accepted", "triage.rejected", "interaction.accepted", "interaction.superseded",
            "interaction.rejected", "cycle.conflict", "cycle.rejected", "gwt.updated",
            "action.request", "action.receipt", "dialogue.reply"}, observe)
        user_input = next(module for module in app.modules if isinstance(module, UserInputProxy))
        control = next(module for module in app.modules if isinstance(module, CognitiveControl))

        def wait_for(predicate: Callable[[], bool], description: str, timeout_seconds: float = 900) -> None:
            """Wait within this explicit gate deadline and expose runtime errors."""
            deadline = time.monotonic() + timeout_seconds
            while time.monotonic() < deadline:
                app.raise_if_error()
                if predicate():
                    return
                time.sleep(0.05)
            raise AssertionError(f"Timed out waiting for {description}. Evidence: {out}")

        def active_main_prompt() -> bool:
            """Observe the actual in-flight provider call before injecting the follow-up."""
            for path in (out / "prompts").glob("*.json"):
                try:
                    raw = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue  # The logger can still be writing this external file.
                attempt = PromptAttempt.model_validate(raw)
                if attempt.module == "main_cog" and attempt.response is None and attempt.error is None:
                    return True
            return False

        def events(topic: str) -> list[ObservedEvent]:
            """Read one topic's actual captured events in observed order."""
            with lock:
                return [event for event in observed if event.envelope.topic == topic]

        app.run()
        try:
            foundation = user_input.submit("Hey Spamuel. Let's invent a tiny sandwich shop. What do you want to call it?")
            wait_for(lambda: bool(events("dialogue.reply")), "the actual cold-start main cycle and its public reply")
            baseline_ticks = {state.tick_id for state in store.states}
            first_started = time.monotonic()
            first = user_input.submit("That name is staying. Give its first sandwich an equally ridiculous name.")
            wait_for(active_main_prompt, "an actual main-cognition completion")
            second = user_input.submit("And put something unnecessarily dramatic on the imaginary menu. Still just playing.")
            source_ids = [record.id for record in store.all_features
                          if record.metadata.get("exchange_correlation") in {first, second}
                          and record.metadata.get("exchange_status") is not None]
            self.assertEqual(len(source_ids), 2)
            wait_for(lambda: any(event.observed_seconds >= first_started for event in events("dialogue.reply")),
                     "ordinary public speech during background calculation")
            first_visible = next(event.observed_seconds for event in events("dialogue.reply")
                                 if event.observed_seconds >= first_started)
            wait_for(lambda: all(any(key in state.claimed_ids() for state in store.states) for key in source_ids),
                     "main appraisal of both original sources")
            main_events = [event for event in events("gwt.updated")
                           if event.observed_seconds >= first_started
                           and event.envelope.payload.get("slot", "main") == "main"]
            self.assertTrue(main_events)
            self.assertLess(first_visible, main_events[0].observed_seconds,
                            "Ordinary speech must finish before the active full main calculation accepts.")
            receipts = interaction_receipts(store.recovery_snapshot(capture=False))
            self.assertTrue(all(key in receipts for key in source_ids))
            for key in source_ids:
                assessment = control.triage_result([key])
                assert assessment is not None
                self.assertNotEqual(assessment.items[0].category, "Critical")
                claims = [state.tick_id for state in store.states if key in state.claimed_ids()]
                self.assertEqual(len(claims), 1, "Original dialogue must receive one main appraisal.")
            for event in events("action.request"):
                if event.envelope.source == "main_cog":
                    request = ActionRequest.model_validate(event.envelope.payload)
                    self.assertFalse(any(item.knoxel_id in source_ids for item in request.stimuli))
            app.request_shutdown()
            wait_for(app.drained.is_set, "normal application drain", timeout_seconds=120)
            result = {"model": connection.model, "endpoint": connection.base_url,
                "provider_hardware": "Unobserved provider-managed hardware", "requests": model.request_count,
                "source_ids": source_ids, "reply_ids": [event.envelope.payload["knoxel_id"] for event in events("dialogue.reply")
                              if event.observed_seconds >= first_started],
                "new_ticks": [state.tick_id for state in store.states if state.tick_id not in baseline_ticks],
                "first_visible_seconds": first_visible - first_started,
                "total_seconds": time.monotonic() - first_started, "normal_drain": True,
                "foundation_correlation": foundation,
                "fixture": "Explicit synthetic two-identity seed with Spamuel's actual persona card and one real cold start",
                "scope": __doc__}
        except BaseException as error:
            (out / "failure.json").write_text(json.dumps({"error_type": type(error).__name__, "reason": str(error),
                "traceback": traceback.format_exc()}, indent=2), encoding="utf-8")
            raise
        finally:
            app.shutdown()
            app.join()
        app.raise_if_error()
        (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Ordinary async gate passed: {out}", flush=True)
