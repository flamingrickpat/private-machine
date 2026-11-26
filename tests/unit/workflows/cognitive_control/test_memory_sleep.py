"""Exercise the memory-sleep checkpoint fence and recurring action identity."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pm.bus.bus import BusSystem
from pm.bus.envelopes import Envelope, new_envelope
from pm.app.control import RuntimeControl
from pm.host.application import Application
from pm.memory.reconsolidation.coordination import ReconsolidationMode
from pm.memory.store import MemoryStore
from pm.records.cycle_state import PersonaConfig
from pm.records.cycle_records import StimulusGroup
from pm.records.enums import ActionType
from pm.records.global_workspace import GwtTurnRecord
from pm.records.memory_records import ActionSchemaRecord
from pm.records.workspace import ShardKey
from pm.runs.run_receipt import ManagedRunReceipt
from pm.vclock import VirtualClock
from workflows.cognitive_control.host import CognitiveControl
from workflows.cognitive_control.schemas import WaitRequest
from workflows.main_cognition.phases.route import action_dispatches, resolve_conflicts
from workflows.main_cognition.schemas import ActionArguments, ActionDispatch, RouteChoice, RoutedDecision
from workflows.public_speech.user_input import UserInputProxy


class MemorySleepTests(unittest.TestCase):
    def test_a_failed_older_run_cannot_wake_a_later_sleep(self) -> None:
        ghost = MemoryStore(PersonaConfig(companion_name="Spamuel", user_name="Jesse"))
        app = Application(BusSystem(), [])
        control = CognitiveControl("cognitive_control", app, None, ghost=ghost)
        terminals: list[Envelope] = []
        for name in ("older", "current"):
            batch = ghost.begin_batch(name, new_tick=True)
            action = batch.create(ActionSchemaRecord(content="Rest while sorting memories.", action_type=ActionType.ToolCall))
            batch.set_gwt(GwtTurnRecord(tick_id=batch.tick_id, knoxel_ids=[action.id]))
            receipt = ManagedRunReceipt(run_id=name, action_id=name, workflow_name="memory_reconsolidation",
                parent_correlation="", state="failed", latest_event="failed", artifact_path="", authority_record_id=action.id)
            batch.set_run_receipt(name, dict(receipt.to_dict()))  # pm-lint: allow PM005 -- persist the managed receipt in the store's wire format
            control.commit(batch, WaitRequest(memory_action_id=action.id), control.generation)
            terminals.append(new_envelope("managed_run", "telemetry", "workflow.event", "",
                {"run_id": name, "event": "failed", "workflow_name": "memory_reconsolidation"}))  # pm-lint: allow PM005 -- bus event wire payload
            if name == "older":
                control.wake_before_input()
        previous_generation = control.generation
        control.observe(terminals[0])
        self.assertTrue(control.waiting())
        self.assertEqual(control.generation, previous_generation)
        control.observe(terminals[1])
        control.observe(terminals[1])
        self.assertFalse(control.waiting())
        self.assertEqual(control.generation, previous_generation + 1)
        wait = next(row for row in reversed(ghost.all_features) if row.metadata.get("control_kind") == "inference_wait")
        self.assertEqual(wait.metadata["wake_reason"], "memory work failed")

    def test_operator_deadline_distinguishes_a_timer_from_a_worker_owned_sleep(self) -> None:
        now = datetime(2026, 11, 24, 16)
        ghost = MemoryStore(PersonaConfig(companion_name="Spamuel", user_name="Jesse"))
        app = Application(BusSystem(), [], clock=VirtualClock(now))
        app.ghost = ghost
        intake = UserInputProxy("input", app, ghost, print)
        operator = RuntimeControl(app, ghost, intake, Path.cwd())
        control = CognitiveControl("cognitive_control", app, None, ghost=ghost)
        timer = ghost.begin_batch("timer", new_tick=True)
        timer.set_gwt(GwtTurnRecord(tick_id=timer.tick_id))
        control.commit(timer, WaitRequest(seconds=45), control.generation)
        deadline = operator.next_due()
        assert deadline is not None
        self.assertEqual((datetime.fromisoformat(deadline).replace(tzinfo=None) - now).total_seconds(), 45)
        question = new_envelope("managed_run", "telemetry", "workflow.question", "timer-wake",
            {"question": "Which memory needs attention?"})  # pm-lint: allow PM005 -- bus event wire payload
        control.observe(question)
        memory = ghost.begin_batch("memory", new_tick=True)
        action = memory.create(ActionSchemaRecord(content="Sleep and organize memories.", action_type=ActionType.ToolCall))
        memory.set_gwt(GwtTurnRecord(tick_id=memory.tick_id, knoxel_ids=[action.id]))
        control.commit(memory, WaitRequest(memory_action_id=action.id), control.generation, envelope_ids={question.id})
        self.assertTrue(control.waiting())
        self.assertIsNone(operator.next_due())

    def test_input_retries_a_failed_wake_checkpoint_before_recording_the_message(self) -> None:
        with TemporaryDirectory() as home:
            checkpoint = Path(home) / "character.pickle"
            persona = PersonaConfig(companion_name="Spamuel", user_name="Jesse")
            ghost = MemoryStore(persona, checkpoint=checkpoint)
            ghost.accept_import()
            clock = VirtualClock(datetime(2026, 11, 24, 16))
            application = Application(BusSystem(), [], clock=clock)
            control = CognitiveControl("control", application, None, ghost=ghost)
            intake = UserInputProxy("input", application, ghost, print)
            intake.cognitive_control = control
            batch = ghost.begin_batch("sleep-choice", new_tick=True)
            action = batch.create(ActionSchemaRecord(content="Sleep and sort through the day.", action_type=ActionType.ToolCall))
            batch.set_gwt(GwtTurnRecord(tick_id=batch.tick_id, knoxel_ids=[action.id]))
            control.commit(batch, WaitRequest(memory_action_id=action.id), control.generation)
            ghost.save()
            generation_before_wake = control.generation

            blocked = Path(home) / "blocked.pickle"
            blocked.mkdir()
            ghost.checkpoint = blocked
            message = "Good morning!"
            with self.assertRaises(OSError):
                intake.submit(message, delivery_id="one-message")
            self.assertFalse(any(row.content == message for row in ghost.all_features))
            self.assertTrue(control.waiting())
            sleep = control.reconsolidation.sleep
            self.assertIsNotNone(sleep)
            assert sleep is not None
            self.assertTrue(sleep.ending)

            ghost.checkpoint = checkpoint
            correlation = intake.submit(message, delivery_id="one-message")
            self.assertEqual(intake.submit(message, delivery_id="one-message"), correlation)
            self.assertEqual(control.generation, generation_before_wake + 1)
            self.assertIsNone(control.reconsolidation.sleep)
            self.assertEqual(sum(row.content == message for row in ghost.all_features), 1)
            raw = MemoryStore.from_checkpoint(checkpoint, raw=True)
            self.assertFalse(any(row.content == message for row in raw.all_features))
            wait = next(row for row in raw.all_features if row.metadata.get("control_kind") == "inference_wait")
            self.assertEqual(wait.metadata["status"], "awake")
            self.assertEqual(wait.metadata["wake_reason"], "user.input")

    def test_equivalent_memory_decisions_can_recur_at_distinct_main_ticks(self) -> None:
        route = RouteChoice(workflow_name="memory_reconsolidation", action_type="ToolCall",
                            arguments=ActionArguments(reconsolidation_mode=ReconsolidationMode.Sleep))
        decision = RoutedDecision(decision="I should sleep.", route=route)
        first = action_dispatches([decision], decision_tick=12)[0]
        replay = action_dispatches([decision], decision_tick=12)[0]
        later = action_dispatches([decision], decision_tick=24)[0]
        self.assertEqual(first.action_id, replay.action_id)
        self.assertNotEqual(first.action_id, later.action_id)

    def test_sleep_and_foreground_dispatch_cannot_start_together(self) -> None:
        partner = StimulusGroup(key=ShardKey(kind="entity", ref="2"), category="Critical")
        self_group = StimulusGroup(key=ShardKey(), category="Moderate")
        sleep = ActionDispatch(action_id="rest", group=self_group.key, workflow_name="memory_reconsolidation",
            action_type="ToolCall", arguments=ActionArguments(reconsolidation_mode=ReconsolidationMode.Sleep),
            status="dispatched")
        reply = ActionDispatch(action_id="answer", group=partner.key, workflow_name="dialogue_story",
                               action_type="Reply", status="dispatched")
        memory, speech = resolve_conflicts([sleep, reply], [self_group, partner])
        self.assertEqual((memory.status, speech.status), ("rejected", "dispatched"))
        self_group.category, partner.category = "Critical", "Moderate"
        memory, speech = resolve_conflicts([sleep, reply], [self_group, partner])
        self.assertEqual((memory.status, speech.status), ("dispatched", "rejected"))
        sleep.arguments.reconsolidation_mode = ReconsolidationMode.Background
        memory, speech = resolve_conflicts([sleep, reply], [self_group, partner])
        self.assertEqual((memory.status, speech.status), ("dispatched", "dispatched"))
