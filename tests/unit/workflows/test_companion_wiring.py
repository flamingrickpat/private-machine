"""Application wiring for W6: managed runs, SelfManagement, and Assistant.

Builds the real application path with a deterministic stage-model boundary.
No network call, module thread, or model endpoint is required.
"""
from __future__ import annotations

import time
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pm.bus.envelopes import Envelope
from workflows.companion import build_app
from tests.support import Recorder, fresh_ghost, load_seed_state


def _wait_for(predicate, timeout: float = 20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class StubStageModel:
    """A deterministic completion boundary for wiring tests. No inference."""

    def __init__(self, completions=None):
        self.completions = list(completions or [])
        self.activations = 0

    def complete(self, messages, **generation):
        return "{}"

    def measure_context(self, messages, **generation):
        """Refuse inference measurement in wiring tests that require no model."""
        raise AssertionError("Management wiring must not measure an inference request.")

    def build_async_tool(self):
        self.activations += 1
        async def complete_async(messages, **generation):
            if "checks their own words" in messages[0].content.text:
                return '{"judgment": "accept", "reason": "grounded", "false_task_result": false, "findings": []}'
            return "Spamuel says: \"wired.\""

        return complete_async

    def build_tool_completion(self):
        async def complete_tool(messages, tools=None, **generation):
            return self.completions.pop(0)

        return complete_tool


class ManagementWiringTests(unittest.TestCase):
    def test_self_management_action_starts_a_consolidated_run(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            ghost = fresh_ghost(load_seed_state())
            app = build_app(
                None, root, ghost=ghost, cognition="real",
                stage_model=StubStageModel(), clock=time.monotonic,
            )
            receipts = Recorder(app.bus, "action.receipt")
            app.self_management.mailbox.put(Envelope(
                "req", "test", "data", "action.request", "corr-mgmt",
                {
                    "action_id": "action-0",
                    "workflow_name": "self_management",
                    "content": "start the local note task",
                    "arguments": {
                        "operation": "start",
                        "workflow_name": "local_task",
                        "instructions": "wire the run",
                    },
                },
                target="self_management",
            ))
            self.assertTrue(app.self_management.step())
            self.assertEqual(len(receipts.events), 1)
            self.assertEqual(receipts.events[0].payload["status"], "accepted")
            run_id = receipts.events[0].payload["receipt"]["run_id"]
            self.assertTrue(
                _wait_for(lambda: app.supervisor.status(run_id)["state"] == "completed"),
                "managed run did not complete",
            )
            # Terminal runs consolidate their real text artifacts automatically.
            self.assertTrue(
                _wait_for(lambda: ghost.consolidations),
                "terminal run did not consolidate its artifacts",
            )
            # The consolidation is evidence-backed and nonempty.
            consolidation = next(iter(ghost.consolidations.values()))
            self.assertEqual(consolidation["run_id"], run_id)
            self.assertTrue(consolidation["memory_ids"])
            app.supervisor.join_all(timeout=10)

    def test_assistant_action_commits_a_reply(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            ghost = fresh_ghost(load_seed_state())
            completions = [{"choices": [{"message": {"content": "Wired and watching."}}]}]
            app = build_app(
                None, root, ghost=ghost, cognition="real",
                stage_model=StubStageModel(completions), clock=time.monotonic,
            )
            # This local text-wiring check has no game evidence. Do not let
            # an independent live fixture change its capability input mid-run.
            app.assistant.capability_provider = None
            replies = Recorder(app.bus, "dialogue.reply")
            from pm.records.memory_records import new_feature
            from pm.records.enums import FeatureType
            stimulus = new_feature(ghost, "say hello", FeatureType.Dialogue, source=ghost.get_user_name())
            app.assistant.mailbox.put(Envelope(
                "req", "test", "data", "action.request", "corr-assistant",
                {
                    "action_id": "action-1",
                    "workflow_name": "dialogue_assistant",
                    "content": "say hello",
                    "arguments": {},
                    "stimuli": [{"knoxel_id": stimulus.id, "text": "say hello"}],
                },
                target="dialogue_assistant",
            ))
            self.assertTrue(app.assistant.step())
            self.assertEqual(len(replies.events), 1)
            knoxel = ghost.get_knoxel_by_id(replies.events[0].payload["knoxel_id"])
            self.assertEqual(knoxel.content, "Wired and watching.")

    def test_real_build_contacts_no_model(self):
        with TemporaryDirectory() as tmp:
            ghost = fresh_ghost(load_seed_state())
            model = StubStageModel()
            app = build_app(
                None, Path(tmp), ghost=ghost, cognition="real",
                stage_model=model, clock=time.monotonic,
            )
            self.assertIsNotNone(app.assistant)
            self.assertIsNotNone(app.self_management)
            self.assertIsNotNone(app.supervisor)
            self.assertIsNotNone(app.consolidator)
            self.assertIsNotNone(app.steward)
            self.assertTrue(app.steward.dormant)
            self.assertEqual(model.activations, 0)


    def test_steward_module_is_dormant_and_refuses(self):
        with TemporaryDirectory() as tmp:
            ghost = fresh_ghost(load_seed_state())
            app = build_app(
                None, Path(tmp), ghost=ghost, cognition="real",
                stage_model=StubStageModel(), clock=time.monotonic,
            )
            receipts = Recorder(app.bus, "action.receipt")
            app.steward_module.mailbox.put(Envelope(
                "req", "test", "data", "action.request", "corr-steward",
                {
                    "workflow_name": "steward",
                    "arguments": {"operation": "status", "run_id": "missing"},
                },
                target="steward",
            ))
            self.assertTrue(app.steward_module.step())
            self.assertEqual(len(receipts.events), 1)
            self.assertEqual(receipts.events[0].payload["status"], "rejected")
            self.assertIn("dormant", receipts.events[0].payload["summary"])


if __name__ == "__main__":
    unittest.main()
