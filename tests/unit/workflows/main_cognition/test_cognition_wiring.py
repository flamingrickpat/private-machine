"""Direct W5 checkpoint-two wiring tests.

No live model and no module thread. A deterministic ``run_stage`` stands in for
the eight real children, and a deterministic ``story_complete`` stands in for
Story generation. The real parent kernel, the real module adapter, the real
MemoryStore commits, and the real bus are exercised.
"""
from __future__ import annotations

import asyncio
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from pm.records.memory_records import Feature
from pm.records.enums import FeatureType

from pm.bus.bus import BusSystem
from pm.bus.envelopes import Envelope, new_id
from workflows.main_cognition.host import MainCogLoop
from workflows.dialogue_story.host import StoryDialogue
from tests.support import ManualClock, Recorder
from pm.runs.workflow_module import PROJECT_ROOT
from workflows.main_cognition.host import build_main_cognition_tools
from workflows.main_cognition.pipeline import stable_action_id
from pm.memory.bash_view import MemoryBashTarget

REAL_MANIFEST = PROJECT_ROOT / "workflows" / "main_cognition" / "main_cognition.workflow.md"
STORY_MANIFEST = PROJECT_ROOT / "workflows" / "dialogue_story" / "dialogue_story.workflow.md"


def _small_ghost():
    from tests.unit.test_design_contracts import small_ghost
    return small_ghost()


def _store(ghost, content: str) -> int:
    return ghost.store_knoxel(Feature(
        content=content, feature_type=FeatureType.Dialogue, causal=True,
        override_source=ghost.get_user_name(),
    )).id


def _fake_reports(stimulus_id: int, candidate_id: int) -> dict:
    return {
        "triage": {"items": [{"knoxel_id": stimulus_id, "category": "Critical"}], "summary": "triaged"},
        "appraisal": {"feeling": "alarmed", "emotion_delta": {"anxiety": 0.4}, "summary": "appraised"},
        "context": {"memories": ["fire extinguisher"], "candidate_ids": [candidate_id], "summary": "context"},
        "codelets": {"proposals": [], "runs": [], "summary": "codelets"},
        "attention": {"selected_ids": [stimulus_id], "nominations": [], "summary": "attention"},
        "goals": {"changes": [], "summary": "goals"},
        "workspace": {
            "frame_reason": "the kitchen is on fire", "focus_ids": [stimulus_id],
            "control_mode": "reactive", "subjective_experience": "loud and urgent",
            "summary": "workspace",
        },
        "action_selection": {
            "actions": [{
                "action_type": "Reply", "workflow_name": "dialogue_story",
                "content": "warn Jesse about the fire", "arguments": {}, "depends_on": [],
            }],
            "summary": "action selection",
        },
    }


class MainCogLoopWiringTests(unittest.TestCase):
    def test_original_game_progress_cannot_start_a_second_run_for_its_active_request(self):
        from workflows.main_cognition.schemas import ActionDispatch
        capture = json.loads((PROJECT_ROOT / "test_data/captures/alpha/game-progress-duplicate-request.json").read_text(encoding="utf-8"))
        ghost = _small_ghost()
        original = capture["started"]["payload"]["receipt"]
        ghost.managed_runs[original["run_id"]] = original
        stimulus_id = _store(ghost, "A game task reported current progress.")
        reports = _fake_reports(stimulus_id, stimulus_id)
        proposed = capture["duplicate_request"]["payload"]
        reports["action_selection"]["actions"] = [{"action_type": "ToolCall",
            "workflow_name": proposed["workflow_name"], "content": proposed["content"],
            "arguments": proposed["arguments"], "depends_on": []}]
        progress = capture["progress"]
        batch = [Envelope(id=progress["id"], source=progress["source"], kind="telemetry",
            topic=progress["topic"], correlation_id=progress["correlation_id"], payload=progress["payload"])]
        with TemporaryDirectory() as directory:
            bus = BusSystem()
            module = MainCogLoop("main_cog", bus, self.fail, ManualClock(), None,
                working_target=MemoryBashTarget(ghost, "read"), workflow=REAL_MANIFEST,
                tools_factory=lambda environment, batch: {"run_stage": lambda name, payload: reports[name]},
                runtime_root=Path(directory), ghost=ghost)
            requests, receipts = Recorder(bus, "action.request"), Recorder(bus, "action.receipt")
            module.handle_batch(batch)
            self.assertFalse(requests.events)
            self.assertTrue(any(row.payload["status"] == "rejected" and "already owns" in row.payload["rejection_reason"]
                                for row in receipts.events))
            dispatch = ActionDispatch(action_id="new-choice", workflow_name="minecraft_project")
            self.assertIsNone(module._active_game_request(dispatch, "different-request", batch))
            direct_update = ActionDispatch(action_id="existing-run-update", action_type="ToolCall",
                workflow_name="minecraft_project", content="Change the active footprint.",
                arguments={"run_id": original["run_id"], "instruction": "Use five by seven."})
            self.assertIsNone(module._active_game_request(direct_update, original["parent_correlation"], batch))
            user = Envelope(id=new_id(), source="user_input", kind="data", topic="user.input",
                            correlation_id=original["parent_correlation"], payload={"knoxel_id": stimulus_id})
            self.assertIsNone(module._active_game_request(dispatch, original["parent_correlation"], [user]))
            ghost.managed_runs[original["run_id"]] = {**original, "state": "completed"}
            self.assertIsNone(module._active_game_request(dispatch, original["parent_correlation"], batch))

    def test_mixed_user_failure_releases_the_world_sources_original_correlation(self):
        from workflows.main_cognition.stage_support import StageOutputError
        ghost = _small_ghost()
        user_id = _store(ghost, "What changed in the world?")
        world = ghost.store_knoxel(Feature(content="A retained world observation changed.",
            feature_type=FeatureType.WorldEvent, causal=True, override_source="world_events"))
        def failed_stage(name, payload):
            raise StageOutputError("The retained model result failed its protocol contract.")
        with TemporaryDirectory() as directory:
            bus = BusSystem()
            module = MainCogLoop("main_cog", bus, self.fail, ManualClock(), None,
                working_target=MemoryBashTarget(ghost, "read"), workflow=REAL_MANIFEST,
                tools_factory=lambda environment, batch: {"run_stage": failed_stage},
                runtime_root=Path(directory), ghost=ghost)
            receipts = Recorder(bus, "action.receipt")
            module.handle_batch([
                Envelope(id=new_id(), source="user_input", kind="data", topic="user.input",
                         correlation_id="user-request", payload={"knoxel_id": user_id}),
                Envelope(id=new_id(), source="world_events", kind="data", topic="minecraft.world",
                         correlation_id="world-observation", payload={"knoxel_id": world.id}),
            ])
            rejected = [row for row in receipts.events if row.payload.get("recovery_outcome") == "cycle_rejected"]
            self.assertEqual([row.correlation_id for row in rejected], ["world-observation"])
            self.assertTrue(any(row.correlation_id == "user-request" for row in receipts.events))

    def test_real_parent_commits_gwt_and_publishes_a_typed_dispatch(self):
        ghost = _small_ghost()
        stimulus_id = _store(ghost, "the kitchen is on fire")
        candidate_id = _store(ghost, "Jesse owns a fire extinguisher")
        reports = _fake_reports(stimulus_id, candidate_id)

        def run_stage(name, payload):
            return dict(reports[name])

        with TemporaryDirectory() as tmp:
            bus = BusSystem()
            errors: list[BaseException] = []
            module = MainCogLoop(
                "main_cog", bus, errors.append, ManualClock(), None,
                working_target=MemoryBashTarget(ghost, "read"),
                workflow=REAL_MANIFEST,
                tools_factory=lambda environment, batch: {"run_stage": run_stage},
                runtime_root=Path(tmp) / "runtime",
                ghost=ghost,
            )
            bus.subscribe(module.module_id, module.accepts, module.deliver)
            requests = Recorder(bus, "action.request")
            gwt = Recorder(bus, "gwt.updated")

            bus.put(Envelope(
                id=new_id(), source="user_input", kind="data", topic="user.input",
                correlation_id="corr-fire", payload={"knoxel_id": stimulus_id},
            ))
            self.assertTrue(module.step(block=False))

            self.assertEqual(errors, [])
            self.assertEqual(len(gwt.events), 1)
            self.assertEqual(len(requests.events), 1)
            event = requests.events[0]
            self.assertEqual(event.target, "dialogue_story")
            self.assertEqual(event.correlation_id, "corr-fire")
            self.assertEqual(
                event.payload["action_id"],
                stable_action_id(reports["action_selection"]["actions"][0], 0),
            )
            self.assertIn(stimulus_id, ghost.gwt_turns[ghost.current_tick_id].knoxel_ids)
            action = ghost.get_knoxel_by_id(event.payload["knoxel_id"])
            self.assertEqual(action.content, "warn Jesse about the fire")

    def test_rejected_dispatch_publishes_a_receipt_and_no_action(self):
        ghost = _small_ghost()
        stimulus_id = _store(ghost, "hello")
        candidate_id = _store(ghost, "a memory")
        reports = _fake_reports(stimulus_id, candidate_id)
        reports["action_selection"] = {
            "actions": [{
                "action_type": "Reply", "workflow_name": "dialogue_story",
                "content": "claim it started", "arguments": {}, "depends_on": ["action-99"],
            }],
            "summary": "action selection",
        }

        def run_stage(name, payload):
            return dict(reports[name])

        with TemporaryDirectory() as tmp:
            bus = BusSystem()
            errors: list[BaseException] = []
            module = MainCogLoop(
                "main_cog", bus, errors.append, ManualClock(), None,
                working_target=MemoryBashTarget(ghost, "read"),
                workflow=REAL_MANIFEST,
                tools_factory=lambda environment, batch: {"run_stage": run_stage},
                runtime_root=Path(tmp) / "runtime",
                ghost=ghost,
            )
            bus.subscribe(module.module_id, module.accepts, module.deliver)
            requests = Recorder(bus, "action.request")
            receipts = Recorder(bus, "action.receipt")
            bus.put(Envelope(
                id=new_id(), source="user_input", kind="data", topic="user.input",
                correlation_id="corr-x", payload={"knoxel_id": stimulus_id},
            ))
            self.assertTrue(module.step(block=False))

            self.assertEqual(requests.events, [])
            self.assertEqual(len(receipts.events), 1)
            self.assertEqual(receipts.events[0].payload["status"], "rejected")
            self.assertIn("action-99", receipts.events[0].payload["rejection_reason"])

    def test_workflow_question_enters_cognition_as_a_stimulus(self):
        ghost = _small_ghost()
        candidate_id = _store(ghost, "a memory")
        reports = _fake_reports(0, candidate_id)

        captured: dict = {}

        def run_stage(name, payload):
            captured[name] = payload
            if name == "triage":
                stimulus_id = int(payload["stimuli"][0]["knoxel_id"])
                return {"items": [{"knoxel_id": stimulus_id, "category": "Moderate"}], "summary": "triaged"}
            return dict(reports[name])

        with TemporaryDirectory() as tmp:
            bus = BusSystem()
            errors: list[BaseException] = []
            module = MainCogLoop(
                "main_cog", bus, errors.append, ManualClock(), None,
                working_target=MemoryBashTarget(ghost, "read"),
                workflow=REAL_MANIFEST,
                tools_factory=lambda environment, batch: {"run_stage": run_stage},
                runtime_root=Path(tmp) / "runtime",
                ghost=ghost,
            )
            bus.subscribe(module.module_id, module.accepts, module.deliver)
            question = Envelope(
                id=new_id(), source="managed_run", kind="telemetry", topic="workflow.question",
                correlation_id="corr-q", payload={"question_id": "q1", "question": "Start the task?"},
            )
            self.assertTrue(module.accepts(question))
            bus.put(question)
            self.assertTrue(module.step(block=False))

            self.assertEqual(errors, [])
            stimulus = captured["triage"]["stimuli"][0]
            self.assertEqual(stimulus["kind"], "workflow.question")
            self.assertIn("Start the task?", stimulus["text"])
            feature = ghost.get_knoxel_by_id(stimulus["knoxel_id"])
            self.assertEqual(feature.feature_type, FeatureType.WorldEvent)


class StoryDialogueWiringTests(unittest.TestCase):
    def _module(self, bus, errors, ghost, root, completion: str):
        async def story_complete(messages, **generation):
            return completion

        async def public_complete(messages, **generation):
            # Boundary fixture for the agentic review: a claimed saved note
            # with no task result is the one hard block.
            import json
            claimed = "I saved the note for you" in completion
            return json.dumps({"judgment": "block" if claimed else "accept",
                               "reason": "no task ran" if claimed else "grounded",
                               "false_task_result": claimed, "findings": []})

        module = StoryDialogue(
            "dialogue", bus, errors.append, ManualClock(), None,
            working_target=MemoryBashTarget(ghost, "read"),
            workflow=STORY_MANIFEST,
            tools_factory=lambda environment, batch: {"story_complete": story_complete, "public_complete": public_complete},
            runtime_root=root / "runtime",
            ghost=ghost,
        )
        bus.subscribe(module.module_id, module.accepts, module.deliver)
        return module

    def _action(self, correlation_id: str):
        return Envelope(
            id=new_id(), source="main_cog", kind="data", topic="action.request",
            correlation_id=correlation_id, target="dialogue_story",
            payload={
                "action_id": "action-0",
                "content": "answer Jesse",
                "stimuli": [{"knoxel_id": 0, "text": "the kitchen is on fire"}],
                "memories": ["Jesse owns a fire extinguisher."],
                "workspace": "loud and urgent",
            },
        )

    def test_accepted_story_is_committed_and_correlated(self):
        ghost = _small_ghost()
        with TemporaryDirectory() as tmp:
            bus = BusSystem()
            errors: list[BaseException] = []
            module = self._module(
                bus, errors, ghost, Path(tmp),
                'Spamuel says: "That sounds like a mess, Jesse."',
            )
            replies = Recorder(bus, "dialogue.reply")
            bus.put(self._action("corr-story"))
            self.assertTrue(module.step(block=False))

            self.assertEqual(errors, [])
            self.assertEqual(len(replies.events), 1)
            reply = replies.events[0]
            self.assertEqual(reply.correlation_id, "corr-story")
            feature = ghost.get_knoxel_by_id(reply.payload["knoxel_id"])
            self.assertEqual(feature.content, "That sounds like a mess, Jesse.")
            self.assertEqual(feature.feature_type, FeatureType.Dialogue)

    def test_blocked_story_is_not_committed(self):
        ghost = _small_ghost()
        with TemporaryDirectory() as tmp:
            bus = BusSystem()
            errors: list[BaseException] = []
            module = self._module(
                bus, errors, ghost, Path(tmp),
                'Spamuel says: "I saved the note for you, Jesse."',
            )
            replies = Recorder(bus, "dialogue.reply")
            receipts = Recorder(bus, "action.receipt")
            bus.put(self._action("corr-bad"))
            self.assertTrue(module.step(block=False))

            self.assertEqual(errors, [])
            self.assertEqual(len(replies.events), 1)
            recovery = ghost.get_knoxel_by_id(replies.events[0].payload["knoxel_id"])
            self.assertNotIn("I saved the note", recovery.content)
            self.assertEqual(recovery.metadata["recovery_outcome"], "rejected")
            self.assertEqual(len(receipts.events), 1)
            self.assertEqual(receipts.events[0].payload["status"], "rejected")


class ModuleToolsTests(unittest.TestCase):
    def test_run_stage_invokes_the_real_child_kernel(self):
        ghost = _small_ghost()
        stimulus_id = _store(ghost, "the kitchen is on fire")
        completion = (
            '{"items": [{"knoxel_id": %d, "category": "Critical", "input_kind": "message",'
            ' "urgency": 0.9, "relevance": 0.9, "reasoning": "fire", "evidence_ids": [%d]}]}'
            % (stimulus_id, stimulus_id)
        )

        def llm_complete(messages, **generation):
            return completion

        with TemporaryDirectory() as tmp:
            factory = build_main_cognition_tools(ghost, Path(tmp) / "runtime", llm_complete)
            tools = factory(None, None)
            report = tools["run_stage"]("triage", {
                "companion_name": "Spamuel",
                "user_name": "Jesse",
                "stimuli": [{"knoxel_id": stimulus_id, "source": "Jesse", "text": "the kitchen is on fire"}],
            })
            self.assertEqual([item["category"] for item in report["items"]], ["Critical"])


class LazyModelTests(unittest.TestCase):
    def test_activation_is_lazy_and_idempotent(self):
        from contextlib import contextmanager

        from pm.llm.stage_model import LazyStageModel
        from pm.llm.config import Connection

        class StubOwner:
            def __init__(self):
                self.activations = 0

            def is_ready(self, connection):
                return self.activations > 0

            async def activate(self, connection, *, live_test=False):
                self.activations += 1

            @contextmanager
            def use(self, connection):
                yield

        owner = StubOwner()
        connection = Connection(base_url="http://127.0.0.1:9/v1", model="qwen")
        model = LazyStageModel(connection, owner)
        self.assertFalse(model.activated)
        first = model.ensure_client()
        second = model.ensure_client()
        self.assertIs(first, second)
        self.assertEqual(owner.activations, 1)
        self.assertTrue(model.activated)


class MultiExchangeTests(unittest.TestCase):
    """Several real bus exchanges with deterministic cognition and Story."""

    def test_each_input_produces_a_correlated_reply(self):
        from workflows.public_speech.user_input import UserInputProxy

        ghost = _small_ghost()
        bus = BusSystem()
        errors: list[BaseException] = []
        counter = {"n": 0}

        def run_stage(name, payload):
            stimulus_id = int(payload["stimuli"][0]["knoxel_id"])
            candidates = [int(item["knoxel_id"]) for item in payload.get("memory_candidates", [])]
            candidate_id = candidates[0] if candidates else None
            context_ids = [candidate_id] if candidate_id is not None else []
            selected = ([candidate_id] if candidate_id is not None else []) + [stimulus_id]
            reports = {
                "triage": {"items": [{"knoxel_id": stimulus_id, "category": "Critical"}], "summary": "t"},
                "appraisal": {"feeling": "curious", "summary": "a"},
                "context": {"candidate_ids": context_ids, "summary": "c"},
                "codelets": {"proposals": [], "runs": [], "summary": "codelets"},
                "attention": {"selected_ids": selected, "nominations": [], "summary": "n"},
                "goals": {"changes": [], "summary": "g"},
                "workspace": {"frame_reason": "reply", "focus_ids": selected, "summary": "w"},
                "action_selection": {
                    "actions": [{"action_type": "Reply", "workflow_name": "dialogue_story",
                                 "content": "reply to Jesse", "arguments": {}, "depends_on": []}],
                    "summary": "s",
                },
            }
            return dict(reports[name])

        async def story_complete(messages, **generation):
            counter["n"] += 1
            return f'Spamuel says: "Reply number {counter["n"]}."'

        async def public_complete(messages, **generation):
            return '{"judgment": "accept", "reason": "grounded", "false_task_result": false, "findings": []}'

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            main_cog = MainCogLoop(
                "main_cog", bus, errors.append, ManualClock(), None,
                working_target=MemoryBashTarget(ghost, "read"),
                workflow=REAL_MANIFEST,
                tools_factory=lambda environment, batch: {"run_stage": run_stage},
                runtime_root=root / "main", ghost=ghost,
            )
            dialogue = StoryDialogue(
                "dialogue", bus, errors.append, ManualClock(), None,
                working_target=MemoryBashTarget(ghost, "read"),
                workflow=STORY_MANIFEST,
                tools_factory=lambda environment, batch: {"story_complete": story_complete, "public_complete": public_complete},
                runtime_root=root / "story", ghost=ghost,
            )
            replies: list[str] = []
            user_input = UserInputProxy("user_input", bus, errors.append, ManualClock(), ghost, replies.append)
            for module in (user_input, main_cog, dialogue):
                bus.subscribe(module.module_id, module.accepts, module.deliver)

            correlations = []
            for index in range(3):
                correlations.append(user_input.submit(f"message {index}"))
                self.assertTrue(main_cog.step(block=False))
                self.assertTrue(dialogue.step(block=False))
                user_input.step(block=False)

            self.assertEqual(errors, [])
            self.assertEqual(replies, [f"Reply number {index}." for index in range(1, 4)])
            self.assertEqual(len(set(correlations)), 3)
            public = [
                knoxel for knoxel in ghost.all_features
                if knoxel.content.startswith("Reply number")
            ]
            self.assertEqual(len(public), 3)


if __name__ == "__main__":
    unittest.main()
