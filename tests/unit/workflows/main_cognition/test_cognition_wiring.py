"""Direct W5 checkpoint-two wiring tests.

No live model and no module thread. A scripted ``llm_complete`` answers the
model calls of the real phases, and a deterministic ``story_complete`` stands
in for Story generation. The real parent kernel, the real module adapter, the real
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
from workflows.main_cognition.phases.route import stable_action_id
from workflows.main_cognition.schemas import ActionIntent
from tests.unit.workflows.main_cognition.cycle_support import ScriptedModel, scripted_tools
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
        "appraisal": {"feeling": "alarmed", "emotion_delta": {"fear": 0.4}, "summary": "appraised"},
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
                tools_factory=scripted_tools(ScriptedModel(reports=reports)),
                runtime_root=Path(directory), ghost=ghost,
                workflow_catalogue=[{"name": proposed["workflow_name"], "description": "game project"}])
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
        from workflows.main_cognition.model_json import StageOutputError
        ghost = _small_ghost()
        user_id = _store(ghost, "What changed in the world?")
        world = ghost.store_knoxel(Feature(content="A retained world observation changed.",
            feature_type=FeatureType.WorldEvent, causal=True, override_source="world_events"))
        def failed_model(messages, **generation):
            raise StageOutputError("The retained model result failed its protocol contract.")
        with TemporaryDirectory() as directory:
            bus = BusSystem()
            module = MainCogLoop("main_cog", bus, self.fail, ManualClock(), None,
                working_target=MemoryBashTarget(ghost, "read"), workflow=REAL_MANIFEST,
                tools_factory=lambda environment, batch: {**ScriptedModel().tools(), "llm_complete": failed_model},
                runtime_root=Path(directory), ghost=ghost)
            receipts = Recorder(bus, "action.receipt")
            with self.assertRaisesRegex(StageOutputError, "protocol contract"):
                module.handle_batch([
                    Envelope(id=new_id(), source="user_input", kind="data", topic="user.input",
                             correlation_id="user-request", payload={"knoxel_id": user_id}),
                    Envelope(id=new_id(), source="world_events", kind="data", topic="minecraft.world",
                             correlation_id="world-observation", payload={"knoxel_id": world.id}),
                ])
            self.assertEqual(receipts.events, [])

    def test_exhausted_format_repair_keeps_the_user_cause_for_the_next_invocation(self):
        from workflows.main_cognition.model_json import ModelFormatError
        ghost = _small_ghost()
        stimulus_id = _store(ghost, "the kitchen is on fire")
        candidate_id = _store(ghost, "Jesse owns a fire extinguisher")
        reports = _fake_reports(stimulus_id, candidate_id)
        calls = []
        scripted = ScriptedModel(reports=reports)

        def model(messages, **generation):
            calls.append(1)
            if len(calls) == 1:
                module.run_stop.set()
                raise ModelFormatError("Captured model syntax stayed invalid.")
            return scripted(messages, **generation)

        with TemporaryDirectory() as tmp:
            bus = BusSystem()
            errors = []
            module = MainCogLoop("main_cog", bus, errors.append, ManualClock(), None,
                working_target=MemoryBashTarget(ghost, "read"), workflow=REAL_MANIFEST,
                tools_factory=lambda environment, batch: {**scripted.tools(), "llm_complete": model},
                runtime_root=Path(tmp), ghost=ghost)
            rejected = Recorder(bus, "cycle.rejected")
            requests = Recorder(bus, "action.request")
            module.handle_batch([Envelope(id=new_id(), source="user_input", kind="data", topic="user.input",
                correlation_id="retained-user", payload={"knoxel_id": stimulus_id})])
            self.assertEqual(len(rejected.events), 1)
            self.assertEqual(rejected.events[0].correlation_id, "retained-user")
            self.assertEqual(len(requests.events), 1)
            self.assertEqual(requests.events[0].correlation_id, "retained-user")
            self.assertEqual(len(list(Path(tmp).rglob("model-output-rejection.json"))), 1)
            self.assertEqual(errors, [])

    def test_explicit_run_cancellation_prevents_a_format_retry(self):
        from workflows.main_cognition.model_json import ModelFormatError
        ghost = _small_ghost()
        stimulus_id = _store(ghost, "the kitchen is on fire")
        calls = []

        def model(messages, **generation):
            calls.append(1)
            module.cancel_run()
            raise ModelFormatError("Captured model syntax stayed invalid.")

        with TemporaryDirectory() as tmp:
            bus = BusSystem()
            module = MainCogLoop("main_cog", bus, self.fail, ManualClock(), None,
                working_target=MemoryBashTarget(ghost, "read"), workflow=REAL_MANIFEST,
                tools_factory=lambda environment, batch: {**ScriptedModel().tools(), "llm_complete": model},
                runtime_root=Path(tmp), ghost=ghost)
            requests = Recorder(bus, "action.request")
            module.handle_batch([Envelope(id=new_id(), source="user_input", kind="data", topic="user.input",
                correlation_id="cancelled-user", payload={"knoxel_id": stimulus_id})])
            self.assertEqual(len(calls), 1)
            self.assertEqual(requests.events, [])
            self.assertEqual(ghost.gwt_shard("main"), [])

    def test_real_parent_commits_gwt_and_publishes_a_typed_dispatch(self):
        ghost = _small_ghost()
        stimulus_id = _store(ghost, "the kitchen is on fire")
        candidate_id = _store(ghost, "Jesse owns a fire extinguisher")
        reports = _fake_reports(stimulus_id, candidate_id)

        with TemporaryDirectory() as tmp:
            bus = BusSystem()
            errors: list[BaseException] = []
            module = MainCogLoop(
                "main_cog", bus, errors.append, ManualClock(), None,
                working_target=MemoryBashTarget(ghost, "read"),
                workflow=REAL_MANIFEST,
                tools_factory=scripted_tools(ScriptedModel(reports=reports)),
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
                stable_action_id(ActionIntent.model_validate(reports["action_selection"]["actions"][0]), 0),
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

        with TemporaryDirectory() as tmp:
            bus = BusSystem()
            errors: list[BaseException] = []
            module = MainCogLoop(
                "main_cog", bus, errors.append, ManualClock(), None,
                working_target=MemoryBashTarget(ghost, "read"),
                workflow=REAL_MANIFEST,
                tools_factory=scripted_tools(ScriptedModel(reports=reports)),
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
        reports["workspace"]["focus_ids"] = []
        model = ScriptedModel(reports=reports)

        with TemporaryDirectory() as tmp:
            bus = BusSystem()
            errors: list[BaseException] = []
            module = MainCogLoop(
                "main_cog", bus, errors.append, ManualClock(), None,
                working_target=MemoryBashTarget(ghost, "read"),
                workflow=REAL_MANIFEST,
                tools_factory=scripted_tools(model),
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
            triage = model.messages[model.calls.index("triage")][-1].content.text
            self.assertIn("kind=workflow.question text='Start the task?'", triage)
            feature = ghost.get_knoxel_by_id(ghost.states[-1].stimuli_ids[0])
            self.assertEqual(feature.feature_type, FeatureType.WorldEvent)
            self.assertIn("Start the task?", feature.content)


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

    def _action(self, correlation_id: str, stimulus_id: int):
        return Envelope(
            id=new_id(), source="main_cog", kind="data", topic="action.request",
            correlation_id=correlation_id, target="dialogue_story",
            payload={
                "action_id": "action-0",
                "content": "answer Jesse",
                "stimuli": [{"knoxel_id": stimulus_id, "text": "the kitchen is on fire"}],
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
            bus.put(self._action("corr-story", _store(ghost, "the kitchen is on fire")))
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
            bus.put(self._action("corr-bad", _store(ghost, "the kitchen is on fire")))
            with self.assertRaisesRegex(RuntimeError, "Public reply rejected"):
                module.step(block=False)

            self.assertEqual(errors, [])
            self.assertEqual(replies.events, [])
            self.assertEqual(len(receipts.events), 1)
            self.assertEqual(receipts.events[0].payload["status"], "rejected")


class ModuleToolsTests(unittest.TestCase):
    def test_the_tool_turns_reasoning_off_and_passes_the_run_cancellation(self):
        import threading
        from workflows.inference_settings import REASONING

        seen = {}
        stop = threading.Event()

        def llm_complete(messages, **generation):
            seen.update(generation, roles=[turn.role for turn in messages])
            return "{}"

        factory = build_main_cognition_tools(_small_ghost(), Path("."), llm_complete, cancel_provider=lambda: stop)
        tools = factory(None, None)
        self.assertEqual(sorted(tools), ["llm_complete"])
        self.assertEqual(tools["llm_complete"]([{"role": "user", "content": "hi"}], temperature=0.7), "{}")
        self.assertIs(seen["cancel"], stop)
        self.assertEqual(seen["reasoning_budget"], REASONING["off"])
        self.assertEqual(seen["temperature"], 0.7)


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

            async def activate(self, connection, *, cancel=None, live_test=False):
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

        model = ScriptedModel({"decide": {"actions": [{"action_type": "Reply", "workflow_name": "dialogue_story",
                                                        "content": "reply to Jesse"}]}})

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
                tools_factory=scripted_tools(model),
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
