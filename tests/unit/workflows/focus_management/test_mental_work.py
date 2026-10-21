"""Deliberate mental-work role boundary tests.

No pm-coder process and no model endpoint. A fake driver returns one plan,
so the test exercises only the host's inspection and validation boundary.
"""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import threading
import unittest

import yaml

from pm.bus.bus import BusSystem
from pm.bus.envelopes import new_envelope
from pm.memory.agent_tools import MemoryReadSession, MemoryToolConfig, MemoryView
from pm.memory.bash_view import MemoryBashEnvironment, MemoryBashTarget
from pm.records.enums import FeatureType, InterlocusType, SourceKind
from pm.records.memory_records import new_feature
from pm_workflows.runtime import AgentModel, WorkflowRuntime

from tests.support import ManualClock, Recorder
from tests.unit.test_design_contracts import small_ghost
from workflows.focus_management import steps
from workflows.focus_management.host import FocusManagement


class _FakeDriver:
    """Stand in for PmCoderDriver. It performs no inspection of its own."""

    plan: dict = {}
    calls = 0

    def __init__(self, **kwargs):
        pass

    def run_session(self, **kwargs):
        type(self).calls += 1
        return SimpleNamespace(result_json=dict(self.plan), session_ref="sess-1")


class _TimeoutDriver(_FakeDriver):
    def run_session(self, **kwargs):
        raise TimeoutError("The bound agent exceeded its session deadline.")


def _context(tmp, state):
    environment = MemoryBashEnvironment({}, {}, "read")
    environment.write_text("inbox/request.yaml",
                           yaml.safe_dump({"instruction": "recall a hobby and propose focus"}))
    ghost = small_ghost()
    goal = new_feature(ghost, "finish the story", FeatureType.Dialogue, source=ghost.get_user_name())
    ids = frozenset(ghost.all_knoxels)
    records = ghost.snapshot().serialize_knoxels()
    scopes = frozenset((raw["data"]["scope"]["reality_id"], raw["data"]["scope"]["frame"])
                       for raw in records.values())
    memory = MemoryReadSession(MemoryToolConfig(ghost, MemoryView(ids, scopes),
                                                cancel=threading.Event()))
    model = AgentModel(base_url="http://localhost", model="stub", api_key="none",
                       context_window=1000)

    def mental_goals(offset: int = 0, limit: int = 16) -> dict:
        """Read one page of goals and mark them inspected, as the host does."""
        state["calls"] += 1
        state["inspected"] = sorted(set(state["inspected"]) | {goal.id})
        return {"goals": [{"id": goal.id, "text": str(goal.content)}],
                "next_offset": None, "inspected_ids": list(state["inspected"])}

    tools = {
        "mental_memory": lambda: memory,
        "mental_goals": mental_goals,
        "mental_inspected": lambda: list(state["inspected"]),
    }
    manifest = Path("workflows/focus_management/focus_management.workflow.md").read_text(encoding="utf-8")
    granted = yaml.safe_load(manifest.split("---", 2)[1])["roles"]["work"]["tools"]
    tools = {name: tools[name] for name in granted}
    runtime = WorkflowRuntime(
        environment=environment, stop_event=threading.Event(), tools=tools,
        propagate_python_errors=True, agent_model=None,
        classification_session=lambda *args: nullcontext(model),
    )
    return SimpleNamespace(runtime=runtime, kernel_data=Path(tmp), task_id="mental-1",
                           workspace=Path(tmp)), goal


def _plan(evidence, **overrides):
    value = {"instruction": "recall a hobby", "disposition": "propose",
             "summary": "I remembered the story draft.", "evidence_ids": list(evidence),
             "goal_changes": [], "focus": None}
    value.update(overrides)
    return value


class MentalWorkBoundaryTests(unittest.TestCase):
    def setUp(self):
        self._real_driver = steps.PmCoderDriver
        steps.PmCoderDriver = _FakeDriver
        _FakeDriver.calls = 0

    def tearDown(self):
        steps.PmCoderDriver = self._real_driver

    def test_validation_does_not_grant_an_implicit_goal_inspection(self):
        state = {"calls": 0, "inspected": []}
        with TemporaryDirectory() as tmp:
            context, goal = _context(tmp, state)
            _FakeDriver.plan = _plan([goal.id])
            with self.assertRaisesRegex(ValueError, "did not inspect"):
                steps.run(context)
        # The old boundary called mental_goals() during validation, which added
        # the first page to the inspected set. The repaired boundary never does.
        self.assertEqual(state["calls"], 0)
        self.assertEqual(state["inspected"], [])

    def test_cited_inspected_goal_is_accepted(self):
        state = {"calls": 0, "inspected": []}
        with TemporaryDirectory() as tmp:
            context, goal = _context(tmp, state)
            state["inspected"] = [goal.id]  # the agent read it through the tool
            _FakeDriver.plan = _plan([goal.id])
            result = steps.run(context)
            self.assertEqual(result["status"], "done")
            report = yaml.safe_load(context.runtime.environment.read_text("outbox/report.yaml"))
            self.assertEqual(report["source_revisions"], {})
        self.assertEqual(state["calls"], 0)

    def test_timeout_retains_a_failed_report_without_a_plan(self):
        steps.PmCoderDriver = _TimeoutDriver
        state = {"calls": 0, "inspected": []}
        with TemporaryDirectory() as tmp:
            context, _ = _context(tmp, state)
            self.assertEqual(steps.run(context)["status"], "done")
            report = yaml.safe_load(context.runtime.environment.read_text("outbox/report.yaml"))
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["error_type"], "TimeoutError")
            self.assertNotIn("plan", report)


class ProposalCommitTests(unittest.TestCase):
    """Exercise the host commit boundary without a kernel or a real session."""

    def _module(self, ghost, tmp):
        bus = BusSystem()
        errors: list[BaseException] = []
        module = FocusManagement(
            "focus_management", bus, errors.append, ManualClock(), None,
            working_target=MemoryBashTarget(ghost, "read"),
            workflow=Path("workflows/focus_management/focus_management.workflow.md").resolve(),
            runtime_root=Path(tmp), ghost=ghost,
        )
        return module, bus, errors

    def _action(self, ghost):
        batch = ghost.begin_batch("action-seed")
        action = new_feature(ghost, "recall a hobby", FeatureType.StateUpdate,
            source=ghost.get_companion_name(), source_kind=SourceKind.Self,
            interlocus=InterlocusType.PrivateInternal, changes=batch,
            metadata={"operation": "action_selection"})
        ghost.commit(batch)
        return action

    def test_prepare_input_uses_the_selected_instruction_argument(self):
        ghost = small_ghost()
        action = self._action(ghost)
        with TemporaryDirectory() as tmp:
            module, _, _ = self._module(ghost, tmp)
            request = module.prepare_input([new_envelope("main_cog", "data", "action.request", "corr",
                {"knoxel_id": action.id, "content": "Do private mental work.",
                 "arguments": {"instruction": "Recall the last playlist conversation."}})])
            self.assertEqual(request["instruction"], "Recall the last playlist conversation.")

    def test_publish_output_commits_a_private_proposal(self):
        ghost = small_ghost()
        action = self._action(ghost)
        with TemporaryDirectory() as tmp:
            module, bus, errors = self._module(ghost, tmp)
            created = Recorder(bus, "thought.created")
            module.invocation_id = "inv-1"
            module._action_id = action.id
            module._correlation = "corr-1"
            module._goal_revisions = {}
            module.last_environment = MemoryBashTarget(ghost, "read").open("inv-1")
            module.publish_output({"plan": _plan([], focus={"instruction": "keep the story in mind",
                                                            "promote_ids": [], "reduce_ids": [],
                                                            "strength": 0.2, "seconds": 300,
                                                            "topic_hints": [], "reason": "it matters"}),
                                   "source_revisions": {}, "agent_session": "sess-1"}, [])
            self.assertEqual(len(created.events), 1)
            proposal = ghost.get_knoxel_by_id(created.events[0].payload["knoxel_id"])
            self.assertEqual(proposal.metadata["mental_plan"]["summary"],
                             "I remembered the story draft.")
            self.assertTrue(proposal.lineage is not None)
            self.assertEqual(errors, [])

    def test_stale_source_rejects_the_proposal(self):
        ghost = small_ghost()
        action = self._action(ghost)
        recalled = new_feature(ghost, "I like the story draft", FeatureType.Dialogue,
                               source=ghost.get_user_name())
        with TemporaryDirectory() as tmp:
            module, bus, errors = self._module(ghost, tmp)
            created = Recorder(bus, "thought.created")
            rejected = Recorder(bus, "mental.rejected")
            module.invocation_id = "inv-2"
            module._action_id = action.id
            module._correlation = "corr-2"
            module._goal_revisions = {}
            module.last_environment = MemoryBashTarget(ghost, "read").open("inv-2")
            module.publish_output({"plan": _plan([]), "source_revisions": {str(recalled.id): 999},
                                   "agent_session": "sess-1"}, [])
            self.assertEqual(len(created.events), 1)
            self.assertEqual(len(rejected.events), 1)
            outcome = ghost.get_knoxel_by_id(created.events[0].payload["knoxel_id"])
            self.assertEqual(outcome.metadata["mental_status"], "failed")
            self.assertNotIn("mental_plan", outcome.metadata)
            self.assertEqual(ghost.get_knoxel_by_id(action.id).metadata["mental_status"], "failed")
            self.assertEqual(errors, [])

    def test_restart_replays_the_result_without_repeating_agent_work(self):
        ghost = small_ghost()
        action = self._action(ghost)
        with TemporaryDirectory() as tmp:
            ghost.checkpoint = Path(tmp) / "ghost.json"
            module, bus, _ = self._module(ghost, tmp)
            module.invocation_id = "recover-mental"
            module._action_id = action.id
            module._correlation = "corr-recover"
            module._goal_revisions = {}
            module.last_environment = MemoryBashTarget(ghost, "read").open("recover-mental")
            module.publish_output({"plan": _plan([]), "source_revisions": {}, "agent_session": "sess-1"}, [])
            reopened = type(ghost).from_checkpoint(ghost.checkpoint)
            recovered, recovered_bus, errors = self._module(reopened, tmp)
            created = Recorder(recovered_bus, "thought.created")
            recovered.resume()
            recovered.resume()
            self.assertEqual(len(created.events), 1)
            result_id = created.events[0].payload["knoxel_id"]
            self.assertEqual(result_id, reopened.get_knoxel_by_id(action.id).metadata["mental_result_id"])
            from tests.unit.workflows.inner_thought.test_episode import _acknowledge
            _acknowledge(reopened, result_id)
            recovered.release_results(created.events)
            recovered.resume()
            self.assertEqual(len(created.events), 1)
            self.assertEqual(errors, [])

    def test_immediate_main_discard_releases_publication_for_retry(self):
        ghost = small_ghost()
        action = self._action(ghost)
        with TemporaryDirectory() as tmp:
            module, bus, _ = self._module(ghost, tmp)
            module.invocation_id = "discarded-mental"
            module._action_id = action.id
            module._correlation = "corr-discard"
            module._goal_revisions = {}
            module.last_environment = MemoryBashTarget(ghost, "read").open("discarded-mental")
            created = Recorder(bus, "thought.created")
            bus.subscribe("discarding-main", lambda event: event.topic == "thought.created",
                          lambda event: module.release_results([event]))
            module.publish_output({"plan": _plan([]), "source_revisions": {}, "agent_session": "sess-1"}, [])
            module.resume()
            self.assertEqual(len(created.events), 2)
            self.assertEqual(created.events[0].payload["knoxel_id"], created.events[1].payload["knoxel_id"])


if __name__ == "__main__":
    unittest.main()
