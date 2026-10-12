"""Regression checks using the real MemoryStore, bus, kernel, workers, and MCP client.

The in-process MCP publishes controlled tool outcomes. These checks establish
adapter behavior, not live Minecraft or model quality.
"""
import json
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml
from fastmcp import FastMCP

from pm.bus.envelopes import Envelope
from main import build_app
from pm.runs.supervisor import ManagedRunError, ManagedRunSupervisor, WorkflowSpec
from runtime.minecraft_tools import build_minecraft_tools
from runtime.body_lease import BODY_LEASES
from tests.live.conversation import SCENARIOS, evaluation_context, generation_complete
from tests.support import ManualClock
from tests.test_design_contracts import small_ghost
from tests.test_minecraft_workflow import observation
from pm.host.workflow_module import PROJECT_ROOT


def wait_for(predicate):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.01)
    raise AssertionError("The worker did not reach its expected boundary.")


def readable_report(path):
    """Wait for readable publication, including transient Windows file locks.

    File existence alone cannot establish that a concurrent rename released
    its handle. Invalid YAML still raises, and the caller's deadline bounds
    the wait for an absent or inaccessible report.
    """
    try:
        text = path.read_text()
    except (FileNotFoundError, PermissionError):
        return None
    return yaml.safe_load(text)


class ReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ghost = small_ghost()
        self.supervisors = []

    def tearDown(self):
        for supervisor in self.supervisors:
            for run_id in supervisor.receipts():
                supervisor.terminate(run_id)
            supervisor.join_all()
        self.tmp.cleanup()

    def supervisor(self, **workflows):
        supervisor = ManagedRunSupervisor(
            self.ghost, runtime_root=self.root, workflows=workflows, clock=time.monotonic,
        )
        self.supervisors.append(supervisor)
        return supervisor

    def local(self):
        return self.supervisor(local_task=WorkflowSpec(
            PROJECT_ROOT / "workflows/local_task/local_task.workflow.md",
        ))

    def minecraft(self, *, ok=True, planner=None):
        server = FastMCP("minecraft-adapter-contract")

        @server.tool
        def minecraft_observe(include_image: bool = False) -> dict:
            return {"ok": True, **observation()["result"]}

        @server.tool
        def minecraft_mine_block(position: dict) -> dict:
            """Mine the block at the supplied position."""
            return {"ok": ok, "reason": None if ok else "unreachable", "position": position}

        tools = build_minecraft_tools(server, plan_model=planner)
        def factory(run_id, environment):
            return {**tools, "minecraft_endpoint": lambda: "fixture://reconciliation",
                    "minecraft_lease": lambda: BODY_LEASES.acquire("fixture://reconciliation", run_id)}
        return self.supervisor(minecraft=WorkflowSpec(
            PROJECT_ROOT / "workflows/minecraft/minecraft.workflow.md",
            tools_factory=factory,
        ))

    def test_real_mode_has_no_periodic_dummy_thought(self):
        clock = ManualClock()
        app = build_app(None, self.root, ghost=self.ghost, cognition="real", clock=clock)
        before = self.ghost.state_dict()
        clock.advance(300)
        self.assertFalse(app.inner_thought.step())
        self.assertEqual(before, self.ghost.state_dict())

    def test_managed_target_reaches_management_through_bus(self):
        app = build_app(None, self.root, ghost=self.ghost, cognition="real")
        self.supervisors.append(app.supervisor)
        app.bus.put(Envelope("req", "test", "data", "action.request", "corr", {
            "action_id": "managed-direct", "content": "write the bus note", "arguments": {},
        }, target="local_task"))
        self.assertTrue(app.self_management.step())
        app.supervisor.join_all()
        receipt = next(iter(app.supervisor.receipts().values()))
        self.assertEqual(receipt["state"], "completed")
        report = yaml.safe_load((Path(receipt["artifact_path"]) / "outbox/report.yaml").read_text())
        self.assertEqual(report["instruction"], "write the bus note")

    def test_update_changes_the_worker_output_and_applied_revision(self):
        supervisor = self.local()
        run = supervisor.start(action_id="update", workflow_name="local_task", instructions="old",
                               arguments={"wait_for_update": True, "instruction": "cannot override identity"})
        wait_for(lambda: (Path(run["artifact_path"]) / "scratch/started.md").exists())
        queued = supervisor.update(run["run_id"], "new instruction", request_id="revision-1")
        self.assertEqual(queued["revision"], 1)
        self.assertEqual(queued["applied_revision"], 0)
        supervisor.join_all()
        status = supervisor.status(run["run_id"])
        self.assertEqual(status["applied_revision"], 1)
        self.assertEqual(status["state"], "completed")
        report = yaml.safe_load((Path(run["artifact_path"]) / "outbox/report.yaml").read_text())
        self.assertEqual(report["instruction"], "new instruction")

    def test_duplicate_concurrent_starts_launch_one_run(self):
        supervisor = self.local()
        with ThreadPoolExecutor(max_workers=2) as pool:
            receipts = list(pool.map(lambda _: supervisor.start(
                action_id="same", workflow_name="local_task", instructions="once",
            ), range(2)))
        supervisor.join_all()
        self.assertEqual(receipts[0]["run_id"], receipts[1]["run_id"])
        self.assertEqual(len(supervisor.receipts()), 1)

    def test_domain_failure_is_failed_even_when_mcp_transport_succeeds(self):
        supervisor = self.minecraft(ok=False)
        run = supervisor.start(action_id="failure", workflow_name="minecraft", arguments={
            "action_tool": "minecraft_mine_block", "action_arguments": {"position": {"x": 1, "y": 2, "z": 3}},
        })
        supervisor.join_all()
        self.assertEqual(supervisor.status(run["run_id"])["state"], "failed")
        report = yaml.safe_load((Path(run["artifact_path"]) / "outbox/report.yaml").read_text())
        self.assertFalse(report["accepted"])
        self.assertEqual(report["world"]["episodes"][0]["outcome"], "failed")

    def test_plain_instruction_uses_schemas_and_plans_an_action(self):
        prompts = []

        def planner(messages):
            prompts.extend(messages)
            return 'minecraft_mine_block {"position":{"x":5,"y":1,"z":3}}'

        supervisor = self.minecraft(planner=planner)
        run = supervisor.start(action_id="plan", workflow_name="minecraft", instructions="mine one nearby log",
                               arguments={"mode": "boundary_check"})
        supervisor.join_all()
        self.assertEqual(supervisor.status(run["run_id"])["state"], "completed")
        self.assertIn("mine one nearby log", prompts[-1]["content"])
        self.assertIn('"parameters"', prompts[-1]["content"])
        self.assertIn('"position"', prompts[-1]["content"])
        report = yaml.safe_load((Path(run["artifact_path"]) / "outbox/report.yaml").read_text())
        self.assertEqual(report["action"]["tool"], "minecraft_mine_block")
        self.assertEqual(len(report["observations"]), 3)

    def test_minecraft_update_replans_and_final_boundary_rejects_late_update(self):
        prompts = []

        def planner(messages):
            prompts.extend(messages)
            return 'minecraft_observe {"include_image":false}'

        supervisor = self.minecraft(planner=planner)
        run = supervisor.start(action_id="replan", workflow_name="minecraft", instructions="old goal", arguments={
            "wait_for_update": True, "wait_for_cancel": True,
            "action_tool": "minecraft_mine_block", "action_arguments": {"position": {"x": 0}},
        })
        root = Path(run["artifact_path"])
        wait_for(lambda: (root / "outbox/world.yaml").exists())
        supervisor.update(run["run_id"], "only observe the world", request_id="new-goal")
        report = wait_for(lambda: readable_report(root / "outbox/report.yaml"))
        self.assertEqual(report["goal"], "only observe the world")
        self.assertEqual(report["action"]["tool"], "minecraft_observe")
        self.assertIn("only observe the world", prompts[-1]["content"])
        self.assertEqual(supervisor.status(run["run_id"])["applied_revision"], 1)
        with self.assertRaises(ManagedRunError):
            supervisor.update(run["run_id"], "too late")
        supervisor.terminate(run["run_id"])
        supervisor.join_all()
        self.assertEqual(supervisor.status(run["run_id"])["state"], "cancelled")
        self.assertTrue((root / "outbox/world.yaml").exists())


class ConversationEvidenceTests(unittest.TestCase):
    def summary(self):
        return {"execution_path": "application", "scenarios": [s.name for s in SCENARIOS],
                "exchanges_per_scenario": 20, "completed_exchanges": 200,
                "hard_violations": [], "infrastructure_errors": []}

    def test_incomplete_or_assistant_only_generation_cannot_pass(self):
        self.assertTrue(generation_complete(self.summary()))
        for override in ({"completed_exchanges": 192}, {"execution_path": "assistant"},
                         {"infrastructure_errors": ["HTTP 400"]}, {"scenarios": ["casual_banter"]}):
            self.assertFalse(generation_complete({**self.summary(), **override}))

    def test_reply_context_preserves_pairing_and_tool_evidence(self):
        exchanges = [
            {"index": 0, "jesse": "first user", "companion": "first reply"},
            {"index": 1, "error": "backend failed"},
            {"index": 2, "jesse": "current user", "companion": "current reply",
             "tool_receipts": [{"name": "managed_status", "result": {"state": "failed"}}]},
        ]
        context = evaluation_context(exchanges)
        self.assertIn("Jesse: current user\nSpamuel: current reply", context)
        self.assertIn('"state": "failed"', context)
        self.assertIn("backend failed", context)
