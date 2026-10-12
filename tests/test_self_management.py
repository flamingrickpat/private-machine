"""SelfManagement direct workflow and module tests.

Real MemoryStore, real supervisor, real kernel. The local worker is a real local
workflow. No model endpoint or module thread is required.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pm_workflows.runtime import WorkflowRuntime

from pm.bus.bus import BusSystem
from pm.bus.envelopes import Envelope
from pm.memory.store_tools import build_tools
from main import build_app
from modules import SelfManagement
from pm.memory.store import MemoryStore
from pm.records.cycle_state import PersonaConfig
from pm.runs.supervisor import ManagedRunSupervisor, WorkflowSpec
from pm.runs.management_api import build_management_tools
from pm.runs.workflow_runner import PROJECT_ROOT, WorkflowRequest, WorkflowRunner, WorkflowStatus
from tests.support import Recorder
from workflows.self_management.schemas import SelfManagementReport
from pm.memory.bash_view import MemoryBashEnvironment

MANIFEST = PROJECT_ROOT / "workflows" / "self_management" / "self_management.workflow.md"
LOCAL_TASK = PROJECT_ROOT / "workflows" / "local_task"


def _wait_for(predicate, timeout: float = 15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class SelfManagementTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.ghost = MemoryStore(
            PersonaConfig(user_name="Jesse", companion_name="Spamuel"),
            checkpoint=self.root / "ghost.json",
        )
        self.supervisor = ManagedRunSupervisor(
            self.ghost,
            runtime_root=self.root / "runtime",
            workflows={
                "local_task": WorkflowSpec(
                    LOCAL_TASK / "local_task.workflow.md", namespace="local_task",
                ),
            },
            clock=time.monotonic,
            task_root=self.root / "tasks",
        )

    def tearDown(self):
        self.supervisor.join_all(timeout=10)
        self._tmp.cleanup()

    def _run(self, request: dict) -> SelfManagementReport:
        environment = MemoryBashEnvironment({}, {}, "read")
        runtime = WorkflowRuntime(
            environment=environment,
            stop_event=threading.Event(),
            tools=build_management_tools(self.supervisor),
            propagate_python_errors=True,
        )
        runner = WorkflowRunner(MANIFEST, self.root / "runtime", namespace="self_management")
        result = runner.invoke(WorkflowRequest("self-management-test", request), runtime)
        self.assertEqual(result.status, WorkflowStatus.Completed)
        return SelfManagementReport.model_validate(result.report)

    def test_start_operation_launches_a_real_run(self):
        report = self._run({
            "operation": "start", "action_id": "action-0",
            "workflow_name": "local_task", "instructions": "note",
        })
        self.assertTrue(report.accepted)
        run_id = report.receipt["run_id"]
        self.assertTrue(_wait_for(lambda: self.supervisor.status(run_id)["state"] == "completed"))

    def test_duplicate_start_does_not_launch_twice(self):
        first = self._run({
            "operation": "start", "action_id": "action-1",
            "workflow_name": "local_task", "instructions": "one",
        })
        second = self._run({
            "operation": "start", "action_id": "action-1",
            "workflow_name": "local_task", "instructions": "two",
        })
        self.assertEqual(first.receipt["run_id"], second.receipt["run_id"])
        self.assertEqual(len(self.ghost.managed_runs), 1)

    def test_status_operation_reads_the_durable_table(self):
        started = self._run({
            "operation": "start", "action_id": "action-2",
            "workflow_name": "local_task", "instructions": "note",
        })
        run_id = started.receipt["run_id"]
        self.assertTrue(_wait_for(lambda: self.supervisor.status(run_id)["state"] == "completed"))
        report = self._run({"operation": "status", "run_id": run_id})
        self.assertTrue(report.accepted)
        self.assertEqual(report.receipt["state"], "completed")

    def test_unavailable_workflow_is_rejected_without_a_second_run(self):
        report = self._run({
            "operation": "start", "action_id": "action-3",
            "workflow_name": "minecraft", "instructions": "dig",
        })
        self.assertFalse(report.accepted)
        self.assertEqual(report.receipt["state"], "failed")
        self.assertEqual(len(self.ghost.managed_runs), 1)

    def test_refused_update_is_a_typed_rejection(self):
        started = self._run({
            "operation": "start", "action_id": "action-4",
            "workflow_name": "local_task", "instructions": "quick",
        })
        run_id = started.receipt["run_id"]
        self.assertTrue(_wait_for(lambda: self.supervisor.status(run_id)["state"] == "completed"))
        report = self._run({"operation": "update", "run_id": run_id, "instructions": "later"})
        self.assertFalse(report.accepted)
        self.assertIn("cannot accept", report.summary)


class SelfManagementModuleTests(unittest.TestCase):
    def test_module_maps_action_arguments_to_a_receipt(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            ghost = MemoryStore(
                PersonaConfig(user_name="Jesse", companion_name="Spamuel"),
                checkpoint=root / "ghost.json",
            )
            supervisor = ManagedRunSupervisor(
                ghost, runtime_root=root / "runtime",
                workflows={
                    "local_task": WorkflowSpec(
                        LOCAL_TASK / "local_task.workflow.md", namespace="local_task",
                    ),
                },
                clock=time.monotonic,
                task_root=root / "tasks",
            )
            app = build_app(None, root / "app-runtime", ghost=ghost, clock=time.monotonic)
            module = SelfManagement(
                "self_management", app.bus, self.fail, time.monotonic, None,
                working_target=app.main_cog.working_target,
                workflow=MANIFEST,
                tools_factory=lambda environment, batch: build_management_tools(supervisor),
                runtime_root=root / "runtime", ghost=ghost,
            )
            receipts = Recorder(app.bus, "action.receipt")
            app.bus.subscribe(module.module_id, module.accepts, module.deliver)
            module.mailbox.put(Envelope(
                "req", "test", "data", "action.request", "corr-1",
                {
                    "action_id": "action-9",
                    "workflow_name": "self_management",
                    "content": "start a note task",
                    "arguments": {
                        "operation": "start", "workflow_name": "local_task",
                        "instructions": "note",
                    },
                },
                target="self_management",
            ))
            self.assertTrue(module.step())
            self.assertEqual(module.last_result.status, WorkflowStatus.Completed)
            self.assertEqual(len(receipts.events), 1)
            self.assertEqual(receipts.events[0].payload["status"], "accepted")
            run_id = receipts.events[0].payload["receipt"]["run_id"]
            self.assertTrue(_wait_for(lambda: supervisor.status(run_id)["state"] == "completed"))
            supervisor.join_all(timeout=10)


class ManagementMcpTests(unittest.TestCase):
    def test_management_operations_are_reachable_over_real_mcp(self):
        from fastmcp import Client
        from pm.runs.management_api import management_mcp
        from pm.host.mcp_listeners import serve_services

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            ghost = MemoryStore(
                PersonaConfig(user_name="Jesse", companion_name="Spamuel"),
                checkpoint=root / "ghost.json",
            )
            supervisor = ManagedRunSupervisor(
                ghost, runtime_root=root / "runtime",
                workflows={
                    "local_task": WorkflowSpec(
                        LOCAL_TASK / "local_task.workflow.md", namespace="local_task",
                    ),
                },
                clock=time.monotonic, task_root=root / "tasks",
            )

            seen: dict = {}

            async def exercise(descriptors):
                async with Client(descriptors["workflow_management"]["url"]) as client:
                    tools = await client.list_tools()
                    names = {tool.name for tool in tools}
                    self.assertIn("managed_start", names)
                    self.assertIn("managed_status", names)
                    result = await client.call_tool("managed_start", {
                        "action_id": "action-mcp", "workflow_name": "local_task",
                        "instructions": "over mcp",
                    })
                    self.assertFalse(result.is_error)
                    seen["run_id"] = result.data["run_id"]
                    status = await client.call_tool("managed_status", {"run_id": seen["run_id"]})
                    self.assertFalse(status.is_error)
                    self.assertEqual(status.data["run_id"], seen["run_id"])

            with serve_services({"workflow_management": management_mcp(supervisor)}) as descriptors:
                import asyncio
                asyncio.run(exercise(descriptors))
            self.assertTrue(
                _wait_for(lambda: supervisor.status(seen["run_id"])["state"] == "completed")
            )
            supervisor.join_all(timeout=10)


if __name__ == "__main__":
    unittest.main()
