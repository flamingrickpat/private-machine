"""Shared-behavior checks for the bus, batch, timer, and termination.

These use real modules, a real bus, and a real Ghost, but step modules directly
so no thread is ever started.
"""
from __future__ import annotations

import unittest
from pathlib import Path

import ghost_mcp
from pm.bus.bus import BusSystem
from pm.bus.envelopes import Envelope, _new_id
from runtime.shell import ShellModule
from working_target import GhostBashWrapper

from tests.support import (
    ManualClock,
    Recorder,
    fresh_ghost,
    load_seed_state,
    make_app,
    store_feature,
)
import tempfile
import threading
import yaml
from pm_workflows.kernel import Kernel
from pm_workflows.runtime import WorkflowRuntime
from pm_workflows.working_target import CommandResult


class _EchoModule(ShellModule):
    """Handler records each batch size and re-publishes one message per item."""

    DATA_TOPICS = frozenset({"ping"})

    def __init__(self, bus):
        super().__init__("echo", bus, lambda exc: None, ManualClock(), None)
        self.batch_sizes: list[int] = []

    def handle_batch(self, batch: list[Envelope]) -> None:
        self.batch_sizes.append(len(batch))
        for _envelope in batch:
            self.bus.put(Envelope(
                id=_new_id(), source=self.module_id, kind="data",
                topic="ping", correlation_id="", payload={},
            ))


def _ping(bus):
    bus.put(Envelope(
        id=_new_id(), source="x", kind="data", topic="ping",
        correlation_id="", payload={},
    ))


def _terminate(bus, module):
    bus.put(Envelope(
        id=_new_id(), source="app", kind="control", topic="terminate",
        correlation_id="", payload={}, target=module.module_id,
    ))


class ShellBusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seed_dict = load_seed_state()

    def test_construction_starts_no_threads(self):
        app = make_app(self.seed_dict)
        for module in app.modules:
            self.assertIsNone(module._thread)

    def test_idle_step_returns_without_waiting(self):
        bus = BusSystem()
        module = _EchoModule(bus)
        self.assertFalse(module.step(block=False))

    def test_fanout(self):
        bus = BusSystem()
        recorder_a = Recorder(bus, topic="data.topic")
        recorder_b = Recorder(bus, topic="data.topic")
        bus.put(Envelope(
            id=_new_id(), source="x", kind="data", topic="data.topic",
            correlation_id="", payload={},
        ))
        self.assertEqual(len(recorder_a.events), 1)
        self.assertEqual(len(recorder_b.events), 1)
        self.assertEqual(recorder_a.events[0].id, recorder_b.events[0].id)

    def test_batch_boundary(self):
        bus = BusSystem()
        module = _EchoModule(bus)
        bus.subscribe(module.module_id, module.accepts, module.deliver)
        _ping(bus)
        self.assertTrue(module.step(block=False))
        # The re-published message was not part of the first batch.
        self.assertEqual(module.batch_sizes, [1])
        self.assertTrue(module.step(block=False))
        self.assertEqual(module.batch_sizes, [1, 1])

    def test_input_after_termination(self):
        bus = BusSystem()
        module = _EchoModule(bus)
        bus.subscribe(module.module_id, module.accepts, module.deliver)
        terminations = Recorder(bus, topic="module.terminated")
        _terminate(bus, module)
        self.assertTrue(module.stop_event.is_set())
        self.assertFalse(module.step(block=False))
        self.assertEqual(len(terminations.events), 1)
        self.assertEqual(module.batch_sizes, [])

    def test_repeated_termination(self):
        bus = BusSystem()
        module = _EchoModule(bus)
        bus.subscribe(module.module_id, module.accepts, module.deliver)
        terminations = Recorder(bus, topic="module.terminated")
        _terminate(bus, module)
        _terminate(bus, module)
        module.step(block=False)
        module.step(block=False)
        self.assertEqual(len(terminations.events), 1)

    def test_scheduler_termination(self):
        app = make_app(self.seed_dict)
        app.tasks.create("task-x", "dummy", "corr")
        app.scheduler.on_terminate()
        self.assertEqual(app.tasks.get("task-x")["status"], "terminated")

    def test_fresh_environment(self):
        ghost = fresh_ghost(self.seed_dict)
        wrapper = GhostBashWrapper(ghost, "read")
        before = set(ghost.all_knoxels)
        new_id = store_feature(ghost, "fresh feature")
        self.assertNotIn(new_id, before)
        environment = wrapper.open("inv-1")
        self.assertIn(new_id, environment.visible_knoxels)

    def test_read_scope(self):
        ghost = fresh_ghost(self.seed_dict)
        all_ids = set(ghost.all_knoxels)
        omitted = max(all_ids)
        kept = [kid for kid in sorted(all_ids) if kid != omitted]
        target = GhostBashWrapper(ghost, "read", visible_ids=kept)
        environment = target.open("inv-1")

        result = environment.exec(f"cat /work/ghost/knoxels/{omitted}.yaml")
        self.assertNotEqual(result.exit_code, 0)

        tools = ghost_mcp.build_tools(environment)
        with self.assertRaises(KeyError):
            tools["ghost_get"](omitted)

        self.assertEqual(tools["ghost_get"](kept[0])["data"]["id"], kept[0])


class _MemoryEnvironment:
    def __init__(self):
        self.files = {}

    def read_text(self, path):
        return self.files[path]

    def write_text(self, path, text):
        self.files[path] = text

    def move(self, source, destination):
        self.files[destination] = self.files.pop(source)

    def list_files(self, directory):
        return sorted(k for k in self.files if k.startswith(directory))

    def exec(self, command):
        return CommandResult("", "", 0)


class _MemoryTarget:
    def open(self, invocation_id):
        return _MemoryEnvironment()


_MANIFEST = """\
---
name: probe
driver: {kind: python}
human_resolver: {mode: forbid}
state_policy: {mutation_model: append_only}
roles:
  work:
    skill: role.py
    result_contract:
      type: json
      schema: {status: {enum: [done]}, summary: string}
phases:
  - name: work
    kind: role
    role: work
    on_status: {done: stop}
    on_invalid: {action: stop}
---
"""


class KernelIntegrationTests(unittest.TestCase):
    def _run_role(self, role_code: str, *, propagate: bool) -> dict:
        tmp = Path(tempfile.mkdtemp(prefix="pmprobe-"))
        (tmp / "role.py").write_text(role_code, encoding="utf-8")
        (tmp / "w.workflow.md").write_text(_MANIFEST, encoding="utf-8")
        runtime = WorkflowRuntime(
            environment=_MemoryEnvironment(),
            stop_event=threading.Event(),
            tools={},
            propagate_python_errors=propagate,
        )
        kernel = Kernel(
            manifest_path=tmp / "w.workflow.md",
            workspace=tmp / "repo",
            task_id="probe",
            task_text="probe",
            base_dir=tmp,
            kernel_data_root=tmp / "kernel_data",
            runtime=runtime,
        )
        return kernel.run()

    def test_workflow_cancellation_inside_role(self):
        result = self._run_role(
            "def run(context):\n"
            "    context.runtime.stop_event.set()\n"
            "    context.runtime.check_cancelled()\n"
            "    return {'status': 'done', 'summary': 'x'}\n",
            propagate=True,
        )
        self.assertEqual(result["terminal_status"], "terminated")

    def test_python_exception_propagates(self):
        with self.assertRaises(ValueError):
            self._run_role(
                "def run(context):\n    raise ValueError('boom')\n",
                propagate=True,
            )


if __name__ == "__main__":
    unittest.main()