"""Run explicit child cancellation through the real cognitive kernels.

No model output or companion behavior is simulated. A stopped child must
not turn an ordinary application shutdown into a stage failure.
"""
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest

from pm.memory.store import MemoryStore
from pm.records.cycle_state import PersonaConfig
from pm_workflows.runtime import WorkflowRuntime
from pm.memory.bash_view import MemoryBashEnvironment
from pm.runs.workflow_runner import PROJECT_ROOT, WorkflowRequest, WorkflowRunner, WorkflowStatus
from workflows.inner_thought.stage_runner import build_inner_stage_runner


class CognitiveCancellationTests(unittest.TestCase):
    def run_cancelled_child(self, factory, stage):
        ghost = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"))
        before = ghost.state_dict()
        cancel = Event()
        cancel.set()

        def no_completion(*_args, **_kwargs):
            raise AssertionError("A stopped child must not call a model.")

        with TemporaryDirectory() as temp:
            root = Path(temp)
            run_stage = factory(ghost, no_completion, root, cancel=cancel)
            with self.assertRaises(asyncio.CancelledError):
                run_stage(stage, {})
            self.assertTrue(list(root.rglob("journal.jsonl")), "The actual child kernel must retain its stop receipt.")
        self.assertEqual(ghost.state_dict(), before)

    def test_main_cycle_preserves_explicit_cancellation(self):
        def no_completion(*_args, **_kwargs):
            raise AssertionError("A stopped cycle must not call a model.")

        cancel = Event()
        cancel.set()
        with TemporaryDirectory() as temp:
            root = Path(temp)
            runtime = WorkflowRuntime(MemoryBashEnvironment({}, {}, "read"), cancel,
                                      {"llm_complete": no_completion}, propagate_python_errors=True)
            runner = WorkflowRunner(PROJECT_ROOT / "workflows/main_cognition/main_cognition.workflow.md", root,
                                    namespace="main_cognition")
            result = runner.invoke(WorkflowRequest("cancelled-cycle", {}), runtime)
            self.assertIs(result.status, WorkflowStatus.Cancelled)
            self.assertIsNone(result.report)
            self.assertTrue(list(root.rglob("journal.jsonl")), "The actual kernel must retain its stop receipt.")

    def test_private_child_preserves_explicit_cancellation(self):
        self.run_cancelled_child(build_inner_stage_runner, "trigger")
