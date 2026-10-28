"""Resume a real parked kernel after checkpoint reload without rerunning selection."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pm.memory.store import MemoryStore
from pm.records.cycle_state import PersonaConfig
from pm.records.enums import ActionType
from pm.records.global_workspace import GwtTurnRecord
from pm.records.memory_records import ActionSchemaRecord
from pm.runs.supervisor import ManagedRunSupervisor
from pm.runs.workflow_spec import WorkflowSpec
from pm.paths import REPO_ROOT
from tests.unit.pm.runs.test_supervisor import _wait_for


class ParkedRecoveryTests(unittest.TestCase):
    def test_accepted_parked_run_survives_graceful_shutdown_and_answers_after_reload(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"), checkpoint=root / "ghost.pickle")
            batch = store.begin_batch("main-selection", new_tick=True)
            action = batch.create(ActionSchemaRecord(content="Run the question workflow", action_type=ActionType.ToolCall,
                metadata={"action_id": "approved-question", "dispatch_status": "pending"}))
            batch.set_gwt(GwtTurnRecord(tick_id=batch.tick_id, knoxel_ids=[action.id]))
            store.commit(batch)
            workflows = {"question": WorkflowSpec(REPO_ROOT / "workflows/local_task/local_task_question.workflow.md", namespace="question")}
            def supervisor(ghost):
                return ManagedRunSupervisor(ghost, runtime_root=root / "runtime", workflows=workflows, clock=lambda: 0)
            original = supervisor(store)
            receipt = original.start(action_id="approved-question", workflow_name="question")
            run_id = receipt["run_id"]
            self.assertTrue(_wait_for(lambda: original.status(run_id)["state"] == "waiting_for_input"))
            original.shutdown(preserve_parked=True)
            parked_bytes = (root / "ghost.pickle").read_bytes()
            restored = MemoryStore.from_checkpoint(root / "ghost.pickle")
            resumed = supervisor(restored)
            try:
                waiting = resumed.status(run_id)
                self.assertEqual(waiting["state"], "waiting_for_input")
                resumed.answer(run_id, waiting["question_id"], "the blue artifact")
                self.assertTrue(_wait_for(lambda: resumed.status(run_id)["state"] == "completed"))
                self.assertIn("the blue artifact", (Path(waiting["artifact_path"]) / "scratch/observations.md").read_text(encoding="utf-8"))
                self.assertEqual(restored.get_knoxel_by_id(action.id).metadata["execution_status"], "completed")
                future_control = resumed.status(run_id)
            finally:
                resumed.shutdown()
            # The restored ghost owns its parked kernel state. Later disk work
            # must not replace the earlier branch's pending question.
            (root / "ghost.pickle").write_bytes(parked_bytes)
            earlier = MemoryStore.from_checkpoint(root / "ghost.pickle")
            branch = supervisor(earlier)
            try:
                waiting = branch.status(run_id)
                from pm.host.durable_files import atomic_json
                atomic_json(Path(waiting["artifact_path"]) / "control.json", future_control)
                self.assertEqual(branch.status(run_id)["state"], "waiting_for_input")
                self.assertEqual(waiting["state"], "waiting_for_input")
                branch.answer(run_id, waiting["question_id"], "a different artifact")
                self.assertTrue(_wait_for(lambda: branch.status(run_id)["state"] == "completed"))
                content = (Path(waiting["artifact_path"]) / "scratch/observations.md").read_text(encoding="utf-8")
                self.assertIn("a different artifact", content)
                self.assertNotIn("the blue artifact", content)
            finally:
                branch.shutdown()
            fork = root / "fork"
            fork.mkdir()
            (fork / "ghost.pickle").write_bytes(parked_bytes)
            forked = MemoryStore.from_checkpoint(fork / "ghost.pickle")
            isolated = ManagedRunSupervisor(forked, runtime_root=fork / "runtime", workflows=workflows, clock=lambda: 0)
            try:
                waiting = isolated.status(run_id)
                self.assertTrue(Path(waiting["artifact_path"]).is_relative_to(fork))
                isolated.answer(run_id, waiting["question_id"], "an isolated artifact")
                self.assertTrue(_wait_for(lambda: isolated.status(run_id)["state"] == "completed"))
                self.assertIn("an isolated artifact", (Path(waiting["artifact_path"]) / "scratch/observations.md").read_text(encoding="utf-8"))
            finally:
                isolated.shutdown()
