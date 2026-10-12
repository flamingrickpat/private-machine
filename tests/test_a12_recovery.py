"""Exercise real process locks, Ghost receipts, and captured live interruptions."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import threading
import unittest

from pm_workflows.runtime import WorkflowRuntime
from runtime.body_lease import BodyBusy, BodyLeases, ProcessLock, StaleEpoch, atomic_json
from runtime.managed_runs import ManagedRunError, ManagedRunReceipt, ManagedRunSupervisor, WorkflowSpec
from runtime.minecraft_actor import observed_target, read_actor_journal, target_block_cell
from pm_core.ghost.service import Ghost
from tests.test_design_contracts import small_ghost
from workflow_adapter import PROJECT_ROOT
from working_target import FolderEnvironment


class RecoveryTests(unittest.TestCase):
    def test_captured_continuous_targets_use_observed_integer_cells(self):
        from jsonschema.validators import validator_for
        capture = json.loads((PROJECT_ROOT / "tests/fixtures/alpha/a13-reconciliation-world-cells.json").read_text(encoding="utf-8"))
        parameters = capture["tool_schema"]["parameters"]
        validator = validator_for(parameters)(parameters)
        self.assertTrue(capture["rejected"])
        for case in capture["cases"]:
            point = case["dispatch"]["arguments"]["position"]
            original = json.dumps(point, sort_keys=True)
            self.assertTrue(list(validator.iter_errors({"position": point})))
            cell = target_block_cell(point)
            self.assertEqual(cell, case["world_read"]["arguments"]["position"])
            self.assertEqual(cell, case["world_read"]["result"]["position"])
            self.assertEqual(list(validator.iter_errors({"position": cell})), [])
            self.assertTrue(observed_target(case["world_read"], cell))
            self.assertEqual(json.dumps(point, sort_keys=True), original)

    def test_captured_negative_positions_use_body_block_coordinates(self):
        capture = json.loads((PROJECT_ROOT / "tests/fixtures/alpha/a13-reconciliation-world-cells.json").read_text(encoding="utf-8"))
        self.assertTrue(capture["body_cases"])
        for case in capture["body_cases"]:
            camera = case["receipt"]["result"]["camera"]
            self.assertEqual(target_block_cell(camera["feet_position"]), camera["block_position"])

    def test_captured_partial_harvests_count_as_one_physical_tree(self):
        from workflows.minecraft.actor import verify_wood_return
        capture = json.loads((PROJECT_ROOT / "tests/fixtures/alpha/a12-partial-tree.json").read_text())
        verification = verify_wood_return(capture["receipts"], capture["initial"], capture["final"], {"trees": 2})
        self.assertEqual(verification["trees"], 1)
        self.assertFalse(verification["checks"]["five_distinct_trees"])

    def test_captured_air_is_a_known_target_after_harvest(self):
        capture = json.loads((PROJECT_ROOT / "tests/fixtures/alpha/a12-air-inspection.json").read_text())
        receipt = capture["receipt"]
        self.assertFalse(receipt["ok"])
        self.assertTrue(observed_target(receipt, receipt["arguments"]["position"]))
        self.assertFalse(observed_target(receipt, {"x": 0, "y": 0, "z": 0}))
        from workflows.minecraft_tasks.steps import scoped_reads
        evidence = scoped_reads([receipt])[0]
        self.assertTrue(evidence["observation_available"])
        self.assertFalse(evidence["ok"])
        self.assertEqual(evidence["result"]["block_name"], "air")

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_a_different_process_cannot_take_the_same_endpoint_alias(self):
        script = "from pathlib import Path; import sys; from runtime.body_lease import BodyLeases; " \
                 "lease=BodyLeases(Path(sys.argv[1])); " \
                 "scope=lease.acquire('http://localhost:6767/mcp/','child'); " \
                 "token=scope.__enter__(); print(token['epoch'],flush=True); sys.stdin.readline(); scope.__exit__(None,None,None)"
        process = subprocess.Popen([sys.executable, "-c", script, str(self.root)],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            first_epoch = int(process.stdout.readline())
            leases = BodyLeases(self.root)
            with self.assertRaises(BodyBusy):
                with leases.acquire("http://127.0.0.1:6767/mcp", "parent"):
                    self.fail("The second process obtained the body.")
            process.stdin.write("finish\n")
            process.stdin.flush()
            process.wait(10)
            self.assertEqual(process.returncode, 0)
            with leases.acquire("http://127.0.0.1:6767/mcp", "parent") as token:
                self.assertGreater(token["epoch"], first_epoch)
        finally:
            if process.poll() is None:
                from dev.work_items.alpha_a12_recovery_gate import kill_worker
                kill_worker(process)
            for stream in (process.stdin, process.stdout, process.stderr):
                stream.close()

    def test_same_owner_name_cannot_reuse_an_earlier_token(self):
        leases = BodyLeases(self.root)
        with leases.acquire("avatar", "same") as old:
            pass
        with leases.acquire("avatar", "same"):
            with self.assertRaises(StaleEpoch):
                leases.validate("avatar", old)

    def test_two_resume_attempts_cannot_pass_the_live_worker_lock(self):
        ghost = small_ghost()
        supervisor = ManagedRunSupervisor(ghost, runtime_root=self.root / "runtime", clock=lambda: 1,
            workflows={"minecraft": WorkflowSpec(PROJECT_ROOT / "workflows/minecraft/minecraft.workflow.md")})
        receipt = ManagedRunReceipt("body-run", "accepted-action", "minecraft", "exchange", "interrupted",
                                    "process exited", str(self.root / "task"))
        supervisor._persist(receipt)
        with ProcessLock(self.root / "task/worker.lock"):
            for request in ("first", "second"):
                with self.assertRaises(ManagedRunError):
                    supervisor.resume("body-run", request_id=request)
        self.assertEqual(supervisor.status("body-run")["execution_epoch"], 1)
        self.assertEqual(supervisor.status("body-run")["state"], "interrupted")

    def test_superseded_revision_cannot_obtain_a_dispatch_marker(self):
        ghost = small_ghost()
        supervisor = ManagedRunSupervisor(ghost, runtime_root=self.root, clock=lambda: 1, workflows={})
        receipt = ManagedRunReceipt("run", "action", "minecraft", "exchange", "running", "active",
                                    str(self.root / "task"), instructions="first", applied_revision=0)
        supervisor._persist(receipt)
        supervisor.update("run", "second", request_id="update")
        path = self.root / "dispatch.json"
        self.assertFalse(supervisor.authorize_dispatch("run", 1, 0, lambda: atomic_json(path, {"sent": True})))
        self.assertFalse(path.exists())
        self.assertTrue(supervisor.authorize_dispatch("run", 1, 1, lambda: atomic_json(path, {"authorized": True})))

    def test_checkpoint_commit_survives_a_crash_before_its_control_mirror(self):
        ghost = small_ghost()
        ghost.checkpoint = self.root / "ghost.json"
        supervisor = ManagedRunSupervisor(ghost, runtime_root=self.root, clock=lambda: 1, workflows={})
        original = ManagedRunReceipt("run", "action", "minecraft", "exchange", "running", "active",
                                     str(self.root / "task"), instructions="original", applied_revision=0)
        supervisor._persist(original)
        # Populate the survivor's cache before the other process commits.
        supervisor.get("run")
        other = Ghost.from_checkpoint(ghost.checkpoint)
        accepted = dict(other.managed_runs["run"])
        accepted.update(state="running", revision=1, instructions="accepted replacement",
                        control_version=accepted["control_version"] + 1)
        changes = other.begin_batch("accepted-before-mirror-crash")
        changes.set_run_receipt("run", accepted)
        other.commit(changes)
        # The old file deliberately remains. Only the actual Ghost commit
        # survives this injected boundary; no workflow response is substituted.
        marker = self.root / "dispatch.json"
        self.assertFalse(supervisor.authorize_dispatch("run", 1, 0, lambda: atomic_json(marker, {"sent": True})))
        self.assertFalse(marker.exists())
        self.assertEqual(supervisor.get("run")["instructions"], "accepted replacement")
        with self.assertRaises(ManagedRunError):
            supervisor._persist(original)

    def test_pending_progress_replays_its_identity_without_starting_a_worker(self):
        ghost = small_ghost()
        ghost.checkpoint = self.root / "ghost.json"
        supervisor = ManagedRunSupervisor(ghost, runtime_root=self.root, clock=lambda: 1, workflows={})
        receipt = ManagedRunReceipt("run", "action", "minecraft", "exchange", "interrupted", "effect uncertain",
                                    str(self.root / "task"))
        supervisor._persist(receipt)
        supervisor._emit("run", "interrupted", evidence=["scratch/actor-journal.jsonl"])
        restored = Ghost.from_checkpoint(ghost.checkpoint)
        events = []
        next_supervisor = ManagedRunSupervisor(restored, runtime_root=self.root, clock=lambda: 1,
                                              workflows={}, event_sink=events.append)
        next_supervisor.replay_events()
        identity = events[0]["event_id"]
        next_supervisor.replay_events()
        self.assertEqual(events[1]["event_id"], identity)
        next_supervisor.acknowledge_event("run", identity)
        next_supervisor.replay_events()
        self.assertEqual(len(events), 2)
        self.assertEqual(next_supervisor._threads, {})

    def test_an_old_cancel_request_cannot_cancel_a_replacement_epoch(self):
        supervisor = ManagedRunSupervisor(small_ghost(), runtime_root=self.root, clock=lambda: 1, workflows={})
        receipt = ManagedRunReceipt("run", "action", "minecraft", "exchange", "running", "active",
                                    str(self.root / "task"))
        supervisor._persist(receipt)
        cancelled = supervisor.terminate("run", request_id="accepted-cancel")
        self.assertEqual(cancelled["state"], "cancelled")
        replacement = ManagedRunReceipt.from_dict(supervisor.status("run"))
        replacement.state, replacement.execution_epoch = "running", 2
        supervisor._persist(replacement)
        supervisor._cancels["run"] = threading.Event()
        replayed = supervisor.terminate("run", request_id="accepted-cancel")
        self.assertEqual(replayed["state"], "running")
        self.assertFalse(supervisor._cancels["run"].is_set())
        self.assertEqual(supervisor.terminate("run", request_id="new-cancel")["state"], "cancelled")

    def test_closed_instruction_boundary_stays_closed_after_a_read(self):
        ghost = small_ghost()
        supervisor = ManagedRunSupervisor(ghost, runtime_root=self.root, clock=lambda: 1, workflows={})
        receipt = ManagedRunReceipt("run", "action", "minecraft", "exchange", "running", "active", str(self.root / "task"))
        supervisor._persist(receipt)
        supervisor._cancels["run"] = threading.Event()
        supervisor.instructions_at_boundary("run", close=True)
        supervisor.instructions_at_boundary("run")
        with self.assertRaises(ManagedRunError):
            supervisor.update("run", "late")

    def test_captured_uncertain_effects_remain_uncertain(self):
        capture = json.loads((PROJECT_ROOT / "tests/fixtures/alpha/a12-interruptions.json").read_text(encoding="utf-8"))
        for phase in ("in_flight", "after_effect"):
            scenario = next(row for row in capture if row["phase"] == phase)
            self.assertEqual(scenario["reconciliation"]["retained_actions"][0]["outcome"], "uncertain")
            self.assertFalse(scenario["reconciliation"]["automatic_mutation_replay"])
            self.assertIn("Stale Minecraft body epoch", scenario["rejection"])

    def test_a_torn_final_append_does_not_hide_earlier_original_results(self):
        capture = json.loads((PROJECT_ROOT / "tests/fixtures/alpha/a12-interruptions.json").read_text(encoding="utf-8"))
        original = capture[1]["reconciliation"]["retained_actions"][0]["dispatch"]
        path = self.root / "journal.jsonl"
        path.write_text(json.dumps(original) + '\n{"event":', encoding="utf-8")
        rows = read_actor_journal(path)
        self.assertEqual(rows[0], original)
        self.assertEqual(rows[1]["event"], "torn_tail")
