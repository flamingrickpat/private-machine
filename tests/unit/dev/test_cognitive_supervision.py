"""Check real worker exits and immutable evidence without model generation."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

from dev.work_items.a18b.cli import bundle, failed_case, review_case, review_worker, summarize
from dev.work_items.a18b.identity import save_suite
from dev.work_items.a18b.report import render
from dev.work_items.a18b.runner import load_suite, make_packets, public_messages
from dev.work_items.a18b.supervision import run_worker
from dev.work_items.shared.evidence import save_json, sha256_file


class CognitiveSupervisionTests(unittest.TestCase):
    def setUp(self):
        self.suite = load_suite()

    def origin(self, root):
        save_json(root / "run-origin.json", {"scope": "mechanism check only", "suite_version": self.suite.version,
            "suite_snapshot_sha256": save_suite(root, self.suite)})

    def test_real_nonzero_worker_exit_is_retained(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            with (root / "worker.log").open("w", encoding="utf-8") as log:
                result = run_worker([sys.executable, "-c", "raise SystemExit(17)"], cwd=root,
                    case=root / "case", log=log, timeout_seconds=5)
            self.assertEqual(result.returncode, 17)
            self.assertIsNone(result.termination_reason)
            self.assertFalse(result.forced)

    def test_hung_worker_has_a_bounded_deadline(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            case = root / "case"
            case.mkdir()
            with (root / "worker.log").open("w", encoding="utf-8") as log:
                result = run_worker([sys.executable, "-c", "import time; time.sleep(60)"], cwd=root,
                    case=case, log=log, timeout_seconds=0.2, cleanup_seconds=0.2)
            self.assertEqual(result.termination_reason, "case deadline exceeded")
            self.assertTrue(result.forced)
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue((case / "STOP").exists())
            self.assertLess(result.elapsed_seconds, 10)

    def test_nonzero_exit_blocks_a_successful_runtime_receipt(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            scenario = self.suite.scenarios[0]
            save_json(root / "private-setup.json", scenario.model_dump(mode="json"))
            packets = make_packets(scenario, self.suite, (), ())
            for audience, packet in packets.items():
                save_json(root / f"{audience}-packet.json", packet.model_dump(mode="json"))
            save_json(root / "receipt.json", {"scenario": scenario.id, "suite_version": self.suite.version,
                "cadence_policy": self.suite.cadence_policy, "runtime_ok": True,
                "episodes": [{"sent": scene.messages, "completed_at": "retained"}
                    for scene in scenario.scenes], "new_companion_messages": 0,
                "packet_sha256": {audience: sha256_file(root / f"{audience}-packet.json") for audience in packets}})
            save_json(root / "generation-worker.json", {"returncode": 17, "termination_reason": None})
            # No model is called: zero new speech leaves both panels incomplete.
            result = review_case(root, scenario, self.suite, None)
            self.assertFalse(result["complete"])
            self.assertTrue(any("exit=17" in error for error in result["review_errors"]))

    def test_malformed_case_keeps_aggregate_and_report_available(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.origin(root)
            scenario = self.suite.scenarios[0]
            case = root / scenario.id
            case.mkdir()
            # The actual child rejects a missing setup before any inference.
            result = review_worker(case, scenario, self.suite, Path("config.alpha.openrouter.yaml"))
            self.assertFalse(result["complete"])
            self.assertTrue(result["review_errors"])
            second = self.suite.scenarios[1]
            other = failed_case(root / second.id, second, self.suite, "Another unavailable case.")
            report = summarize(root, self.suite, [result, other])
            self.assertEqual(report["evaluated_cases"], 2)
            self.assertEqual(report["declared_cases"], len(self.suite.scenarios))
            self.assertEqual(report["status"], "incomplete")
            page = render(root, self.suite, report).read_text(encoding="utf-8")
            self.assertIn("Public packet unavailable", page)
            self.assertIn("Another unavailable case.", page)

    def test_seed_only_bundle_is_not_new_reviewable_companion_output(self):
        with TemporaryDirectory(dir=Path.cwd() / ".runtime") as temp:
            root = Path(temp)
            self.origin(root)
            scenario = self.suite.scenarios[0]
            case = root / scenario.id
            row = failed_case(case, scenario, self.suite, "No fresh companion speech.")
            packet = make_packets(scenario, self.suite, public_messages([{
                "at": "2026-10-01T00:00:00+00:00", "speaker": "Spamuel", "text": "Historical boundary input.",
            }]), ())["editor"]
            save_json(case / "editor-packet.json", packet.model_dump(mode="json"))
            result = json.loads(bundle(root, self.suite, summarize(root, self.suite, [row])).read_text(encoding="utf-8"))
            self.assertFalse(result["items"][0]["reviewable"])

    def test_changed_packet_is_rejected_before_review(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            scenario = self.suite.scenarios[0]
            save_json(root / "private-setup.json", scenario.model_dump(mode="json"))
            save_json(root / "receipt.json", {"scenario": scenario.id, "suite_version": self.suite.version,
                "cadence_policy": self.suite.cadence_policy,
                "runtime_ok": False, "new_companion_messages": 1, "packet_sha256": {"editor": "not-the-file-hash"}})
            (root / "editor-packet.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "terminal receipt"):
                review_case(root, scenario, self.suite, None)
