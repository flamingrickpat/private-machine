"""Focused checks for the K003 evidence reader.

These tests build a small evidence folder to prove the file joins and path
checks. They are not a substitute for the real K002 acceptance run; the real
runs are the integration evidence in ``docs/work_items/k003/index.md``.
"""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pm.tools.timeline.calls import index_calls
from pm.tools.timeline.evidence import EvidenceError, EvidenceStore, normalize_time
from pm.tools.timeline.records import index_records
from pm.tools.timeline.timeline import AttemptView
from pm.tools.timeline.workflows import index_kernels


def write_json(path: Path, value: object) -> None:
    """Write one UTF-8 JSON file, creating its folder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


class TimelineEvidenceTests(unittest.TestCase):
    def test_normalize_time_compares_offsets_on_one_clock(self) -> None:
        self.assertEqual(normalize_time("2026-10-29T21:00:00+01:00"), "2026-10-29T20:00:00+00:00")
        self.assertEqual(normalize_time("2026-10-29T20:00:00"), "2026-10-29T20:00:00+00:00")

    def test_run_identity_cannot_escape_the_evidence_root(self) -> None:
        with TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp))
            with self.assertRaises(EvidenceError):
                store.run_root("../outside")

    def test_call_workflow_and_speech_join_in_one_timeline(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run1"
            write_json(run / "state.json", {
                "id": "run1", "title": "Test run", "status": "completed", "created_at": "2026-10-29T20:00:00+00:00",
                "participants": [{"id": "primary"}], "edges": [], "clock": "2026-10-29T20:00:00+00:00",
                "timeline": [{"kind": "actor", "speaker": "Jesse", "target": "primary",
                              "timestamp": "2026-10-29T20:00:00+00:00", "content": "hello",
                              "delivery_id": "d1"}],
                "transit": [], "facts": {}, "failures": [], "reports": ["attempts/1/report.json"],
            })
            write_json(run / "plan.json", {"title": "Test run", "brief": "brief", "start_time": "2026-10-29T20:00:00+00:00",
                                           "facts": [], "steps": []})
            attempt = run / "attempts" / "1"
            write_json(attempt / "report.json", {"run": {"status": "completed", "timeline": [], "failures": [], "facts": {}}})
            evidence = attempt / "evidence" / "primary"
            record = {
                "__type__": "Feature",
                "data": {
                    "id": 1, "tick_id": 1, "content": "hello", "feature_type": "Dialogue",
                    "metadata": {"delivery_id": "d1"}, "provenance": {"source_ref": "Jesse"},
                    "timestamp_world_begin": "2026-10-29T20:00:00",
                    "timestamp_creation": "2026-10-29T20:00:01",
                },
            }
            write_json(evidence / "chat.json", {"schema_id": "pm.record-package.v1", "view": "chat",
                                                "record_ids": [1], "records": [record], "revisions": {}})
            write_json(evidence / "activity.json", {"schema_id": "pm.record-package.v1", "view": "records",
                                                    "record_ids": [1], "records": [record], "revisions": {}})
            write_json(evidence / "cycles" / "1.json", {
                "schema_id": "pm.record-package.v1", "view": "cycle", "record_ids": [1], "records": [record],
                "revisions": {}, "gwt": {"tick_id": 1, "timestamp": "2026-10-29T20:00:02",
                                         "summary": "a cycle", "knoxel_ids": [1], "stimulus_ids": [1],
                                         "trace": {"codelet_runs": []}},
                "cycle": {"tick_id": 1, "timestamp": "2026-10-29T20:00:00"},
            })
            prompt = evidence / "logs" / "prompts"
            write_json(prompt / "call1.json", {
                "endpoint": "http://127.0.0.1:8080/v1", "model": "test", "route": "chat/completions",
                "owner_thread": "inference_0", "module": "main_cog", "invocation_id": "inv1",
                "world_time": "2026-10-29T20:00:00+00:00", "started_at": "2026-10-29T20:00:01+00:00",
                "payload": {"messages": [{"role": "user", "content": "hello"}]},
                "response": {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                             "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}},
            })
            (prompt / "usage-1.jsonl").write_text(
                json.dumps({"call_id": "call1", "event": "start", "world_time": "2026-10-29T20:00:00+00:00",
                            "module": "main_cog", "invocation_id": "inv1"}) + "\n" +
                json.dumps({"call_id": "call1", "event": "end", "elapsed_seconds": 0.2,
                            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}}) + "\n",
                encoding="utf-8")
            kernel = evidence / "runtime" / "kernel" / "inv1"
            write_json(kernel / "journal.jsonl", {})
            (kernel / "journal.jsonl").write_text(
                json.dumps({"phase": "intake", "kind": "role", "role": "phase", "ok": True,
                            "verdict": "done", "status": "done", "result": {"summary": "one stimulus"},
                            "timestamp": "2026-10-29T20:00:01+00:00"}) + "\n", encoding="utf-8")
            (kernel / "state.md").write_text("# Workflow state\n\n- workflow: main_cognition\n", encoding="utf-8")

            store = EvidenceStore(root)
            view = AttemptView.load(store, "run1", 1)
            calls = index_calls("primary", evidence / "logs" / "prompts", evidence)
            kernels = index_kernels("primary", evidence / "runtime" / "kernel", evidence, calls)
            records = index_records("primary", evidence)
            self.assertEqual(calls.get("call1").response_text, "hi")
            self.assertEqual(kernels.get("inv1").workflow, "main_cognition")
            self.assertEqual(records.conversation[0].speaker, "Jesse")
            events = view.timeline()
            kinds = {event.kind for event in events}
            self.assertTrue({"actor", "speech", "call", "workflow", "cycle"} <= kinds)
            self.assertEqual(len(view.overviews()), 1)


if __name__ == "__main__":
    unittest.main()
