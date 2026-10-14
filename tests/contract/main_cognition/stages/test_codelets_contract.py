"""Direct Codelets stage tests.

No model endpoint, module thread, or bus. The kernel test uses the real kernel
and the real Python role with a deterministic ``llm_complete`` tool. The tests
check proposal contracts: origin, reality, allowed targets, source revisions,
and failure records. Live proposal quality is checked by the A15 gate.
"""
from __future__ import annotations

import json
import threading
import unittest
from pathlib import Path

from pm_workflows.runtime import WorkflowRuntime
from pm.runs.workflow_runner import PROJECT_ROOT, WorkflowRequest, WorkflowRunner, WorkflowStatus
from workflows.main_cognition.schemas import CodeletsResult
from workflows.main_cognition.stages.codelets import steps
from pm.memory.bash_view import MemoryBashEnvironment

MANIFEST = Path(steps.__file__).resolve().parent / "codelets.workflow.md"


def request(kind: str = "user.input", **extra) -> dict:
    return {
        "companion_name": "Spamuel",
        "user_name": "Jesse",
        "stimuli": [{"knoxel_id": 11, "kind": kind, "source": "user_input" if kind == "user.input" else "Spamuel",
                     "text": "You forgot again."}],
        "stimulus_revisions": {"11": 1},
        "memory_candidates": [{"knoxel_id": 42, "text": "Jesse waited all day last week.", "revision": 3}],
        "context": {"candidate_ids": [42]},
        "affect": {"summary": "hurt", "regulation": "reflective"},
        "state_signals": {"core.negative_valence": 0.6, "emotion.sadness": 0.6, "emotion.strongest": 0.6},
        "active_goal_records": [],
        **extra,
    }


class KernelTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(".runtime/w5-direct-tests")
        self.root.mkdir(parents=True, exist_ok=True)

    def _run(self, payload: dict, answer):
        calls: list = []

        def llm_complete(messages, **generation):
            calls.append(messages)
            return answer(messages) if callable(answer) else answer

        runtime = WorkflowRuntime(
            environment=MemoryBashEnvironment({}, {}, "read"), stop_event=threading.Event(),
            tools={"llm_complete": llm_complete}, propagate_python_errors=True,
        )
        runner = WorkflowRunner(MANIFEST, self.root, namespace="codelets", base_dir=PROJECT_ROOT)
        result = runner.invoke(WorkflowRequest("codelets-1", payload), runtime)
        self.assertEqual(result.status, WorkflowStatus.Completed)
        return CodeletsResult.model_validate(result.report), calls

    def test_llm_targets_outside_the_grant_are_dropped_and_revisions_recorded(self):
        answer = json.dumps({"fires": True, "text": "I remember waiting before.", "target_ids": [42, 999]})
        report, calls = self._run(request(), answer)
        llm = [item for item in report.proposals if item.codelet == "disappointment_echo"]
        self.assertEqual(len(llm), 1)
        self.assertEqual(llm[0].target_ids, [42])
        self.assertEqual(llm[0].dropped_target_ids, [999])
        self.assertEqual(llm[0].source_revisions, {"42": 3})
        # The bound is the catalogue policy, not a fixed number: A16 raised
        # the LLM budget when the repertoire grew.
        self.assertLessEqual(report.cost["llm_calls"], report.policy["max_llm_codelets"])
        self.assertEqual(report.cost["llm_calls"], len(calls))
        self.assertTrue(report.catalogue_digest)

    def test_thought_only_cycle_never_produces_an_outside_origin(self):
        answer = json.dumps({"fires": True, "text": "I keep thinking about it.", "target_ids": [11]})
        report, _ = self._run(request(kind="thought.created"), answer)
        self.assertTrue(report.proposals)
        self.assertNotIn("outside", {item.origin for item in report.proposals})
        direct = next(item for item in report.proposals if item.codelet == "direct_stimulus")
        self.assertEqual((direct.origin, direct.reality), ("thought", "internal"))
        self.assertTrue(direct.text.startswith("I thought"))
        self.assertIn("thought_echo", {item.codelet for item in report.runs if item.status == "ran"})

    def test_the_current_event_is_not_recalled_as_a_memory(self):
        payload = request()
        payload["memory_candidates"] += [
            {"knoxel_id": 11, "text": "You forgot again.", "revision": 1},
            {"knoxel_id": 50, "text": "Jesse: You forgot again.", "revision": 1, "source_ids": [11]},
        ]
        payload["context"] = {"candidate_ids": [42, 11, 50]}
        report, _ = self._run(payload, json.dumps({"fires": False, "text": "", "target_ids": []}))
        recalled = [item.target_ids for item in report.proposals if item.codelet == "context_recall"]
        self.assertEqual(recalled, [[42]])

    def test_silent_and_skipped_codelets_are_recorded(self):
        report, _ = self._run(request(), json.dumps({"fires": False, "text": "", "target_ids": []}))
        statuses = {item.codelet: item.status for item in report.runs}
        self.assertEqual(statuses["direct_stimulus"], "ran")
        self.assertEqual(statuses["thought_echo"], "skipped")
        self.assertIn("silent", statuses.values())

    def test_only_belief_proposals_carry_a_topic(self):
        answer = json.dumps({"fires": True, "text": "I doubt myself here.",
                             "topic": "my_reliability", "target_ids": [42]})
        # Mild hurt with shame and guilt: the belief codelet consistency_check
        # clears the threshold while calmer interpretations wait their turn.
        report, _ = self._run(request(state_signals={
            "core.negative_valence": 0.16, "emotion.sadness": 0.36, "emotion.strongest": 0.5,
            "emotion.shame": 0.5, "emotion.guilt": 0.5,
        }), answer)
        beliefs = [item for item in report.proposals if item.proposal_kind == "belief"]
        self.assertTrue(beliefs)
        self.assertTrue(all(item.topic == "my_reliability" for item in beliefs))
        others = [item for item in report.proposals if item.proposal_kind != "belief"]
        self.assertTrue(others)
        # A model that answers a different codelet with a topic cannot smuggle
        # a belief key into a non-belief proposal.
        self.assertTrue(all(item.topic == "" for item in others))

    def test_precursor_records_raise_pathway_signals(self):
        payload = request()
        payload["pathway_precursors"] = [{
            "codelet": "curiosity_probe", "tick_id": 5, "selected": True,
            "record_id": 900, "attention_kind": "simulation", "reality": "imagined",
            "family": "imagination", "pathways": ["curiosity"], "revision": 1,
        }]
        payload["pathway_latest_tick"] = 6
        report, _ = self._run(payload, json.dumps({"fires": False, "text": "", "target_ids": []}))
        # One accepted precursor, one cycle old, at decay 0.6.
        self.assertAlmostEqual(report.signals["pathway.curiosity"], 0.6)
        self.assertEqual(report.pathway_signals, {"curiosity": 0.6})
        causes = [cause for item in report.runs for cause in item.cause if "pathway.curiosity" in cause]
        self.assertTrue(causes, "no scheduled codelet read the pathway signal")


if __name__ == "__main__":
    unittest.main()
