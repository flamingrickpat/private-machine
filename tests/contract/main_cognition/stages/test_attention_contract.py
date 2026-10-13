"""Direct Attention tests.

No model endpoint, module thread, or bus. The kernel test uses the real kernel
and the real Python role with a deterministic ``llm_complete`` tool. These
tests check the A15 contract: the model only scores nominations, and the pure
competition chooses winners. They do not establish model scoring quality;
the live A15 gate does that.
"""
from __future__ import annotations

import json
import threading
import unittest
from pathlib import Path

from pm_workflows.runtime import WorkflowRuntime
from pm.runs.workflow_runner import PROJECT_ROOT, WorkflowRequest, WorkflowRunner, WorkflowStatus
from workflows.main_cognition.schemas import AttentionNomination, AttentionResult, CodeletProposal
from workflows.main_cognition.stage_support import StageOutputError
from workflows.main_cognition.stages.attention import steps
from pm.memory.bash_view import MemoryBashEnvironment

MANIFEST = Path(steps.__file__).resolve().parent / "attention.workflow.md"


def proposals() -> list[dict]:
    return [
        CodeletProposal(nomination_id="n1", codelet="direct_stimulus", reality="observed",
                        text="Jesse said: the kitchen is on fire", target_ids=[11]).model_dump(),
        CodeletProposal(nomination_id="n2", codelet="direct_stimulus", reality="observed",
                        text="Jesse said: the carpet is beige", target_ids=[12]).model_dump(),
        CodeletProposal(nomination_id="n3", codelet="curiosity_probe", reality="imagined",
                        text="I wonder whether the carpet was always beige.", target_ids=[12]).model_dump(),
    ]


def example_request() -> dict:
    return {
        "companion_name": "Spamuel",
        "user_name": "Jesse",
        "state_summary": "smug but curious",
        "active_goals": ["keep the bit going"],
        "triage": {"items": [{"knoxel_id": 11, "category": "Critical"}]},
        "appraisal": {"feeling": "alarmed", "reasoning": "fire"},
        "context": {"memories": [], "candidate_ids": []},
        "codelets": {"proposals": proposals()},
        "candidates": [
            {"knoxel_id": 11, "kind": "stimulus", "text": "the kitchen is on fire"},
            {"knoxel_id": 12, "kind": "stimulus", "text": "the carpet is beige"},
        ],
    }


def scores(*rows) -> str:
    return json.dumps({"scores": [
        {"id": key, "urgency": u, "novelty": n, "goal_relevance": g, "reportability": r, "why": "x"}
        for key, u, n, g, r in rows]})


class MessageTests(unittest.TestCase):
    def test_messages_list_every_nomination_with_its_reality_label(self):
        items = [CodeletProposal.model_validate(item) for item in proposals()]
        messages = steps.build_messages(example_request(), steps.load_prompts(), items)
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertIn("Spamuel", messages[0]["content"])
        self.assertIn("n1 (happening now)", messages[1]["content"])
        self.assertIn("n3 (a possibility, it did not happen)", messages[1]["content"])


class ValidationGuardTests(unittest.TestCase):
    def test_selected_id_outside_candidates_is_rejected(self):
        result = AttentionResult(selected_ids=[999], reasoning="bad")
        with self.assertRaises(StageOutputError):
            steps.ensure_candidate_ids(result, {11, 12})

    def test_target_id_outside_candidates_is_rejected(self):
        result = AttentionResult(
            nominations=[AttentionNomination(target_ids=[999], reasoning="bad")],
            reasoning="bad",
        )
        with self.assertRaises(StageOutputError):
            steps.ensure_candidate_ids(result, {11, 12})


class KernelTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(".runtime/w5-direct-tests")
        self.root.mkdir(parents=True, exist_ok=True)

    def _run(self, *completions: str):
        calls: list = []
        replies = list(completions)

        def llm_complete(messages, **generation):
            calls.append(messages)
            return replies.pop(0)

        environment = MemoryBashEnvironment({}, {}, "read")
        runtime = WorkflowRuntime(
            environment=environment, stop_event=threading.Event(),
            tools={"llm_complete": llm_complete}, propagate_python_errors=True,
        )
        runner = WorkflowRunner(MANIFEST, self.root, namespace="attention", base_dir=PROJECT_ROOT)
        result = runner.invoke(WorkflowRequest("attention-1", example_request()), runtime)
        return result, calls

    def test_scores_decide_winners_and_every_loser_gets_a_reason(self):
        result, calls = self._run(scores(("n1", 10, 8, 9, 9), ("n2", 0, 1, 0, 1), ("n3", 1, 3, 1, 2)))
        self.assertEqual(result.status, WorkflowStatus.Completed)
        report = AttentionResult.model_validate(result.report)
        status = {item.nomination_id: item.status for item in report.nominations}
        self.assertEqual(status["n1"], "selected")
        self.assertEqual(status["n2"], "deferred")
        self.assertIn("below_min_score", next(item.reason for item in report.nominations if item.nomination_id == "n2"))
        self.assertEqual(report.selected_ids, [11])
        self.assertEqual(len(calls), 1)

    def test_missing_scores_get_one_repair_then_rejection(self):
        result, calls = self._run(scores(("n1", 9, 5, 9, 9)), scores(("n2", 1, 1, 1, 1)))
        report = AttentionResult.model_validate(result.report)
        status = {item.nomination_id: item.status for item in report.nominations}
        self.assertEqual(len(calls), 2)
        self.assertIn("n3", calls[1][-1]["content"])
        self.assertNotEqual(status["n2"], "rejected")
        self.assertEqual(status["n3"], "rejected")

    def test_invalid_model_output_raises_instead_of_a_successful_empty_answer(self):
        with self.assertRaises(StageOutputError):
            self._run("this is not json at all")


if __name__ == "__main__":
    unittest.main()
