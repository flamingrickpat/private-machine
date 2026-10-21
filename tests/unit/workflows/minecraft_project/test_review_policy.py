"""Check passing judgments without converting plan approval into task proof."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pm.runs.review import review_passes
from workflows.minecraft_project.planning.review import record_review
from workflows.minecraft_project.schemas import PlanReview, ProjectRequest, TaskTree, VerificationReview
from workflows.minecraft_project.tree_store import TreeStore

ROOT = Path(__file__).resolve().parents[4]


class ReviewPolicyTests(unittest.TestCase):
    def test_flexible_approval_releases_a_reviewed_plan_without_claiming_execution(self):
        captured = json.loads((ROOT / "test_data/captures/alpha/project-stable-child-rejection.json").read_text(encoding="utf-8"))
        for judgment in ("good_enough", "equivalent_or_better"):
            with self.subTest(judgment=judgment), TemporaryDirectory() as directory:
                store = TreeStore(Path(directory))
                retained = store.revise(TaskTree.model_validate(captured["tree"]))
                record_review(store, PlanReview(status=judgment, summary="The practical request is covered."))
                restored = TreeStore(Path(directory))
                self.assertEqual(restored.state["reviewed_revision"], retained.revision)
                self.assertEqual(restored.state["review_judgment"], judgment)
                self.assertFalse(restored.state["acceptances"])
                self.assertEqual(restored.load(), retained)

    def test_strict_mode_removes_an_earlier_flexible_approval(self):
        with TemporaryDirectory() as directory:
            store = TreeStore(Path(directory))
            store.state.update(revision=1, reviewed_revision=1)
            record_review(store, PlanReview(status="good_enough", summary="An approximate result."), mode="strict")
            self.assertNotIn("reviewed_revision", TreeStore(Path(directory)).state)
            self.assertFalse(review_passes("good_enough", "strict"))
            self.assertEqual(ProjectRequest().review_mode, "flexible")

    def test_verification_does_not_require_the_model_to_retype_criterion_ids(self):
        review = VerificationReview(status="equivalent_or_better", summary="The observed result meets the request.")
        self.assertEqual(review.criterion_ids, [])
        self.assertTrue(review_passes(review.status))
