"""Deterministic checks for the retained-question rating arithmetic.

These are small in-process checks of a pure function. The vectors are explicit
inputs with known cosines, not the output of an embedding service. They prove
the weighting, the noise bound, and the two selection modes. Whether a real
embedding model rates related text as similar belongs to the live check in
``tests/live/test_displaced_offer_salience.py``.
"""
from __future__ import annotations

from collections import Counter
import random
import unittest

from workflows.inner_thought.offer_salience import (
    EXPLOIT_SALIENCE_BIAS, RATING_NOISE, SIMILARITY_WEIGHT, SOURCE_OVERLAP_WEIGHT, AttentionCue, RetainedQuestion,
    attention_cues, rate_retained_questions, select_retained_offer,
)
from workflows.inner_thought.request_views import MainFrameView, RecentThoughtView
from workflows.inner_thought.schemas import (
    InnerRequest, RetainedOfferSelection, RetainedQuestionRating, ThoughtCandidate,
)

#: Two orthogonal directions and the diagonal between them (cosine 0.7071 to each).
EAST = [1.0, 0.0]
NORTH = [0.0, 1.0]
NORTH_EAST = [1.0, 1.0]


class RatingTests(unittest.TestCase):
    """Source overlap and similarity are separate signals with one weighted sum."""

    def setUp(self) -> None:
        self.sketch = RetainedQuestion(receipt_id=10, text="Why is the sketch unfinished?", source_ids=(1, 2))
        self.tap = RetainedQuestion(receipt_id=20, text="Is the dripping tap a sign of a larger leak?", source_ids=(3,))

    def rate(self, cues: list[AttentionCue], cue_vectors: list[list[float]], *,
             salience_bias: float = 1.0) -> tuple[RetainedQuestionRating, ...]:
        """Rate both questions; salience bias 1 removes the noise from the arithmetic."""
        return rate_retained_questions([self.sketch, self.tap], cues, question_vectors=[EAST, NORTH],
            cue_vectors=cue_vectors, salience_bias=salience_bias, rng=random.Random("rating"))

    def test_the_weights_sum_to_one(self) -> None:
        self.assertEqual(SOURCE_OVERLAP_WEIGHT + SIMILARITY_WEIGHT, 1.0)

    def test_a_held_source_cues_its_question_without_any_similar_text(self) -> None:
        # The context holds record 1, a source of the sketch question, and its text points elsewhere.
        sketch, tap = self.rate([AttentionCue(1, "candidate", "an unrelated remark")], [[-1.0, -1.0]])
        self.assertEqual(sketch.shared_source_ids, (1,))
        self.assertEqual(sketch.source_overlap, 0.5)
        self.assertEqual(sketch.similarity, 0.0)
        self.assertEqual(sketch.relevance, SOURCE_OVERLAP_WEIGHT * 0.5)
        self.assertEqual(tap.shared_source_ids, ())
        self.assertEqual(tap.relevance, 0.0)

    def test_similar_text_cues_a_question_that_shares_no_source(self) -> None:
        # Record 99 is no source of either question; its meaning matches the tap question.
        sketch, tap = self.rate([AttentionCue(99, "candidate", "water keeps dripping in the kitchen")], [NORTH])
        self.assertEqual(tap.source_overlap, 0.0)
        self.assertAlmostEqual(tap.similarity, 1.0)
        self.assertEqual(tap.most_similar_cue_id, 99)
        self.assertAlmostEqual(tap.relevance, SIMILARITY_WEIGHT)
        self.assertEqual(sketch.similarity, 0.0)
        self.assertIsNone(sketch.most_similar_cue_id)

    def test_one_related_cue_is_enough_among_unrelated_ones(self) -> None:
        cues = [AttentionCue(7, "recent_thought", "unrelated"), AttentionCue(None, "main_turn", "half related"),
                AttentionCue(8, "goal", "the sketch itself")]
        sketch, _ = self.rate(cues, [NORTH, NORTH_EAST, EAST])
        self.assertAlmostEqual(sketch.similarity, 1.0)
        self.assertEqual(sketch.most_similar_cue_id, 8)

    def test_a_cue_without_a_record_can_be_similar_and_shares_no_source(self) -> None:
        sketch, _ = self.rate([AttentionCue(None, "main_turn", "talk about the sketch")], [EAST])
        self.assertAlmostEqual(sketch.similarity, 1.0)
        self.assertIsNone(sketch.most_similar_cue_id)
        self.assertEqual(sketch.shared_source_ids, ())

    def test_both_signals_add_up(self) -> None:
        sketch, _ = self.rate([AttentionCue(1, "candidate", "a"), AttentionCue(2, "candidate", "b")],
                              [NORTH_EAST, NORTH])
        self.assertEqual(sketch.source_overlap, 1.0)
        self.assertAlmostEqual(sketch.similarity, 0.7071, places=4)
        self.assertAlmostEqual(sketch.relevance, SOURCE_OVERLAP_WEIGHT + SIMILARITY_WEIGHT * 0.7071, places=4)

    def test_noise_stays_inside_its_bound_and_vanishes_under_full_focus(self) -> None:
        cues = [AttentionCue(99, "candidate", "something")]
        for salience_bias in (0.0, 0.5):
            for rating in self.rate(cues, [NORTH_EAST], salience_bias=salience_bias):
                self.assertLessEqual(abs(rating.noise), RATING_NOISE * (1.0 - salience_bias))
        self.assertEqual([rating.noise for rating in self.rate(cues, [NORTH_EAST], salience_bias=1.0)], [0.0, 0.0])

    def test_no_cue_gives_no_relevance(self) -> None:
        sketch, tap = self.rate([], [])
        self.assertEqual((sketch.relevance, tap.relevance), (0.0, 0.0))

    def test_every_text_needs_its_vector(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one embedding vector"):
            rate_retained_questions([self.sketch, self.tap], [], question_vectors=[EAST], cue_vectors=[],
                                    salience_bias=1.0, rng=random.Random("short"))


class SelectionTests(unittest.TestCase):
    """Focused attention takes the top rating; wider attention samples by rating."""

    def setUp(self) -> None:
        self.strong = RetainedQuestion(receipt_id=10, text="strongly cued", source_ids=(1,))
        self.weak = RetainedQuestion(receipt_id=20, text="weakly cued", source_ids=(2,))
        # Record 1 is held and its text matches: the strong question has relevance 1.
        self.cues = [AttentionCue(1, "candidate", "the strong topic")]

    def select(self, *, salience_bias: float, seed: str) -> RetainedOfferSelection:
        """Select between the weak and the strong question, passed in the opposite of receipt order."""
        return select_retained_offer([self.weak, self.strong], self.cues, question_vectors=[NORTH, EAST],
            cue_vectors=[EAST], salience_bias=salience_bias, seed=seed)

    def test_focused_attention_offers_the_highest_rating(self) -> None:
        selection = self.select(salience_bias=EXPLOIT_SALIENCE_BIAS, seed="focused")
        self.assertEqual(selection.mode, "highest_rating")
        self.assertEqual(selection.offered_receipt_id, self.strong.receipt_id)
        # Ratings are reported in receipt order, whatever order the caller passed.
        self.assertEqual([rating.receipt_id for rating in selection.ratings], [10, 20])

    def test_the_same_opportunity_seed_repeats_the_same_selection(self) -> None:
        self.assertEqual(self.select(salience_bias=0.2, seed="opportunity-a"),
                         self.select(salience_bias=0.2, seed="opportunity-a"))

    def test_wider_attention_samples_and_still_favours_the_stronger_cue(self) -> None:
        both_cued = [AttentionCue(1, "candidate", "the strong topic"), AttentionCue(None, "main_turn", "in between")]
        offered: Counter[int] = Counter()
        for index in range(400):
            # The weak question has no shared source and cosine 0.7071 to the second cue.
            selection = select_retained_offer([self.weak, self.strong], both_cued,
                question_vectors=[NORTH, EAST], cue_vectors=[EAST, NORTH_EAST], salience_bias=0.0,
                seed=f"opportunity-{index}")
            self.assertEqual(selection.mode, "weighted_random")
            offered[selection.offered_receipt_id] += 1
        self.assertGreater(offered[self.weak.receipt_id], 0, "A weakly cued question must still surface sometimes.")
        self.assertGreater(offered[self.strong.receipt_id], offered[self.weak.receipt_id])

    def test_uncued_questions_are_sampled_uniformly_instead_of_never(self) -> None:
        offered: Counter[int] = Counter()
        for index in range(200):
            selection = select_retained_offer([self.weak, self.strong], [], question_vectors=[NORTH, EAST],
                cue_vectors=[], salience_bias=0.5, seed=f"idle-{index}")
            offered[selection.offered_receipt_id] += 1
        self.assertGreater(offered[self.weak.receipt_id], 40)
        self.assertGreater(offered[self.strong.receipt_id], 40)

    def test_a_tie_under_focus_offers_the_newer_receipt(self) -> None:
        selection = select_retained_offer([self.weak, self.strong], [], question_vectors=[NORTH, EAST],
            cue_vectors=[], salience_bias=1.0, seed="tie")
        self.assertEqual(selection.offered_receipt_id, self.weak.receipt_id)

    def test_one_question_is_never_rated(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two waiting questions"):
            select_retained_offer([self.strong], self.cues, question_vectors=[EAST], cue_vectors=[EAST],
                                  salience_bias=0.5, seed="single")


class CueTests(unittest.TestCase):
    """The cues are the records the private stages can cite, plus the main turn summary."""

    def test_cues_keep_record_identity_and_skip_empty_text(self) -> None:
        request = InnerRequest(
            main_frame=MainFrameView(tick=4, frame_id=None, summary="Jesse mentioned the kitchen."),
            candidates=[ThoughtCandidate(knoxel_id=5, text="The tap drips."), ThoughtCandidate(knoxel_id=6, text=" ")],
            recent_thoughts=[RecentThoughtView(knoxel_id=7, text="Water again.")])
        cues = attention_cues(request)
        self.assertEqual([(cue.record_id, cue.kind) for cue in cues],
                         [(None, "main_turn"), (5, "candidate"), (7, "recent_thought")])

    def test_an_empty_request_has_no_cue(self) -> None:
        self.assertEqual(attention_cues(InnerRequest()), ())


if __name__ == "__main__":
    unittest.main()
