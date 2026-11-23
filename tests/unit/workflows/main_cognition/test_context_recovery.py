"""Deterministic tests of the recall candidate-id rule.

A recall answer may only name memories the phase actually offered. The rule
lives in the answer schema, validated with a caller-supplied context, so the
structured-output repair loop of ``pm.llm.structured.complete_json`` rejects a
wrong id and asks again instead of ending the cycle. These tests cover the
contract and the prompt shape only. They replay no model answer, and they
prove nothing about a real model converging on a valid answer. That gate is
the live gate ``test_a_copied_story_id_is_rejected_and_repaired_into_the_candidates``
in ``tests/live/test_recall_candidate_repair.py``.
"""
from __future__ import annotations

import unittest

from pydantic import ValidationError

from pm.memory.evidence import EvidenceItem as MemoryCandidate
from workflows.main_cognition.phases import retrieve
from workflows.main_cognition.schemas import ContextResult, RecallValidationContext


def _candidate(knoxel_id: int) -> MemoryCandidate:
    return MemoryCandidate(knoxel_id=knoxel_id, text=f"memory {knoxel_id}", score=1.0, revision=1,
                           memory_kind="Dialogue", evidence_role="user_report", historical=False)


class RecallCandidateIdRuleTests(unittest.TestCase):
    """The schema, not the phase body, refuses an id that was never offered."""

    def test_an_id_outside_the_supplied_candidates_is_rejected_by_name(self) -> None:
        answer = {"memories": [], "active_goals": [], "candidate_ids": [15377, 15368], "belief_updates": []}
        offered = RecallValidationContext(candidate_ids=frozenset({15342, 15377}))

        with self.assertRaises(ValidationError) as raised:
            ContextResult.model_validate(answer, context=offered)

        problem = str(raised.exception)
        self.assertIn("15368", problem)
        self.assertIn("candidate_id=", problem)

    def test_ids_inside_the_supplied_candidates_are_accepted(self) -> None:
        answer = {"memories": [], "active_goals": [], "candidate_ids": [15342, 15377], "belief_updates": []}
        offered = RecallValidationContext(candidate_ids=frozenset({15342, 15377}))

        result = ContextResult.model_validate(answer, context=offered)

        self.assertEqual(result.candidate_ids, [15342, 15377])

    def test_an_answer_without_the_context_keeps_the_old_permissive_rule(self) -> None:
        # Only the retrieve phase knows the offered ids. Other callers of the
        # schema, such as a plain replay, keep their old permissive behaviour.
        result = ContextResult.model_validate({"candidate_ids": [999]})

        self.assertEqual(result.candidate_ids, [999])


class RecallCandidateRenderingTests(unittest.TestCase):
    """The offered id must be readable as the only copyable number."""

    def test_each_candidate_leads_with_its_marker_line(self) -> None:
        rendered = retrieve.render_memories([_candidate(15377), _candidate(15342)])

        first, _, remainder = rendered.partition("\n")
        self.assertEqual(first, "candidate_id=15377 | revision 1 | Dialogue | user_report | current")
        self.assertTrue(remainder.startswith("memory 15377"))
        self.assertIn("candidate_id=15342", rendered)

    def test_the_phase_prompt_names_the_marker_the_validator_demands(self) -> None:
        # The repair turn quotes the validator message. If the prompt and the
        # message disagree about the marker, the model cannot repair the answer.
        template = retrieve.phase_prompts("retrieve").user_template

        self.assertIn("candidate_id=", template)


if __name__ == "__main__":
    unittest.main()
