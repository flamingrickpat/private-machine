"""Deterministic checks of template boundaries and rating evidence contracts.

These exercise real validation. Live planner, rating, and companion evidence
remain separate prerequisites for behavioral claims.
"""
from datetime import datetime, timezone
import unittest

from pydantic import ValidationError

from simulation.analysis import AnalysisSettings, RatingCriterion, SectionRating
from simulation.plans import DirectorSettings, Fact, FactKind, ScenarioPlan, Step
from simulation.templates import SimulationTemplate, TemplateConnection, TemplateParticipant


class TemplateTests(unittest.TestCase):
    def test_exact_plan_survives_json_round_trip_with_participant_graph(self) -> None:
        at = datetime(2026, 6, 1, 8, tzinfo=timezone.utc)
        plan = ScenarioPlan(title="A shared day", brief="A conversation", start_time=at,
            facts=[Fact(id="arrival", text="Jesse arrives", kind=FactKind.Event, at=at)],
            steps=[Step(id="morning", at=at, instruction="Jesse arrives", actor_intent="Talk",
                        fact_ids=["arrival"], completion="Arrival is conveyed", target="spamuel-a", minimum_turns=4)])
        template = SimulationTemplate(title=plan.title, brief=plan.brief, plan=plan,
            participants=[TemplateParticipant(id="spamuel-a", home="a"), TemplateParticipant(id="spamuel-b", home="b")],
            edges=[TemplateConnection(source="spamuel-a", target="spamuel-b")])
        restored = SimulationTemplate.model_validate_json(template.model_dump_json())
        self.assertEqual(restored, template)
        self.assertEqual(restored.plan.steps[0].minimum_turns if restored.plan else 0, 4)

    def test_duplicate_participant_and_unknown_connection_fail_before_execution(self) -> None:
        with self.assertRaisesRegex(ValidationError, "unique"):
            SimulationTemplate(brief="Hello", participants=[TemplateParticipant(id="a", home="one"), TemplateParticipant(id="a", home="two")])
        with self.assertRaisesRegex(ValidationError, "distinct"):
            SimulationTemplate(brief="Hello", participants=[TemplateParticipant(id="a", home="one")],
                edges=[TemplateConnection(source="a", target="missing")])

    def test_rating_cannot_claim_success_without_cited_evidence(self) -> None:
        with self.assertRaisesRegex(ValidationError, "citations"):
            SectionRating(status="rated", score=5, summary="Looks good")
        with self.assertRaisesRegex(ValidationError, "no numerical score"):
            SectionRating(status="insufficient_evidence", score=5, summary="Cannot tell")
        self.assertIsNone(SectionRating(status="insufficient_evidence", summary="No relevant exchange").score)

    def test_ambiguous_director_and_duplicate_rating_ids_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValidationError, "last conversation hour"):
            DirectorSettings(first_hour=22, last_hour=8)
        criterion = RatingCriterion(id="fluency", name="Fluency", instructions="Read the exchange")
        with self.assertRaisesRegex(ValidationError, "unique"):
            AnalysisSettings(criteria=[criterion, criterion])


if __name__ == "__main__":
    unittest.main()
