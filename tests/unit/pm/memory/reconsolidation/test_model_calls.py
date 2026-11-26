"""Deterministic checks of memory answer validation and the correction prompt boundary."""
from __future__ import annotations

import asyncio
from threading import Event
import unittest

from pydantic import ValidationError

from pm.app.character_cli import build_models
from pm.context_render.correction import CorrectionHistory, JsonCorrectionInput, render_json_correction
from pm.context_render.prompt import RenderSettings
from pm.context_render.turns import Role, Turn
from pm.memory.reconsolidation.model_calls import ModelCalls
from pm.memory.reconsolidation.schemas import (FactClaimOption, FactClassificationValidation, FactExtractionValidation,
                                              FactsAnswer, FilingAnswer, MentalStateAnswer, NarrativeAnswer, SummaryAnswer)


class ModelInputContractTests(unittest.TestCase):
    def test_summary_and_narrative_require_actual_content(self) -> None:
        for schema in (SummaryAnswer, MentalStateAnswer, NarrativeAnswer):
            with self.assertRaises(ValidationError):
                schema.model_validate_json("{}")
        context = FactExtractionValidation(line_count=2)
        with self.assertRaises(ValidationError):
            FactsAnswer.model_validate_json("{}", context=context)
        supplied_empty = FactsAnswer.model_validate_json('{"facts":[]}', context=context)
        self.assertEqual(supplied_empty.facts, [])

    def test_invalid_citation_diagnostic_preserves_original_input(self) -> None:
        context = FactExtractionValidation(line_count=2)
        rejected = '{"facts":[{"text":"Jesse says he likes tea.","lines":[9]}]}'
        with self.assertRaises(ValidationError) as failure:
            FactsAnswer.model_validate_json(rejected, context=context)
        original = (Turn(Role.System, "Recover the source-supported assertion."), Turn(Role.User, "[1] Jesse says he likes tea."))
        diagnostic = CorrectionHistory().record_rejection("memory fact extraction", str(failure.exception))
        request = JsonCorrectionInput("memory fact extraction", original, rejected, diagnostic)
        settings = RenderSettings(context_window=32000, input_fraction=.8, reserved_output_tokens=1024)
        rendered = render_json_correction(request, settings)
        self.assertEqual(rendered.turns[:-1], original)
        self.assertIn("do not exist", rendered.turns[-1].content.text)
        self.assertEqual(rendered.turns[-1].role, Role.User)

    def test_claim_from_another_source_or_subject_cannot_be_selected(self) -> None:
        a = FactClaimOption(claim_id="a", about="entity:1", statement="Jesse says he likes tea.")
        b = FactClaimOption(claim_id="b", about="entity:2", statement="Mara plays the guitar.")
        context = FactClassificationValidation(fact_count=2, allowed_subjects=(a.about, b.about),
                                               claim_options=((a,), (b,)))
        first = '{"index":1,"about":"entity:1","entity_claim":"a","essence":"Jesse likes tea.","category":["people_preferences"]}'
        second = '{"index":2,"about":"entity:2","entity_claim":"b","essence":"Mara plays guitar.","category":["people_preferences"]}'
        valid = '{"facts":[' + first + ',' + second + ']}'
        accepted = FilingAnswer.model_validate_json(valid, context=context)
        self.assertEqual([row.entity_claim for row in accepted.facts], ["a", "b"])
        with self.assertRaisesRegex(ValidationError, "listed claim"):
            FilingAnswer.model_validate_json(valid.replace('"entity_claim":"b"', '"entity_claim":"a"'), context=context)
        with self.assertRaisesRegex(ValidationError, "listed claim"):
            FilingAnswer.model_validate_json(valid.replace('"about":"entity:2"', '"about":"entity:1"'), context=context)

    def test_cancellation_stops_request_before_transport(self) -> None:
        stop = Event()
        stop.set()
        model = build_models("config.yaml").stage
        calls = ModelCalls(model, cancel=stop)
        with self.assertRaises(asyncio.CancelledError):
            calls.session.complete((Turn(Role.User, "Do not send this cancelled request."),))
        self.assertEqual(model.request_count, 0)
