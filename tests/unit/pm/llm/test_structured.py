"""Reproduce invalid local-model answers without inference."""
import asyncio
from functools import partial
import json
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pydantic import BaseModel, Field, model_validator

from pm.llm.structured import complete_json, complete_json_async, ModelFormatError, repair_agent_result
from pm.memory.reconsolidation.schemas import NarrativeAnswer


class Answer(BaseModel):
    value: int = Field(ge=0)


def measure(turns, **settings):
    return {"input_tokens": len(json.dumps(turns)), "input_budget_tokens": 100000}


def complete_json_via_async(complete, messages, schema, **settings):
    """Run the async repair policy with the same completion as the sync policy."""
    async def async_complete(*args, **kwargs):
        return complete(*args, **kwargs)

    return asyncio.run(complete_json_async(async_complete, messages, schema, **settings))


class RepairPolicyTests(unittest.TestCase):
    def test_local_syntax_repair_needs_no_extra_completion_or_measurement(self):
        for run in (complete_json, complete_json_via_async):
            for raw in ('{value: 3}', '```json\n{"value": 3,}\n```', '{"value": 3',
                        'Answer: {"value": 3}'):
                with self.subTest(policy=run.__name__, raw=raw):
                    calls, rejected = [], []

                    def complete(turns, **settings):
                        calls.append(turns)
                        return raw

                    result = run(complete, [], Answer, on_rejection=lambda *row: rejected.append(row))
                    self.assertEqual(result.value, 3)
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(rejected, [])

    def test_ninth_rejection_resets_history_and_attempts_each_time(self):
        for run in (complete_json, complete_json_via_async):
            with self.subTest(policy=run.__name__):
                messages = [{"role": "system", "content": "Keep source 17."},
                            {"role": "user", "content": "Return a nonnegative value."}]
                original = [dict(turn) for turn in messages]
                responses = iter(['{"value": -1}'] * 19 + ['{"value": 3}'])
                calls, measured, rejected = [], [], []

                def complete(turns, **settings):
                    calls.append(turns)
                    self.assertEqual(turns[:2], original)
                    self.assertEqual(settings["temperature"], 0.2)
                    return next(responses)

                def measure_context(turns, **settings):
                    measured.append(turns)
                    self.assertEqual(settings["temperature"], 0.2)
                    return measure(turns)

                result = run(complete, messages, Answer, measure=measure_context, temperature=0.2,
                             on_rejection=lambda *row: rejected.append(row))
                self.assertEqual(result.value, 3)
                self.assertEqual([len(turns) for turns in calls], [2 + 2 * (i % 9) for i in range(20)])
                self.assertEqual([row[0] for row in rejected], list(range(1, 10)) * 2 + [1])
                self.assertEqual(measured, [turns for i, turns in enumerate(calls) if i % 9])
                self.assertEqual(messages, original)

    def test_reset_keeps_validation_context(self):
        class Cited(BaseModel):
            source: int

            @model_validator(mode="after")
            def allowed(self, info):
                if self.source not in info.context["allowed"]:
                    raise ValueError("Cite an inspected source.")
                return self

        for run in (complete_json, complete_json_via_async):
            with self.subTest(policy=run.__name__):
                responses = iter(['{"source": 9}'] * 10 + ['{"source": 17}'])
                result = run(lambda *args, **kwargs: next(responses), [], Cited, measure=measure,
                             validation_context={"allowed": {17}})
                self.assertEqual(result.source, 17)

    def test_local_repair_keeps_raw_answer_for_custom_validation(self):
        for run in (complete_json, complete_json_via_async):
            with self.subTest(policy=run.__name__):
                seen = []

                def validate(payload, raw):
                    seen.append((payload, raw))
                    if payload["value"] == 2:
                        raise ValueError("Keep the requested source.")
                    return Answer.model_validate(payload)

                responses = iter(['{value: 2}', '{value: 3}'])
                result = run(lambda *args, **kwargs: next(responses), [], Answer,
                             measure=measure, validate=validate)
                self.assertEqual(result.value, 3)
                self.assertEqual(seen, [({"value": 2}, '{value: 2}'), ({"value": 3}, '{value: 3}')])

    def test_invalid_field_type_receives_another_repair_turn(self):
        for run in (complete_json, complete_json_via_async):
            with self.subTest(policy=run.__name__):
                responses = iter(['{"value": {"invalid": true}}', '{"value": 3}'])
                rejected = []
                result = run(lambda *args, **kwargs: next(responses), [], Answer, measure=measure,
                             on_rejection=lambda *row: rejected.append(row))
                self.assertEqual(result.value, 3)
                self.assertEqual(len(rejected), 1)
                self.assertIn("Expected integer", rejected[0][2])

    def test_cancellation_at_reset_stops_before_another_completion(self):
        for run in (complete_json, complete_json_via_async):
            with self.subTest(policy=run.__name__):
                cancel, calls = Event(), []

                def complete(turns, **settings):
                    self.assertIs(settings["cancel"], cancel)
                    calls.append(turns)
                    self.assertLessEqual(len(calls), 9)
                    return '{"value": -1}'

                def reject(attempt, raw, problem):
                    if attempt == 9:
                        cancel.set()

                with self.assertRaises(asyncio.CancelledError):
                    run(complete, [], Answer, measure=measure, cancel=cancel, on_rejection=reject)
                self.assertEqual(len(calls), 9)

    def test_schema_programming_error_is_not_repaired(self):
        class Broken(BaseModel):
            @classmethod
            def model_json_schema(cls, **settings):
                raise ValueError("schema bug")

        for run in (complete_json, complete_json_via_async):
            for settings in ({}, {"validate": lambda payload, raw: Broken()}):
                with self.subTest(policy=run.__name__, custom_validator=bool(settings)), \
                     self.assertRaisesRegex(ValueError, "schema bug"):
                    run(lambda *args, **kwargs: "{}", [], Broken, measure=measure, **settings)


class StructuredTests(unittest.TestCase):
    def test_errors_accumulate_until_a_valid_schema_answer(self):
        responses = iter(["not json", '{"value": -2}', '{"value": 3}'])
        calls, rejected = [], []

        def complete(turns, **settings):
            calls.append(turns)
            return next(responses)

        result = complete_json(complete, [{"role": "user", "content": "Keep source 17."}], Answer,
            measure=measure, on_rejection=lambda *row: rejected.append(row))
        self.assertEqual(result.value, 3)
        self.assertEqual([len(row) for row in calls], [1, 3, 5])
        self.assertIn("minimum of 0", calls[-1][-1]["content"])
        self.assertEqual(calls[-1][1]["content"], "not json")
        self.assertEqual(len(rejected), 2)

    def test_narrative_can_recover_after_more_than_four_repairs(self):
        responses = iter([json.dumps({"narrative": "x" * 6001})] * 6 +
                         [json.dumps({"narrative": "A" * 80})])
        result = complete_json(lambda *args, **kwargs: next(responses), [], NarrativeAnswer, measure=measure)
        self.assertEqual(result.narrative, "A" * 80)

    def test_full_context_stops_before_another_completion(self):
        calls = []

        def complete(turns, **settings):
            calls.append(turns)
            return "invalid"

        with self.assertRaisesRegex(ModelFormatError, "after 1 invalid answers"):
            complete_json(complete, [], Answer, measure=lambda *args, **kwargs:
                          {"input_tokens": 101, "input_budget_tokens": 100})
        self.assertEqual(len(calls), 1)

    def test_prompt_media_repairs_on_the_estimate(self):
        """An image in the prompt costs an estimate, not the whole cycle.

        No route counts image tiles before inference, so measurement reports
        ``measurement_complete=False`` with a per-image allowance. Repair must
        continue on that estimate. Refusing it stopped every cycle that held a
        photo before a reply was ever routed.
        """
        for run in (complete_json, complete_json_via_async):
            with self.subTest(policy=run.__name__):
                calls = []
                responses = iter(['{"value": -1}', '{"value": 3}'])

                def complete(turns, **settings):
                    calls.append(turns)
                    return next(responses)

                result = run(complete, [], Answer, measure=lambda *args, **kwargs: {
                    "input_tokens": 9000, "input_budget_tokens": 100000,
                    "measurement": "estimated.images_by_constant.o200k_base",
                    "measurement_complete": False})
                self.assertEqual(result.value, 3)
                self.assertEqual(len(calls), 2)

    def test_media_estimate_over_budget_names_the_estimate(self):
        """The stop reason keeps the measurement label, so an estimate reads as one."""
        with self.assertRaisesRegex(ModelFormatError, "estimated.images_by_constant"):
            complete_json(lambda *args, **kwargs: "invalid", [], Answer, measure=lambda *args, **kwargs: {
                "input_tokens": 101, "input_budget_tokens": 100,
                "measurement": "estimated.images_by_constant.o200k_base",
                "measurement_complete": False})

    def test_cancellation_and_programming_errors_keep_their_boundary(self):
        cancel = Event()

        def complete(*args, **kwargs):
            cancel.set()
            return "invalid"

        with self.assertRaises(asyncio.CancelledError):
            complete_json(complete, [], Answer, measure=measure, cancel=cancel)
        for error in (KeyError("bug"), TypeError("bug")):
            with self.subTest(error=error), self.assertRaises(type(error)):
                complete_json(lambda *args, **kwargs: (_ for _ in ()).throw(error), [], Answer, measure=measure)

    def test_unexpected_validator_type_error_is_not_repaired(self):
        class Broken(BaseModel):
            @model_validator(mode="after")
            def broken(self):
                raise TypeError("validator bug")

        with self.assertRaisesRegex(TypeError, "validator bug"):
            complete_json(lambda *args, **kwargs: "{}", [], Broken, measure=measure)

    def test_repair_keeps_validation_context(self):
        class Cited(BaseModel):
            source: int

            @model_validator(mode="after")
            def allowed(self, info):
                if self.source not in info.context["allowed"]:
                    raise ValueError("Cite an inspected source.")
                return self

        responses = iter(['{"source": 9}', '{"source": 17}'])
        result = complete_json(lambda *args, **kwargs: next(responses), [], Cited, measure=measure,
                               validation_context={"allowed": {17}})
        self.assertEqual(result.source, 17)

    def test_agent_final_answer_repair_never_restarts_the_agent(self):
        model = SimpleNamespace(base_url="http://127.0.0.1:1/v1", model="fixture", api_key="",
                                context_window=8192, live_test=True)
        result = SimpleNamespace(result_json={"value": -1}, stdout="")
        with patch("pm.llm.stage_model.LazyStageModel.complete", return_value='{"value": 4}') as complete, \
             patch("pm.llm.stage_model.LazyStageModel.measure_context", side_effect=measure):
            self.assertEqual(repair_agent_result(result, Answer, model, instruction="Completed task").value, 4)
        self.assertEqual(complete.call_count, 1)
        self.assertNotIn("tools", complete.call_args.kwargs)


class AsyncStructuredTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_repair_and_bound_cancellation(self):
        cancel, calls = Event(), []

        async def complete(turns, **settings):
            self.assertIs(settings["cancel"], cancel)
            calls.append(turns)
            return '{"value": 2}' if len(calls) == 2 else "invalid"

        complete.measure_context = measure
        self.assertEqual((await complete_json_async(partial(complete, cancel=cancel), [], Answer)).value, 2)
        self.assertEqual(len(calls), 2)
