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
        self.assertIn("greater than or equal to 0", calls[-1][-1]["content"])
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
