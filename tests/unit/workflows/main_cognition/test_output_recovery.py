"""Replay a captured syntax failure through the pure stage-output boundary.

These checks establish recovery and cancellation, not model quality.
"""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from pm.context_render.turns import Role, Turn
from pm.memory.bash_view import MemoryBashEnvironment
from workflows.inner_thought.schemas import ThoughtResult
from workflows.main_cognition.model_json import ModelFormatError, complete_json

CAPTURE = Path(__file__).resolve().parents[4] / "test_data/captures/alpha/cognitive-format-recovery.json"


class StageOutputRecoveryTests(unittest.TestCase):
    def test_granted_completion_retains_its_models_measurement(self):
        from workflows.main_cognition.host import build_main_cognition_tools

        class Model:
            def __init__(self):
                self.calls = 0
                self.measures = 0

            def complete(self, turns, **generation):
                self.calls += 1
                return "invalid" if self.calls == 1 else '{"thought": "A quiet moment.", "evidence_ids": []}'

            def measure_context(self, turns, **generation):
                self.measures += 1
                return {"input_tokens": 100, "input_budget_tokens": 1000}

        model = Model()
        tools = build_main_cognition_tools(None, None, model.complete)(self.environment, None)
        context = SimpleNamespace(runtime=SimpleNamespace(environment=self.environment,
                                  tools={"llm_complete": tools["llm_complete"]}))
        result = complete_json(context, self.turns, ThoughtResult)
        self.assertEqual(result.thought, "A quiet moment.")
        self.assertEqual(model.measures, 1)

    def setUp(self):
        self.raw = json.loads(CAPTURE.read_text(encoding="utf-8"))["malformed_thought"]["text"]
        self.environment = MemoryBashEnvironment({}, {}, "read")
        self.turns = [Turn(Role.System, "Captured format boundary."), Turn(Role.User, "Keep source IDs.")]

    def context(self, tool):
        return SimpleNamespace(runtime=SimpleNamespace(environment=self.environment, tools={"llm_complete": tool, "llm_measure_context":
            lambda turns, **settings: {"input_tokens": len(turns), "input_budget_tokens": 8}}))

    def test_captured_unescaped_quote_recovers_without_changing_evidence(self):
        quote = '"World\'s Okayest Employee."'
        escaped = self.raw.replace(quote, quote.replace('"', '\\"'))
        replies = iter((self.raw, escaped))
        calls = []

        def replay(turns, **generation):
            calls.append(turns)
            return next(replies)

        result = complete_json(self.context(replay), self.turns, ThoughtResult)
        self.assertEqual(result.evidence_ids, [11323, 11374, 11387])
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1][:2], self.turns)
        rejected = json.loads(self.environment.read_text("scratch/model-output-rejection-01.json"))
        self.assertEqual(rejected["raw"], self.raw)
        self.assertIn("invalid JSON", rejected["error"])

    def test_repeated_corruption_stops_at_context_budget_and_keeps_rejections(self):
        calls = []

        def replay(turns, **generation):
            calls.append(turns)
            return self.raw

        with self.assertRaises(ModelFormatError):
            complete_json(self.context(replay), self.turns, ThoughtResult)
        self.assertEqual(len(calls), 4)
        self.assertEqual([len(turns) for turns in calls], [2, 4, 6, 8])
        self.assertEqual(json.loads(self.environment.read_text("scratch/model-output-rejection-04.json"))["attempt"], 4)

    def test_cancellation_and_tool_errors_do_not_become_format_retries(self):
        for error in (asyncio.CancelledError(), TypeError("broken completion tool")):
            with self.subTest(error=type(error).__name__):
                calls = []

                def fail(turns, **generation):
                    calls.append(turns)
                    raise error

                with self.assertRaises(type(error)) as raised:
                    complete_json(self.context(fail), self.turns, ThoughtResult)
                self.assertIs(raised.exception, error)
                self.assertEqual(len(calls), 1)
