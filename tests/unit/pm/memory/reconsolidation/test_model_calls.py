"""Validated model calls: a repair turn for invalid output, a repeat for provider errors."""
import unittest

from pm.llm.http_client import ModelError
from pm.memory.reconsolidation.model_calls import ModelCalls, ReconsolidationOutputError
from pm.memory.reconsolidation.schemas import FactsAnswer, SummaryAnswer


class Script:
    """Return or raise the scripted answers in order. Keep every request."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.requests = []

    def __call__(self, turns, **generation):
        self.requests.append(turns)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


class ModelCallTests(unittest.TestCase):
    def test_invalid_output_gets_a_repair_turn(self):
        script = Script('{"facts": [{"text": "Bob says that he likes tea.", "lines": [9]}]}',
                        '{"facts": [{"text": "Bob says that he likes tea.", "lines": [1]}]}')
        calls = ModelCalls(script, workers=1)
        answer = calls.json("system", "task", FactsAnswer, context={"lines": 2})
        self.assertEqual(answer.facts[0].lines, [1])
        self.assertEqual(calls.calls, 2)
        repair = script.requests[1]
        self.assertEqual([turn["role"] for turn in repair], ["system", "user", "assistant", "user"])
        self.assertIn("do not exist", repair[-1]["content"])

    def test_provider_errors_repeat_the_call(self):
        script = Script(ModelError("cut off"), '{"summary": "A long enough summary of the scene."}')
        calls = ModelCalls(script, workers=1)
        self.assertEqual(calls.json("system", "task", SummaryAnswer).summary, "A long enough summary of the scene.")
        self.assertEqual(script.requests[0], script.requests[1])

    def test_exhausted_repairs_raise(self):
        calls = ModelCalls(Script(*["no json"] * 5), workers=1)
        with self.assertRaises(ReconsolidationOutputError):
            calls.json("system", "task", SummaryAnswer)

    def test_other_errors_propagate(self):
        with self.assertRaises(KeyError):
            ModelCalls(Script(KeyError("bug")), workers=1).json("system", "task", SummaryAnswer)

    def test_map_keeps_the_input_order(self):
        self.assertEqual(ModelCalls(Script(), workers=3).map(lambda value: value * 2, [3, 1, 2]), [6, 2, 4])


if __name__ == "__main__":
    unittest.main()
