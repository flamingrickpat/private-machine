"""Budget arithmetic and real worker ownership need no model substitute."""
import asyncio
from threading import Barrier, Event
import unittest

from pm.llm.concurrency import ordered_parallel_map
from pm.llm.reasoning import ReasoningBudget
from pm.llm.reasoning import ReasoningOptions
from pm.llm.http_client import ModelError
from pm.llm.activation import provider_generation
from pm.llm.config import resolve_connection


class ReasoningTests(unittest.TestCase):
    def test_budget_uses_remaining_output_after_answer_reserve(self):
        budget = ReasoningBudget(fraction=0.1, ceiling_tokens=8192, input_margin_fraction=0)
        result = budget.generation(context_window=10000, input_tokens=5000, max_output_tokens=8000)
        self.assertEqual(result["reasoning_tokens"], 295)

    def test_provider_output_limit_and_ceiling_both_apply(self):
        budget = ReasoningBudget(fraction=0.1, ceiling_tokens=512, input_margin_fraction=0)
        self.assertEqual(budget.generation(context_window=1000000, input_tokens=1000,
            max_output_tokens=131072)["reasoning_tokens"], 512)
        self.assertEqual(budget.generation(context_window=1000000, input_tokens=1000,
            max_output_tokens=4096)["reasoning_tokens"], 204)

    def test_low_headroom_fails_without_truncating_input(self):
        with self.assertRaises(ModelError):
            ReasoningBudget(fraction=0.1).generation(context_window=4096, input_tokens=3000, max_output_tokens=4096)

    def test_disabled_budget_and_hosted_translation(self):
        self.assertEqual(ReasoningBudget(fraction=0).generation(context_window=1, input_tokens=1,
            max_output_tokens=1), {"enable_thinking": False})
        connection = resolve_connection(base_url="https://openrouter.ai/api/v1", model="qwen", api_key="")
        result = provider_generation(connection, {"enable_thinking": True, "reasoning_tokens": 512})
        self.assertEqual(result, {"reasoning": {"enabled": True, "max_tokens": 512},
                                  "provider": {"require_parameters": True}})

    def test_all_reasoning_controls_reach_each_adapter(self):
        hosted = resolve_connection(base_url="https://openrouter.ai/api/v1", model="qwen", api_key="")
        local = resolve_connection(base_url="http://127.0.0.1:8080/v1", model="qwen", api_key="")
        cases = [
            (ReasoningOptions(enabled=False), {"chat_template_kwargs": {"enable_thinking": False}}),
            (ReasoningOptions(enabled=True, max_tokens=768),
             {"chat_template_kwargs": {"enable_thinking": True}, "reasoning_budget_tokens": 768}),
            (ReasoningOptions(effort="low"), {"reasoning_effort": "low"}),
            (ReasoningOptions(exclude=True), {"reasoning": {"exclude": True}}),
        ]
        for options, expected_local in cases:
            with self.subTest(options=options):
                generation = {"reasoning": options, "max_tokens": 4096}
                self.assertEqual(provider_generation(hosted, generation), {
                    "reasoning": options.model_dump(exclude_none=True), "max_tokens": 4096,
                    "provider": {"require_parameters": True}})
                self.assertEqual(provider_generation(local, generation), {**expected_local, "max_tokens": 4096})
                self.assertIs(generation["reasoning"], options)

    def test_effort_levels_and_defaults_remain_explicit(self):
        for effort in ("none", "minimal", "low", "medium", "high", "xhigh", "max"):
            self.assertEqual(ReasoningOptions(effort=effort).effort, effort)
        connection = resolve_connection(base_url="https://openrouter.ai/api/v1", model="qwen", api_key="")
        self.assertEqual(provider_generation(connection, {"temperature": 0.2}), {"temperature": 0.2})

    def test_contradictory_controls_fail_instead_of_losing_a_setting(self):
        for controls in ({"max_tokens": 512, "effort": "low"},
                         {"enabled": False, "max_tokens": 512},
                         {"enabled": True, "effort": "none"},
                         {"max_tokens": True}, {"max_tokens": 0}):
            with self.subTest(controls=controls), self.assertRaises(ValueError):
                ReasoningOptions.model_validate(controls)

    def test_local_budget_and_legacy_effort_translate_for_openrouter(self):
        connection = resolve_connection(base_url="https://openrouter.ai/api/v1", model="qwen", api_key="")
        self.assertEqual(provider_generation(connection, {"reasoning_budget_tokens": 768})["reasoning"], {"max_tokens": 768})
        self.assertEqual(provider_generation(connection, {"reasoning_effort": "low"})["reasoning"], {"effort": "low"})


class ParallelTests(unittest.TestCase):
    def test_real_calls_overlap_but_results_keep_input_order(self):
        barrier, second_finished, cancel = Barrier(2), Event(), Event()

        def operation(index):
            barrier.wait(timeout=3)
            if index == 0:
                if not second_finished.wait(3):
                    raise TimeoutError("The second worker never ran.")
            else:
                second_finished.set()
            return index

        self.assertEqual(ordered_parallel_map(operation, [0, 1], workers=2, cancel=cancel), [0, 1])
        self.assertFalse(cancel.is_set())

    def test_failure_cancels_and_joins_sibling_before_return(self):
        started, closed, cancel = Event(), Event(), Event()

        def operation(index):
            if index == 0:
                if not started.wait(3):
                    raise TimeoutError("The sibling never started.")
                raise ValueError("original worker failure")
            started.set()
            try:
                if not cancel.wait(3):
                    raise TimeoutError("The sibling was not cancelled.")
                raise asyncio.CancelledError
            finally:
                closed.set()

        with self.assertRaisesRegex(ValueError, "original worker failure"):
            ordered_parallel_map(operation, [0, 1], workers=2, cancel=cancel)
        self.assertTrue(closed.is_set())

    def test_cancelled_work_never_enters_callback(self):
        cancel = Event()
        cancel.set()
        with self.assertRaises(asyncio.CancelledError):
            ordered_parallel_map(lambda item: self.fail("Cancelled work started."), [1], workers=2, cancel=cancel)
