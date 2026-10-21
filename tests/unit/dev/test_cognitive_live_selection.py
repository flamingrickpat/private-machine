"""Check native case selection and score validation without model calls.

Constructed scores test the live harness only. They do not establish model
accuracy or companion quality. Test discovery does not run live cases.
"""

from pm.records.copying import pickle_copy
import unittest

from dev.work_items.a18b.policy import score_suite
from dev.work_items.a18b.runner import load_suite
from tests.live import test_cognitive_evaluation as live_selection


class CognitiveLiveSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ids = live_selection.SCENARIO_IDS
        self.first, self.second = self.ids[:2]

    def test_absent_selection_disables_all_cases(self):
        self.assertEqual(live_selection.select_cases(None, self.ids), ())

    def test_selection_keeps_requested_ids_and_strips_spaces(self):
        self.assertEqual(live_selection.select_cases(f" {self.second}, {self.first} ", self.ids),
                         (self.second, self.first))

    def test_all_manifest_ids_are_valid_without_a_fixed_case_count(self):
        self.assertEqual(live_selection.select_cases(",".join(self.ids), self.ids), self.ids)

    def test_empty_ids_fail_explicit_validation(self):
        for value in ("", " ", ",", f"{self.first},", f",{self.first}", f"{self.first}, ,{self.second}"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "PM_A18B_CASES.*nonempty"):
                live_selection.select_cases(value, self.ids)

    def test_duplicate_ids_fail_explicit_validation(self):
        with self.assertRaisesRegex(ValueError, "PM_A18B_CASES.*duplicate"):
            live_selection.select_cases(f"{self.first}, {self.first} ", self.ids)

    def test_unknown_ids_fail_even_beside_a_valid_id(self):
        for value in ("unknown_case", f"{self.first},unknown_case", self.first.upper(), "*"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "PM_A18B_CASES.*unknown"):
                live_selection.select_cases(value, self.ids)

    def test_native_methods_match_the_actual_manifest(self):
        suite = load_suite()
        self.assertEqual(self.ids, tuple(scenario.id for scenario in suite.scenarios))
        names = unittest.TestLoader().getTestCaseNames(live_selection.LiveA18BSelectedCaseTests)
        self.assertEqual(set(names), {f"test_{scenario.id}" for scenario in suite.scenarios})
        for name in names:
            with self.subTest(name=name):
                self.assertEqual(getattr(live_selection.LiveA18BSelectedCaseTests, name).__name__, name)

    def test_aggregate_and_saved_classes_keep_their_original_tests(self):
        loader = unittest.TestLoader()
        self.assertEqual(loader.getTestCaseNames(live_selection.LiveA18BFreshSuiteTests), ["test_complete_repertoire"])
        self.assertEqual(loader.getTestCaseNames(live_selection.LiveA18BSavedReviewTests), ["test_frozen_repertoire"])


class CognitiveSelectedResultTests(unittest.TestCase):
    def setUp(self) -> None:
        suite = load_suite()
        self.ids = tuple(scenario.id for scenario in suite.scenarios)
        self.scenario = suite.scenarios[0]
        self.case_id = self.scenario.id
        criteria = suite.common_criteria + self.scenario.criteria
        case = {"scenario": self.case_id, "complete": True, "passed_criteria": len(criteria),
                "total_criteria": len(criteria), "fraction": "1", "criteria": {item.id: "pass" for item in criteria},
                "critical_failures": [], "critical_unresolved": [], "manual_review_required": [],
                "review_protocol_complete": True, "strong": True, "floor": True, "review_errors": []}
        self.receipt = {**score_suite(suite, [case]), "scope": "fresh runtime execution"}

    def validate(self, receipt: dict, returncode: int = 1) -> None:
        """Apply the harness check to a constructed suite score, without I/O."""
        live_selection.validate_selected_result(returncode, receipt, self.case_id, self.ids)

    def test_expected_subset_exit_one_passes_only_the_selected_case(self):
        self.validate(self.receipt)
        self.assertFalse(self.receipt["passed"])
        self.assertEqual(self.receipt["status"], "incomplete")
        self.assertEqual(self.receipt["declared_cases"], len(self.ids))
        self.assertEqual(self.receipt["evaluated_cases"], 1)

    def test_exit_zero_still_requires_a_valid_selected_score(self):
        self.validate(self.receipt, returncode=0)
        with self.assertRaises(ValueError):
            self.validate({}, returncode=0)

    def test_arbitrary_nonzero_exit_never_passes(self):
        for returncode in (-1, 2, 3, 130, 0xC0000005):
            with self.subTest(returncode=returncode), self.assertRaisesRegex(ValueError, "exit code"):
                self.validate(self.receipt, returncode=returncode)

    def test_absent_or_malformed_suite_scores_never_pass(self):
        for receipt in ({}, None, [], "not a score"):
            with self.subTest(receipt=receipt), self.assertRaises(ValueError):
                live_selection.validate_selected_result(1, receipt, self.case_id, self.ids)
        for field in self.receipt:
            receipt = pickle_copy(self.receipt)
            del receipt[field]
            if field in ("minecraft_cases", "human_review"):
                continue
            with self.subTest(missing=field), self.assertRaises(ValueError):
                self.validate(receipt)

    def test_saved_review_or_missing_scope_never_establishes_runtime(self):
        for scope in ("reviewer regression on frozen evidence; not runtime regression", "", None):
            with self.subTest(scope=scope), self.assertRaisesRegex(ValueError, "fresh runtime"):
                self.validate({**self.receipt, "scope": scope})

    def test_complete_or_passing_suite_status_cannot_excuse_exit_one(self):
        for fields in ({"status": "pass"}, {"status": "fail"}, {"passed": True}, {"passed": None}):
            with self.subTest(fields=fields), self.assertRaisesRegex(ValueError, "incomplete suite"):
                self.validate({**self.receipt, **fields})

    def test_subset_keeps_full_denominator_and_one_strong_case(self):
        for field, value in (("declared_cases", 1), ("evaluated_cases", 0), ("evaluated_cases", 2),
                             ("strong_cases", 0), ("strong_cases", 2), ("strong_cases", True)):
            with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, "manifest denominator"):
                self.validate({**self.receipt, field: value})

    def test_missing_coverage_must_match_exactly_the_unselected_ids(self):
        missing = self.receipt["missing_or_incomplete"]
        for value in ([], missing[:-1], [*missing, missing[0]], [*missing, self.case_id],
                      [*missing[:-1], "unknown_case"], "not a list", [None] * len(missing)):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "unselected cases"):
                self.validate({**self.receipt, "missing_or_incomplete": value})

    def test_score_must_contain_exactly_the_requested_case(self):
        case = self.receipt["cases"][0]
        for cases in ([], [case, case], [{**case, "scenario": self.ids[1]}], [None], {}):
            with self.subTest(cases=cases), self.assertRaisesRegex(ValueError, "requested"):
                self.validate({**self.receipt, "cases": cases})
        with self.assertRaisesRegex(ValueError, "not in the manifest"):
            live_selection.validate_selected_result(1, self.receipt, "unknown_case", self.ids)

    def test_incomplete_weak_or_below_floor_case_never_passes(self):
        for field in ("complete", "strong", "floor"):
            for value in (False, None, 1, "true"):
                receipt = pickle_copy(self.receipt)
                receipt["cases"][0][field] = value
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, "complete, strong"):
                    self.validate(receipt)

    def test_critical_failures_and_review_errors_never_pass(self):
        for field in ("critical_failures", "review_errors"):
            receipt = pickle_copy(self.receipt)
            receipt["cases"][0][field] = ["Constructed harness rejection input."]
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "critical failures or review errors"):
                self.validate(receipt)
        with self.assertRaisesRegex(ValueError, "suite score contains critical"):
            self.validate({**self.receipt, "critical_failures": {self.case_id: ["external_results"]}})

    def test_missing_case_fields_never_pass(self):
        for field in ("scenario", "complete", "strong", "floor", "critical_failures", "review_errors"):
            receipt = pickle_copy(self.receipt)
            del receipt["cases"][0][field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate(receipt)
