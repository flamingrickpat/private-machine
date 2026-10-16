"""Native live gates for A18B, with explicit evidence and execution scope.

Calibration calls the real evaluator. Fresh companion generation and saved
review are opt-in because they incur many model calls. Missing configured
services fail an enabled gate. Saved review never establishes runtime behavior.
"""
from collections.abc import Callable
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from uuid import uuid4

from dev.work_items.a18b.runner import load_suite

ROOT = Path(__file__).resolve().parents[2]
_SUITE = load_suite()
SCENARIO_IDS = tuple(scenario.id for scenario in _SUITE.scenarios)
NORMAL_IDS = frozenset(scenario.id for scenario in _SUITE.scenarios if scenario.priority == "normal")


def select_cases(value: str | None, declared_ids: tuple[str, ...]) -> tuple[str, ...]:
    """Parse PM_A18B_CASES without I/O. Reject empty, duplicate, or unknown IDs.

    An absent value selects no cases. An explicitly empty value is an error.
    Invalid input raises ValueError before any selected case can run.
    """
    if value is None:
        return ()
    selected = tuple(part.strip() for part in value.split(","))
    if not all(selected):
        raise ValueError("PM_A18B_CASES must contain nonempty comma-separated scenario IDs.")
    if len(selected) != len(set(selected)):
        raise ValueError("PM_A18B_CASES contains duplicate scenario IDs.")
    unknown = set(selected) - set(declared_ids)
    if unknown:
        raise ValueError("PM_A18B_CASES contains unknown scenario IDs: " + ", ".join(sorted(unknown)))
    return selected


def validate_selected_result(returncode: int, receipt: dict, scenario_id: str,
                             declared_ids: tuple[str, ...]) -> None:
    """Reject failed cases and invalid subset scores with ValueError. Do no I/O.

    Exit 1 is expected because the suite keeps its full denominator.
    Neither that exit code nor a strong case can excuse missing coverage fields.
    """
    if returncode not in (0, 1):
        raise ValueError(f"The selected CLI command failed with exit code {returncode}.")
    if scenario_id not in declared_ids:
        raise ValueError("The requested scenario is not in the manifest.")
    if not isinstance(receipt, dict):
        raise ValueError("The suite score must be a JSON object.")
    if receipt.get("scope") != "fresh runtime execution":
        raise ValueError("The suite score must capture fresh runtime execution.")
    if receipt.get("cadence_policy") != "normal_wait_v1":
        raise ValueError("The selected run must establish normal-by-default cadence.")
    if (type(receipt.get("normal_cases")) is not int or receipt["normal_cases"] != len(NORMAL_IDS)
            or type(receipt.get("strong_normal_cases")) is not int
            or receipt["strong_normal_cases"] != int(scenario_id in NORMAL_IDS)):
        raise ValueError("The selected run must retain its primary-case coverage.")
    if receipt.get("status") != "incomplete" or receipt.get("passed") is not False:
        raise ValueError("A selected case must retain incomplete suite coverage.")
    if (any(type(receipt.get(field)) is not int for field in ("declared_cases", "evaluated_cases", "strong_cases"))
            or receipt.get("declared_cases") != len(declared_ids)
            or receipt.get("evaluated_cases") != 1 or receipt.get("strong_cases") != 1):
        raise ValueError("The suite score must retain the manifest denominator and one strong case.")
    missing = receipt.get("missing_or_incomplete")
    expected_missing = set(declared_ids) - {scenario_id}
    if (not isinstance(missing, list) or len(missing) != len(expected_missing)
            or any(not isinstance(item, str) for item in missing) or set(missing) != expected_missing):
        raise ValueError("The suite score must mark exactly the unselected cases as missing.")
    if receipt.get("critical_failures") != {}:
        raise ValueError("The suite score contains critical failures or lacks that field.")
    if receipt.get("critical_unresolved") != {}:
        raise ValueError("The suite score contains unresolved critical evidence or lacks that field.")
    if receipt.get("manual_review_required") != {}:
        raise ValueError("The suite score requires human inspection or lacks that field.")
    cases = receipt.get("cases")
    if not isinstance(cases, list) or len(cases) != 1 or not isinstance(cases[0], dict):
        raise ValueError("The suite score must contain exactly the requested scored case.")
    case = cases[0]
    if case.get("scenario") != scenario_id:
        raise ValueError("The scored case does not match the requested scenario.")
    if any(case.get(field) is not True for field in ("complete", "strong", "floor")):
        raise ValueError("The selected case must be complete, strong, and above the minimum floor.")
    if case.get("critical_failures") != [] or case.get("critical_unresolved") != [] or case.get("review_errors") != []:
        raise ValueError("The selected case contains critical failures or review errors, or lacks those fields.")
    if case.get("manual_review_required") != [] or case.get("review_protocol_complete") is not True:
        raise ValueError("The selected case requires human inspection or lacks complete reviewer evidence.")


def invoke(command: str, *extra: str) -> tuple[subprocess.CompletedProcess, Path]:
    """Run the documented CLI and retain stdout, stderr and terminal receipts."""
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:8]
    parent = ROOT / ".runtime" / "live" / "a18b" / name
    parent.mkdir(parents=True)
    out = parent / command
    config = os.environ.get("PM_MODEL_CONFIG", str(ROOT / "config.alpha.openrouter.yaml"))
    with (parent / "command.log").open("w", encoding="utf-8") as log:
        result = subprocess.run([sys.executable, "-m", "dev.work_items.a18b", command,
            "--config", config, "--out", str(out), *extra], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    return result, out


class LiveA18BCalibrationTests(unittest.TestCase):
    def test_planted_defects_and_legitimate_controls(self):
        """Require both fresh reviews to match every independent control label."""
        process, out = invoke("calibrate")
        receipt = json.loads((out / "receipt.json").read_text(encoding="utf-8"))
        self.assertEqual(process.returncode, 0, str(out))
        self.assertTrue(receipt["complete"], str(out))
        self.assertTrue(receipt["passed"], str(out))


@unittest.skipUnless(os.environ.get("PM_A18B_FRESH") == "1", "Set PM_A18B_FRESH=1 to generate all 16 real cases.")
class LiveA18BFreshSuiteTests(unittest.TestCase):
    def test_complete_repertoire(self):
        """Apply the declared suite gate after full-runtime conversations."""
        process, out = invoke("run")
        receipt = json.loads((out / "suite-score.json").read_text(encoding="utf-8"))
        self.assertEqual(process.returncode, 0, str(out))
        self.assertTrue(receipt["passed"], str(out))


@unittest.skipUnless(os.environ.get("PM_A18B_SAVED"), "Set PM_A18B_SAVED to a saved A18B corpus for reviewer regression.")
class LiveA18BSavedReviewTests(unittest.TestCase):
    def test_frozen_repertoire(self):
        """Run real judges over saved packets, without executing the companion."""
        process, out = invoke("review-saved", "--source", os.environ["PM_A18B_SAVED"])
        receipt = json.loads((out / "suite-score.json").read_text(encoding="utf-8"))
        self.assertEqual(process.returncode, 0, str(out))
        self.assertTrue(receipt["passed"], str(out))


class LiveA18BSelectedCaseTests(unittest.TestCase):
    """Run only PM_A18B_CASES entries. Full-suite opt-in remains independent."""

    selected_cases: tuple[str, ...] = ()

    @classmethod
    def setUpClass(cls) -> None:
        """Validate the complete selection before any live case starts."""
        cls.selected_cases = select_cases(os.environ.get("PM_A18B_CASES"), SCENARIO_IDS)


def make_case_test(scenario_id: str) -> Callable[[LiveA18BSelectedCaseTests], None]:
    """Create one native test. Each enabled test owns a fresh CLI directory."""
    def test(self: LiveA18BSelectedCaseTests) -> None:
        if scenario_id not in self.selected_cases:
            self.skipTest(f"Add {scenario_id} to PM_A18B_CASES to run this live case.")
        process, out = invoke("run", "--case", scenario_id)
        score_path = out / "suite-score.json"
        self.assertTrue(score_path.is_file(), f"Missing suite score after exit {process.returncode}: {out}")
        receipt = json.loads(score_path.read_text(encoding="utf-8"))
        try:
            validate_selected_result(process.returncode, receipt, scenario_id, SCENARIO_IDS)
        except ValueError as error:
            self.fail(f"{out}: {error}")

    test.__name__ = f"test_{scenario_id}"
    test.__doc__ = f"Require fresh runtime and strong reviews for {scenario_id}."
    return test


for _scenario_id in SCENARIO_IDS:
    setattr(LiveA18BSelectedCaseTests, f"test_{_scenario_id}", make_case_test(_scenario_id))
