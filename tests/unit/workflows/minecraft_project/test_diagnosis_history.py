"""Read the instruction revision from the body gateway's retained diagnosis."""
import json
from pathlib import Path
import unittest

from workflows.minecraft_project.planning.context_contracts import Diagnosis


class DiagnosisHistoryTests(unittest.TestCase):
    def test_captured_gateway_diagnosis_keeps_its_instruction_revision(self) -> None:
        fixture = Path(__file__).resolve().parents[4] / "test_data/captures/alpha/a13-prefixed-diagnosis.json"
        row = json.loads(fixture.read_bytes())["diagnosis"]
        recorded_revision = row["revision"]
        self.assertEqual(Diagnosis.model_validate(row).revision, recorded_revision)

        del row["revision"]
        self.assertIsNone(Diagnosis.model_validate(row).revision)
