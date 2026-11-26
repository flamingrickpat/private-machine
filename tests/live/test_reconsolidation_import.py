"""Interrupt and reopen a disposable import on the local chat and embedding services.

PM_TEST_MODEL_CONFIG can explicitly select a different model configuration.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import unittest
from uuid import uuid4

from pm.app.character_cli import ImportedSourceReceipt, import_chat, init_home
from pm.app.character_home import CharacterHome
from pm.records.enums import FeatureType
from pm.utils.operating_system import OperatingSystem
from tests.live.configuration import live_model_config


class LiveReconsolidationImportTests(unittest.TestCase):
    def test_resume_after_durable_source_import(self) -> None:
        root = Path(__file__).resolve().parents[2]
        output = root / ".runtime" / "g02-memory" / (
            "import-resume-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:8])
        output.mkdir(parents=True)
        print(f"Import resume evidence in {output}", flush=True)
        config_path = OperatingSystem.environment.get("PM_TEST_MODEL_CONFIG", str(root / "config.example.yaml"))
        config = live_model_config(Path(config_path), output)
        home = CharacterHome(output / "character")
        card = output / "card.md"
        card.write_text("Ava restores bicycles in her workshop.", encoding="utf-8")
        csv = output / "chat.csv"
        csv.write_text("name,message,timestamp\nBob,Please keep the red folding bicycle for me.,2026-11-18T07:00:00\n"
                       "Ava,I'll put your red bicycle in the workshop.,2026-11-18T07:02:00\n", encoding="utf-8")
        init_home(home, companion_name="Ava", user_name="Bob", card=card)
        now = datetime(2026, 11, 18, 10)

        # A genuine configuration failure occurs after raw import and before inference.
        with self.assertRaises(FileNotFoundError):
            import_chat(home, csv, str(output / "missing-config.yaml"), now=now)
        partial = home.open_store()
        markers = [ImportedSourceReceipt.model_validate(value) for key, value in partial.consolidations.items()
                   if key.startswith("import_sources:")]
        self.assertEqual(len(markers), 1)
        original = markers[0].receipt
        self.assertEqual(len(original.imported_feature_ids), 2)
        self.assertTrue(all(key in partial.all_knoxels for key in original.imported_feature_ids))
        self.assertFalse(any(key.startswith("chat_import:") for key in partial.consolidations))

        receipt = import_chat(home, csv, str(config), now=now)
        self.assertEqual(receipt.status, "completed")
        self.assertEqual(receipt.import_id, original.import_id)
        self.assertEqual(receipt.imported_feature_ids, original.imported_feature_ids)
        self.assertGreater(receipt.model_requests, 0)
        assert receipt.reconsolidation is not None
        self.assertTrue(receipt.reconsolidation.topical_ids)
        reopened = home.open_store()
        self.assertIn("chat_import:" + original.import_id, reopened.consolidations)
        dialogue = [item.id for item in reopened.all_features if item.feature_type == FeatureType.Dialogue]
        self.assertEqual(sorted(dialogue), sorted(original.imported_feature_ids))
        for record_id in [*receipt.reconsolidation.created_ids, *receipt.transfer_ids]:
            record = reopened.get_knoxel_by_id(record_id)
            self.assertIsNotNone(record)
            assert record is not None
            self.assertTrue(record.embedding)
        with self.assertRaisesRegex(ValueError, "already imported"):
            import_chat(home, csv, str(config), now=now)
