"""Initialize both public examples and import their CSVs without inference."""
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from pm.app.character_cli import init_home, import_chat
from pm.app.character_home import CharacterHome
from pm.records.enums import FeatureType
from pm.records.import_marks import import_mark, ImportOrigin


class ExampleCharacterTests(unittest.TestCase):
    def test_empty_and_imported_examples_use_separate_homes(self):
        with TemporaryDirectory() as directory:
            homes = []
            for name in ("sunny", "rowan"):
                with self.subTest(character=name):
                    source = Path("examples/characters") / name
                    home = CharacterHome(Path(directory) / name)
                    init_home(home, companion_name=name.title(), user_name="Alex", card=source / "card.md",
                              personality=source / "personality.json")
                    self.assertEqual(home.open_store().states, [])
                    embeddings = SimpleNamespace(embed=AsyncMock(return_value=[1.0]), dimensions=8,
                                                 config=SimpleNamespace(model="test-embeddings"),
                                                 smoke_test=AsyncMock(return_value={}),
                                                 prepare_index=AsyncMock(return_value=None))
                    models = (SimpleNamespace(request_count=0), embeddings)
                    with patch("pm.app.character_cli.build_models", return_value=models), \
                         patch("pm.app.character_cli._print"):
                        receipt = import_chat(home, source / "chat.csv", "unused", do_reconsolidate=False,
                                              now=datetime(2026, 10, 20))
                    self.assertEqual(receipt.status, "completed")
                    self.assertEqual(receipt.rows_imported, 12)
                    store = home.open_store()
                    dialogue = [item for item in store.all_features if item.feature_type == FeatureType.Dialogue]
                    self.assertEqual(len(dialogue), 12)
                    self.assertTrue(all(import_mark(item).origin == ImportOrigin.ChatLog for item in dialogue))
                    self.assertEqual(store.get_companion_name(), name.title())
                    homes.append(home)
            self.assertNotEqual(homes[0].ghost_json, homes[1].ghost_json)
            self.assertEqual(homes[0].open_store().get_companion_name(), "Sunny")
