"""Character home folder: personality validation, init, refusal to overwrite, and loading."""
import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from pm.app.character_cli import init_home
from pm.app.character_home import CharacterHome, parse_personality
from pm.records.import_marks import ImportOrigin, import_mark
from pm.records.mental_state import PersonalityProfile


class PersonalityTests(unittest.TestCase):
    def test_missing_keys_keep_the_neutral_default(self):
        profile = parse_personality({"openness": 0.9})
        self.assertEqual(profile.openness, 0.9)
        self.assertEqual(profile.neuroticism, 0.5)

    def test_unknown_key_names_the_allowed_keys(self):
        with self.assertRaises(ValueError) as caught:
            parse_personality({"openess": 0.9})
        self.assertIn("openess", str(caught.exception))
        self.assertIn("openness", str(caught.exception))

    def test_value_outside_the_range_fails(self):
        with self.assertRaises(ValidationError):
            parse_personality({"openness": 1.5})


class HomeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.card = self.root / "card.md"
        self.card.write_text("{companion_name} teases {user_name}.", encoding="utf-8")
        self.traits = self.root / "personality.json"
        self.traits.write_text(json.dumps({"humor_playfulness": 0.8}), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def init(self, home: CharacterHome, **options):
        init_home(home, companion_name="Ava", user_name="Bob", card=self.card, personality=self.traits, **options)

    def test_init_creates_the_layout_and_an_empty_history(self):
        home = CharacterHome(self.root / "home")
        self.init(home)
        for path in (home.character_yaml, home.personality_json, home.ghost_json, home.runtime_root, home.import_dir):
            self.assertTrue(path.exists(), path)
        persona = home.load_persona()
        self.assertEqual(persona.character_card_story.strip(), "Ava teases Bob.")
        self.assertEqual(PersonalityProfile.init_from_vector(persona.personality_profile).humor_playfulness, 0.8)
        store = home.open_store()
        self.assertEqual(store.states, [])
        self.assertEqual(sorted(entity.content for entity in store.all_entities), ["Ava", "Bob"])
        self.assertTrue(all(import_mark(entity).origin == ImportOrigin.CharacterInit for entity in store.all_entities))
        self.assertEqual(store.checkpoint, home.ghost_json)

    def test_init_refuses_an_existing_home_without_force(self):
        home = CharacterHome(self.root / "home")
        self.init(home)
        with self.assertRaises(FileExistsError):
            self.init(home)
        self.init(home, force=True)
        self.assertTrue(home.ghost_json.exists())

    def test_open_store_needs_init(self):
        with self.assertRaises(FileNotFoundError) as caught:
            CharacterHome(self.root / "missing").open_store()
        self.assertIn("character init", str(caught.exception))

    def test_open_store_applies_the_persona_files(self):
        home = CharacterHome(self.root / "home")
        self.init(home)
        home.personality_json.write_text(json.dumps({"openness": 0.1}), encoding="utf-8")
        store = home.open_store()
        self.assertEqual(PersonalityProfile.init_from_vector(store.ghost_config.personality_profile).openness, 0.1)

    def test_a_renamed_speaker_fails(self):
        home = CharacterHome(self.root / "home")
        self.init(home)
        text = home.character_yaml.read_text(encoding="utf-8").replace("user_name: Bob", "user_name: Rob")
        home.character_yaml.write_text(text, encoding="utf-8")
        with self.assertRaises(ValueError):
            home.open_store()


if __name__ == "__main__":
    unittest.main()
