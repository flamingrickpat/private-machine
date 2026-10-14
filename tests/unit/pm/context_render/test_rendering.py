"""Check record references, typed dispatch, and retained media."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pm.context_render import ContextKind, render_context
from pm.context_render.turns import Media, MediaKind, Text
from workflows.public_speech.public_context import PublicContext


class ContextRenderTests(unittest.TestCase):
    def data(self, content):
        return json.loads(content.text.split("\n", 1)[1])

    def test_only_identical_records_become_original_id_references(self):
        record = {"knoxel_id": 23456, "text": "original", "revision": 4,
                  "scope": {"frame": "in_world"}, "provenance": {"source_kind": "User"}}
        changed = {**record, "text": "correction", "revision": 5}
        source = {"records": [record, copy.deepcopy(record), changed, record]}
        before = copy.deepcopy(source)
        rendered = self.data(render_context(source))
        self.assertEqual(rendered["records"][1], {"record_ref": 23456})
        self.assertEqual(rendered["records"][2]["text"], "correction")
        self.assertEqual(rendered["records"][3]["revision"], 4)
        table = self.data(render_context(source, dedupe=False))["records"]
        self.assertEqual(dict(zip(table["table_fields"], table["table_rows"][1])), record)
        self.assertEqual(source, before)

    def test_repeated_record_media_is_attached_once_and_not_text_encoded(self):
        image = Media(MediaKind.Image, "image/png", data=b"binary")
        record = {"knoxel_id": 9, "text": "sent image", "attachments": [image]}
        content = render_context({"records": [record, record]})
        self.assertEqual(sum(isinstance(part, Media) for part in content.parts), 1)
        self.assertEqual(content.parts[-1].data, b"binary")
        self.assertIn('"record_ref":9', content.parts[0].text)

    def test_field_selection_cannot_leak_excluded_media(self):
        image = Media(MediaKind.Image, "image/png", data=b"private")
        with TemporaryDirectory() as folder:
            schema = Path(folder) / "rules.json"
            schema.write_text('{"version":1,"select_fields":{"evidence":["knoxel_id","text"]}}', encoding="utf-8")
            content = render_context({"evidence": [{"knoxel_id": 9, "text": "caption", "attachments": [image]}]}, schema_path=schema)
            self.assertTrue(all(isinstance(part, Text) for part in content.parts))
            self.assertNotIn("attachment", content.text)

    def test_public_context_dispatch_preserves_meaning_without_storage_fields(self):
        context = PublicContext(companion_name="Spamuel", user_name="Jesse", affect="old affect", goals=["inspect the site"],
            capabilities={"human_senses": [], "emotion_state": {"shame": 0.18}}, revisions={99: 4},
            ontology=[{"id": 99, "term": "trust", "definition": "Expectation of reliability.",
                       "usage_rule": "Preserve attribution.", "timestamp_creation": "old", "metadata": {"storage": True}}])
        before = context.model_dump()
        data = self.data(render_context(context, ContextKind.Facts | ContextKind.Ontology))
        self.assertNotIn("affect", data)
        self.assertNotIn("emotion_state", data["capabilities"])
        self.assertNotIn("revisions", data)
        self.assertEqual(data["goals"], ["inspect the site"])
        self.assertEqual(data["ontology"][0]["definition"], "Expectation of reliability.")
        self.assertNotIn("metadata", data["ontology"][0])
        self.assertEqual(context.model_dump(), before)

    def test_emotions_keep_two_decimals_and_explicit_thresholds_stay_exact(self):
        data = self.data(render_context({"emotion": {"shame": 0.8932695940694135, "anger": 0.1742}, "threshold": 0.1742}))
        self.assertEqual(data["emotion"], {"shame": 0.89, "anger": 0.17})
        self.assertEqual(data["threshold"], 0.1742)

    def test_composed_sections_send_the_shared_format_note_once(self):
        content = render_context({"emotions": {"shame": 0.89}}) + "\n" + render_context({"records": []})
        self.assertEqual(content.text.count("Tables: zip"), 1)

    def test_factual_profile_keeps_measurement_precision(self):
        data = self.data(render_context({"measurement": 0.00123456}, ContextKind.Facts))
        self.assertEqual(data["measurement"], 0.00123456)

    def test_typed_records_keep_content_and_scope_without_embedding_vectors(self):
        from pm.records.base import RecordBase
        record = RecordBase(id=123, content="survey observation", embedding=[0.0001234] * 100)
        before = record.model_dump()
        data = self.data(render_context(record))
        self.assertEqual(data["id"], 123)
        self.assertEqual(data["content"], "survey observation")
        self.assertEqual(data["scope"], before["scope"])
        self.assertNotIn("embedding", data)
        self.assertEqual(record.model_dump(), before)
