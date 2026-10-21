"""Alt text: written by the model before embedding, stored without a revision change, used by memory.

A scripted completion stands in for the multimodal model. The store, the
checkpoint, the embedding cache, and the reconsolidation renderer are real.
"""
from __future__ import annotations

import asyncio
import struct
import tempfile
import unittest
import zlib
from datetime import datetime
from pathlib import Path

from pm.llm.config import EmbeddingConfig
from pm.llm.http_client import ModelError
from pm.llm.prompt_encoding import normalize_turns
from pm.context_render.turns import Media, Role
from pm.memory.embeddings import EmbeddingService, embedding_text
from pm.memory.media_alt import MediaLibrary, complete_alt_text
from pm.memory.reconsolidation.pipeline import reconsolidate
from pm.memory.reconsolidation.story import feature_line
from pm.memory.store import MemoryStore
from pm.records.cycle_state import PersonaConfig
from pm.records.enums import FeatureType, InterlocusType
from pm.records.media import attach_images, image_from_bytes, install_media_lookup, record_media, with_tags
from pm.records.memory_records import Feature, new_feature


def png() -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"\x00\x80")) + chunk(b"IEND", b""))


class ScriptedVision:
    """Answer alt-text calls from a script. Any other prompt is a test failure."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def __call__(self, turns, **generation):
        turns = normalize_turns(turns)
        self.calls.append((turns, generation))
        if not any(isinstance(part, Media) for turn in turns if turn.role is Role.User for part in turn.content.parts):
            raise AssertionError("Only the alt-text call may reach this model.")
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


class AltTextTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.store = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"),
                                 checkpoint=self.root / "ghost.pickle")
        self.image = image_from_bytes(png())
        metadata: dict = {}
        tags = attach_images(metadata, [self.image])
        self.feature = new_feature(self.store, with_tags("what is this?", tags), FeatureType.Dialogue,
                                   source="Jesse", interlocus=InterlocusType.Public, metadata=metadata)
        self.addCleanup(install_media_lookup, None)

    def test_the_model_sees_the_image_and_invalid_output_and_provider_errors_are_retried(self):
        model = ScriptedVision("", ModelError("HTTP 429"), "A grey cat sits on a sofa.")
        revision = self.store.revisions[self.feature.id]
        written = complete_alt_text(self.store, [self.image.id, self.image.id], model)
        self.assertEqual(written, {self.image.id: "A grey cat sits on a sofa."})
        self.assertEqual(len(model.calls), 3)
        first, generation = model.calls[0]
        image_part = [part for part in first[-1].content.parts if isinstance(part, Media)][0]
        self.assertEqual(image_part.url, self.image.data_url())
        self.assertIs(generation["enable_thinking"], False)
        self.assertEqual(model.calls[1][0][-1].role, Role.User)
        self.assertIn("cannot be used", model.calls[1][0][-1].content.text)
        stored = record_media(self.store.get_knoxel_by_id(self.feature.id)).images[0]
        self.assertEqual((stored.alt_text, stored.alt_text_source), ("A grey cat sits on a sofa.", "model"))
        # The description is derived data: the revision stays, the checkpoint keeps it.
        self.assertEqual(self.store.revisions[self.feature.id], revision)
        reopened = MemoryStore.from_checkpoint(self.root / "ghost.pickle")
        self.assertEqual(reopened.find_image(self.image.id).alt_text, "A grey cat sits on a sofa.")
        # A second call finds the alt text and asks nothing.
        complete_alt_text(self.store, [self.image.id], model)
        self.assertEqual(len(model.calls), 3)

    def test_a_stale_replacement_keeps_the_alt_text(self):
        batch = self.store.begin_batch("stale")
        record = batch.view().get(self.feature.id).copy_for_edit()
        complete_alt_text(self.store, [self.image.id], ScriptedVision("A grey cat."))
        record.metadata = {**record.metadata, "exchange_status": "resolved"}
        batch.replace(record)
        self.store.commit(batch)
        after = self.store.get_knoxel_by_id(self.feature.id)
        self.assertEqual(after.metadata["exchange_status"], "resolved")
        self.assertEqual(record_media(after).images[0].alt_text, "A grey cat.")

    def test_alt_text_is_written_before_the_embedding_and_the_embedding_uses_it(self):
        model = ScriptedVision("A grey cat.")
        install_media_lookup(MediaLibrary(self.store, model))
        config = EmbeddingConfig(base_url="http://127.0.0.1:1/v1", model="test-embed", api_key="", dimensions=4,
                                 legacy_cache_dir=self.root / "legacy", cache_dir=self.root / "cache")
        service = EmbeddingService(config, live_test=True)
        expected = "what is this?\n[image: A grey cat.]"
        from pm.llm.http_client import ModelError
        self.assertEqual(service.inspect_index({self.feature.id: self.feature}).missing, [self.feature.id])
        with self.assertRaises(ModelError):
            asyncio.run(service.record_embedding(self.feature))
        self.assertGreater(service.client.request_count, 0)
        self.assertEqual(self.feature.embedding, [])
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(embedding_text(self.feature.content), expected)
        self.assertNotIn(self.image.data_b64, embedding_text(self.feature.content))

    def test_reconsolidation_reads_images_as_alt_text_and_completes_them_first(self):
        stored = self.store.get_knoxel_by_id(self.feature.id)
        self.assertIn("[image: no description yet]", feature_line(stored, "%H:%M", self.store.find_image))
        model = ScriptedVision("A grey cat.")
        report = reconsolidate(self.store, complete=model, embeddings=None, now=stored.timestamp_world_begin)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(report.created_ids, [])
        line = feature_line(self.store.get_knoxel_by_id(self.feature.id), "%H:%M", self.store.find_image)
        self.assertTrue(line.endswith('Jesse says: "what is this?\n[image: A grey cat.]"'))
        self.assertNotIn("[image:img-", line)


class ScreenshotRecordTests(unittest.TestCase):
    def test_a_tool_screenshot_on_a_world_event_resolves_like_an_upload(self):
        store = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"))
        shot = image_from_bytes(png(), provenance={"source": "tool", "server": "pm-minecraft",
                                                   "tool": "minecraft_screenshot", "run_id": "r1"})
        metadata: dict = {}
        content = with_tags("The body looks around.", attach_images(metadata, [shot]))
        event = store.store_knoxel(Feature(content=content, feature_type=FeatureType.WorldEvent, metadata=metadata,
                                           timestamp_world_begin=datetime(2026, 1, 1)))
        found = store.find_image(shot.id)
        self.assertEqual((found.provenance.source, found.provenance.tool), ("tool", "minecraft_screenshot"))
        self.assertEqual(record_media(store.get_knoxel_by_id(event.id)).images[0].data(), png())


if __name__ == "__main__":
    unittest.main()
