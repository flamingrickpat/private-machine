"""Chat log import: CSV checks, exchange grouping, and the marks of imported records."""
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from pm.app.chat_import import (add_transfer_records, group_exchanges, import_exchanges, read_chat_csv,
                                synthesize_times)
from pm.constants import IMPORT_SESSION_GAP_SECONDS, TRANSFER_EVENT_SOURCE
from pm.memory.store import MemoryStore
from pm.records.cycle_state import PersonaConfig
from pm.records.enums import FeatureType, NarrativeTypes, SourceKind
from pm.records.import_marks import ImportOrigin, import_mark
from pm.records.memory_records import Narrative, SemanticRecord
from pm.records.mental_state import PersonalityProfile


class CsvTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "chat.csv"

    def tearDown(self):
        self.temp.cleanup()

    def read(self, text: str, *, bom: bool = False):
        self.path.write_bytes((b"\xef\xbb\xbf" if bom else b"") + text.encode("utf-8"))
        return read_chat_csv(self.path, companion_name="Ava", user_name="Bob")

    def test_bom_and_multi_line_messages(self):
        rows = self.read('name,message,timestamp\nbob,"first line\nsecond line",2026-05-01 20:10\n'
                         'AVA,hi,2026-05-01T20:11:30Z\n', bom=True)
        self.assertEqual([row.speaker for row in rows], ["user", "companion"])
        self.assertEqual(rows[0].message, "first line\nsecond line")
        self.assertEqual(rows[1].timestamp, datetime(2026, 5, 1, 20, 11, 30))

    def test_offsets_become_naive_utc(self):
        rows = self.read("name,message,timestamp\nBob,hi,2026-05-01T22:10:00+02:00\n")
        self.assertEqual(rows[0].timestamp, datetime(2026, 5, 1, 20, 10))

    def test_missing_timestamps_are_synthesized_before_now(self):
        rows = self.read("name,message\nBob,hi\nAva,hello\nBob,how are you\n")
        now = datetime(2026, 10, 1, 12, 0)
        timed = synthesize_times(rows, now)
        self.assertEqual([row.timestamp for row in timed], sorted(row.timestamp for row in timed))
        self.assertEqual((now - timed[-1].timestamp).total_seconds(), IMPORT_SESSION_GAP_SECONDS)
        self.assertEqual((timed[1].timestamp - timed[0].timestamp).total_seconds(), 120)

    def test_unknown_name_names_the_row(self):
        with self.assertRaises(ValueError) as caught:
            self.read("name,message\nBob,hi\nCarol,hey\n")
        self.assertIn("Row 2", str(caught.exception))

    def test_decreasing_time_fails(self):
        with self.assertRaises(ValueError) as caught:
            self.read("name,message,timestamp\nBob,hi,2026-05-01 20:10\nAva,hey,2026-05-01 20:09\n")
        self.assertIn("Row 2", str(caught.exception))

    def test_partial_timestamps_fail(self):
        with self.assertRaises(ValueError):
            self.read("name,message,timestamp\nBob,hi,2026-05-01 20:10\nAva,hey,\n")

    def test_bad_timestamp_fails(self):
        with self.assertRaises(ValueError):
            self.read("name,message,timestamp\nBob,hi,01.05.2026\n")

    def test_grouping_puts_replies_after_their_messages(self):
        rows = self.read("name,message\nAva,hello first\nBob,a\nBob,b\nAva,c\nAva,d\nBob,e\n")
        groups = group_exchanges(rows)
        self.assertEqual([([r.message for r in g.user_rows], [r.message for r in g.companion_rows]) for g in groups],
                         [([], ["hello first"]), (["a", "b"], ["c", "d"]), (["e"], [])])


def _store() -> MemoryStore:
    return MemoryStore(PersonaConfig(user_name="Bob", companion_name="Ava"))


class ImportMarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        path = Path(self.temp.name) / "chat.csv"
        path.write_text("name,message,timestamp\nBob,hi there,2026-05-01 20:10\nAva,hello you,2026-05-01 20:11\n"
                        "Bob,how was your day,2026-05-01 20:12\nAva,long,2026-05-01 20:13\n", encoding="utf-8")
        self.rows = read_chat_csv(path, companion_name="Ava", user_name="Bob")
        self.now = datetime(2026, 10, 1, 12, 0)

    def tearDown(self):
        self.temp.cleanup()

    def test_records_look_lived_and_carry_the_mark(self):
        store = _store()
        profile = PersonalityProfile(openness=0.9)
        result = import_exchanges(store, group_exchanges(self.rows), import_id="import-x", source_sha256="abc",
                                  personality=profile, imported_at=self.now)
        self.assertEqual(result.ticks, [1, 2])
        self.assertEqual(store.current_tick_id, 2)
        self.assertEqual([state.tick_id for state in store.states], [1, 2])
        state = store.states[0]
        self.assertEqual(state.mental_state.state_personality.openness, 0.9)
        self.assertEqual(state.mental_state.state_core.valence, 0.0)
        self.assertEqual(state.decisions, [])
        user_id = store.get_entity_id("Bob")
        dialogue = sorted(store.all_features, key=lambda item: item.id)
        self.assertEqual(len(dialogue), 4)
        user, reply = dialogue[0], dialogue[1]
        self.assertEqual(user.provenance.source_kind, SourceKind.User)
        self.assertEqual(user.entity_id, user_id)
        self.assertEqual(user.metadata["exchange_status"], "resolved")
        self.assertEqual(user.metadata["public_reply_id"], reply.id)
        self.assertEqual(reply.provenance.source_kind, SourceKind.Self)
        self.assertEqual(reply.addressee_entity_id, user_id)
        self.assertEqual(reply.metadata["publication_status"], "published")
        self.assertTrue(all(item.causal and item.timestamp_world_begin.year == 2026 for item in dialogue))
        marks = [import_mark(item) for item in dialogue]
        self.assertEqual([mark.source_row for mark in marks], [1, 2, 3, 4])
        self.assertTrue(all(mark.origin == ImportOrigin.ChatLog and not mark.synthetic for mark in marks))
        turn = store.gwt_turns[1]
        self.assertEqual(turn.stimulus_ids, [user.id])
        self.assertEqual(turn.knoxel_ids, [user.id, reply.id])
        self.assertEqual(state.stimulus_groups[0].stimulus_ids, [user.id])

    def test_transfer_records_are_synthetic(self):
        store = _store()
        import_exchanges(store, group_exchanges(self.rows), import_id="import-x", source_sha256="abc",
                         personality=PersonalityProfile(), imported_at=self.now)
        companion = store.get_entity_id("Ava")
        batch = store.begin_batch("seed-narrative")
        batch.create(Narrative(content="Ava is sharp.", narrative_type=NarrativeTypes.SelfImage, entity_id=companion,
                               last_refined_with_tick=2))
        store.commit(batch)
        ids = add_transfer_records(store, import_id="import-x", imported_at=self.now)
        records = [store.get_knoxel_by_id(key) for key in ids]
        self.assertEqual(len(records), 3)
        event, semantic, narrative = records
        self.assertEqual(event.feature_type, FeatureType.WorldEvent)
        self.assertEqual(event.override_source, TRANSFER_EVENT_SOURCE)
        self.assertIn("2026-10-01", event.content)
        self.assertIsInstance(semantic, SemanticRecord)
        self.assertEqual(semantic.subject_entity_id, companion)
        self.assertEqual(semantic.source_knoxel_ids, [event.id])
        self.assertEqual(semantic.semantic_record_id, semantic.id)
        self.assertTrue(narrative.content.startswith("Ava is sharp."))
        self.assertIn(event.id, narrative.lineage.source_event_ids)
        self.assertTrue(all(import_mark(item).synthetic and import_mark(item).origin == ImportOrigin.Transfer
                            for item in records))


if __name__ == "__main__":
    unittest.main()
