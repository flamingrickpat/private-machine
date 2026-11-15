"""A stimulus the cycle materialized itself is still readable at publication.

Real store, no model. The month run of 2026-11-15 ended with ``KeyError: 11999``
because a managed-run event became a new ``WorldEvent`` after the main-cognition
working view froze, and publication read every stimulus from the frozen view
only (B07 Pass 9). This check holds the rule that publication now follows.
"""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pm.memory.store import MemoryStore
from pm.records.cycle_state import PersonaConfig
from pm.records.enums import FeatureType, InterlocusType, SourceKind
from pm.records.memory_records import Feature
from workflows.main_cognition.host import stimulus_record


class StimulusRecordTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"),
                                 checkpoint=Path(directory.name) / "stimulus.pickle")

    def _commit(self, text: str, feature_type: FeatureType, source_kind: SourceKind) -> int:
        """Commit one record and return its identity."""
        batch = self.store.begin_batch(text, new_tick=True, embed=False)
        record = batch.create(Feature(content=text, feature_type=feature_type, source_kind=source_kind,
                                      interlocus=InterlocusType.Public))
        self.store.commit(batch)
        return record.id

    def test_view_member_comes_from_the_view_and_a_later_event_from_the_store(self) -> None:
        dialogue_id = self._commit("Jesse: are you there?", FeatureType.Dialogue, SourceKind.User)
        view = self.store.snapshot()
        event_id = self._commit("Managed run abc123 started", FeatureType.WorldEvent, SourceKind.Workflow)
        self.assertIsNone(view.get_knoxel_by_id(event_id))
        self.assertIs(stimulus_record(view, self.store, dialogue_id), view.get(dialogue_id))
        from_store = stimulus_record(view, self.store, event_id)
        self.assertEqual(from_store.id, event_id)
        self.assertEqual(from_store.content, "Managed run abc123 started")

    def test_a_record_in_neither_place_fails_loudly(self) -> None:
        dialogue_id = self._commit("Jesse: are you there?", FeatureType.Dialogue, SourceKind.User)
        view = self.store.snapshot()
        self.assertEqual(stimulus_record(view, self.store, dialogue_id).id, dialogue_id)
        with self.assertRaisesRegex(RuntimeError, "neither the frozen view nor the store"):
            stimulus_record(view, self.store, 987654)


if __name__ == "__main__":
    unittest.main()
