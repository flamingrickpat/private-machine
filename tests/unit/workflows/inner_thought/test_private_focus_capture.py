"""Check private focus selection against actual accepted advice records.

These deterministic inputs exercise frozen visibility and source authority.
They do not replace a model or certify generated attention decisions.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pm.memory.store import MemoryStore
from pm.records.base import Provenance
from pm.records.cycle_state import PersonaConfig
from pm.records.enums import FeatureType, InterlocusType, SourceKind
from pm.records.memory_records import Feature
from workflows.cognitive_control.attention import (
    PrivateFocusChanged, captured_private_focus, validate_private_focus,
)
from workflows.cognitive_control.schemas import FocusState


class PrivateFocusCaptureTests(unittest.TestCase):
    """Keep the selected focus tied to its frozen record and actual expiry."""

    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store: MemoryStore = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"),
            checkpoint=Path(directory.name) / "focus.pickle")
        self.now: datetime = datetime(2026, 11, 12, 18)

    def advice(self, instruction: str, *, seconds: float = 60) -> Feature:
        """Append an explicit accepted advice input through the real store."""
        batch = self.store.begin_batch(f"focus:{instruction}", embed=False)
        record = batch.create(Feature(content=instruction, feature_type=FeatureType.StateUpdate,
            interlocus=InterlocusType.PrivateInternal, provenance=Provenance(source_kind=SourceKind.Self),
            metadata={"control_kind": "attention_focus",
                "focus": FocusState(accept=True, instruction=instruction, strength=0.4).model_dump(mode="json"),
                "expires_at": (self.now + timedelta(seconds=seconds)).isoformat()}))
        self.store.commit(batch)
        return record

    def test_snapshot_never_adopts_later_advice_and_preserves_operator_cap(self) -> None:
        first = self.advice("Examine the unfinished sketch")
        frozen = self.store.snapshot(capture=False)
        self.advice("Revisit another concern")
        captured = captured_private_focus(frozen, self.now, maximum_strength=0.25)
        self.assertEqual(captured.source_id, first.id)
        assert captured.focus is not None
        self.assertEqual(captured.focus.instruction, "Examine the unfinished sketch")
        self.assertEqual(captured.focus.strength, 0.25)
        with self.store.lock, self.assertRaises(PrivateFocusChanged):
            validate_private_focus(self.store.snapshot(capture=False), captured, now_utc=self.now)

    def test_expiry_invalidates_current_selection_without_a_record_mutation(self) -> None:
        self.advice("Examine this concern", seconds=10)
        captured = captured_private_focus(self.store.snapshot(capture=False), self.now)
        with self.store.lock, self.assertRaises(PrivateFocusChanged):
            validate_private_focus(self.store.snapshot(capture=False), captured,
                now_utc=self.now + timedelta(seconds=10))

    def test_already_expired_advice_remains_absent_and_has_no_future_deadline(self) -> None:
        first = self.advice("Earlier advice", seconds=-10)
        captured = captured_private_focus(self.store.snapshot(capture=False), self.now)
        self.assertIsNone(captured.focus)
        self.assertEqual(captured.source_id, first.id)
        self.assertIsNone(captured.expires_at)
        validate_private_focus(self.store.snapshot(capture=False), captured,
            now_utc=self.now + timedelta(seconds=10))

    def test_new_advice_invalidates_a_selection_that_observed_no_advice(self) -> None:
        captured = captured_private_focus(self.store.snapshot(capture=False), self.now)
        self.advice("A new accepted focus")
        with self.store.lock, self.assertRaises(PrivateFocusChanged):
            validate_private_focus(self.store.snapshot(capture=False), captured, now_utc=self.now)
