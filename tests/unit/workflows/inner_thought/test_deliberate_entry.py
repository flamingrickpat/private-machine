"""Verify deliberate-action admission into the shared private slot through actual stores.

The instruction text is a synthetic storage input standing in for main cognition's
accepted deliberate work. These checks execute no attention model, no dispatch
worker, and no substitute service. They establish that a deliberate action can own
a private inquiry without a nomination, that the single private slot is still
exclusive, and that no judgments are invented for unscored instructions.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest

from pm.memory.snapshot import GhostConflict
from pm.memory.store import MemoryStore
from pm.memory.write_batch import WriteBatch
from pm.records.base import Provenance, RecordLineage, RecordScope
from pm.records.cycle_state import PersonaConfig
from pm.records.enums import ActionType, EntityClass, FeatureType, FrameEnum, InterlocusType, SourceKind
from pm.records.memory_records import ActionSchemaRecord, Entity, Feature
from workflows.cognitive_control.schemas import ControlConfig
from workflows.inner_thought.graph_host import PrivateInquiryOwner, active_inquiry
from workflows.inner_thought.inquiry import (
    InquiryRecord, inquiry_history, inquiry_origin, stage_attention, stage_inquiry_step,
)
from workflows.inner_thought.nominations import capture_inquiry_grant, stage_nominations
from workflows.inner_thought.schemas import (
    DeliberateAdmission, InquiryGrant, InquiryProgress, InquiryStep, ThoughtAttentionDecision, ThoughtJudgments,
    ThoughtNomination,
)


class DeliberateEntryTests(unittest.TestCase):
    """Keep deliberate admission, spontaneous selection, and progress on separate authority lines."""

    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path: Path = Path(directory.name) / "deliberate.pickle"
        self.store: MemoryStore = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"),
            checkpoint=self.path)
        self.now: datetime = datetime(2026, 11, 14, 0, 0)
        self.limits: ControlConfig = ControlConfig()
        self.instruction = "Work out why Jesse stopped describing the unfinished story."
        anchors = self.store.begin_batch("anchors", embed=False)
        self.observer: Entity = anchors.create(Entity(content="Spamuel", aliases=[], entity_class=EntityClass.Self))
        self.partner: Entity = anchors.create(Entity(content="Jesse", aliases=[], entity_class=EntityClass.Human))
        self.source: Feature = anchors.create(Feature(content="I stopped talking about the story.", causal=True,
            feature_type=FeatureType.Dialogue, entity_id=self.partner.id, addressee_entity_id=self.observer.id,
            interlocus=InterlocusType.Public, provenance=Provenance(source_kind=SourceKind.User)))
        self.action: ActionSchemaRecord = anchors.create(ActionSchemaRecord(content=self.instruction,
            action_type=ActionType.Loopback,
            provenance=Provenance(source_kind=SourceKind.Workflow),
            lineage=RecordLineage(episode_id="dispatch:deliberate-story", operation="action_dispatch",
                parent_record_ids=[self.source.id]),
            metadata={"action_id": "deliberate-story", "workflow_name": "inner_thought",
                      "dispatch_status": "admitted", "evidence_ids": [self.source.id]}))
        self.commit(anchors)
        self.grant: InquiryGrant = capture_inquiry_grant(self.store.snapshot(capture=False),
            observer_entity_id=self.observer.id, scope=RecordScope(),
            source_ids=[self.source.id, self.action.id], critical_epoch=4)
        self.judgments = ThoughtJudgments(interestingness=0.5, character_relevance=0.7, plausibility=0.2,
            consequence=0.4, desirability=0.5, action_suitability=0.3, reason="A withdrawn topic needs explaining.")

    def commit(self, batch: WriteBatch) -> None:
        """Persist append-only proposals through actual clean recovery roots."""
        batch.preserve_on_recovery()
        self.store.commit(batch)

    def owner(self) -> PrivateInquiryOwner:
        """Build the private owner that keeps commit authority over the store."""
        return PrivateInquiryOwner(self.store, commit=self.commit, cancel=Event(), limits=self.limits)

    def admission(self, *, action_id: int | None = None, operation: str = "feasibility_check") -> DeliberateAdmission:
        """State the accepted deliberate work exactly as the deliberate path assigned it."""
        return DeliberateAdmission(action_record_id=action_id or self.action.id, instruction=self.instruction,
            evidence_ids=(self.source.id,), operation=operation,
            reason="Main cognition assigned this private work after its own appraisal.")

    def enter(self, *, action_id: int | None = None, operation: str = "feasibility_check",
              commit_id: str = "deliberate-entry") -> InquiryRecord:
        """Commit one deliberate admission through the owner's durable boundary."""
        return self.owner().accept_deliberate_entry(self.admission(action_id=action_id, operation=operation),
            grant=self.grant, commit_id=commit_id, now_utc=self.now)

    def select(self, episode_id: str) -> InquiryRecord:
        """Allocate the same private slot through a spontaneous nomination selection."""
        batch = self.store.begin_batch("nomination-" + episode_id, embed=False)
        staged = stage_nominations(batch, [ThoughtNomination(text="The withdrawn topic may hide a concern.",
            question="Why did the topic disappear?", evidence_ids=(self.source.id,), reality="inferred",
            judgments=self.judgments, suggested_operation="perspective_shift")], grant=self.grant,
            batch_id=episode_id)
        self.commit(batch)
        attention = self.store.begin_batch("attention-" + episode_id, embed=False)
        selected = stage_attention(attention, [staged[0].record.id],
            ThoughtAttentionDecision(nomination_id=staged[0].record.id, disposition="select",
                reason="Inspect this private question."), episode_id=episode_id, now_utc=self.now)
        self.commit(attention)
        assert selected.inquiry is not None
        return selected.inquiry

    def test_deliberate_admission_allocates_the_slot_without_a_nomination(self) -> None:
        entered = self.enter()
        state = entered.receipt.state
        self.assertEqual(state.status, "active")
        self.assertEqual(state.sequence, 0)
        self.assertEqual(state.origin_record_id, self.action.id)
        self.assertEqual(state.node_ids, ())
        self.assertEqual(state.next_operation, "feasibility_check")
        self.assertIsNone(entered.receipt.attention)
        self.assertIsNone(entered.receipt.operation_record_id)
        self.assertEqual(entered.receipt.deliberate, self.admission())
        self.assertEqual(self.store.gwt_turns, {})
        origin = inquiry_origin(self.store.snapshot(capture=False), state)
        self.assertEqual(origin.kind, "deliberate_action")
        self.assertEqual(origin.record_id, self.action.id)
        self.assertEqual(origin.question, self.instruction)
        self.assertEqual(origin.evidence_ids, (self.source.id,))
        self.assertIsNone(origin.judgments)
        self.assertEqual(origin.suggested_operation, "")
        active = active_inquiry(self.store.snapshot(capture=False), observer_entity_id=self.observer.id)
        assert active is not None
        self.assertEqual(active.record.id, entered.record.id)
        restored = MemoryStore.load(self.path)
        history = inquiry_history(restored.snapshot(capture=False), f"deliberate:{self.action.id}")
        self.assertEqual(tuple(item.receipt for item in history), (entered.receipt,))

    def test_a_replayed_dispatch_cannot_allocate_the_same_work_twice(self) -> None:
        self.enter()
        before = self.path.read_bytes()
        with self.assertRaisesRegex(GhostConflict, "already owns a private inquiry"):
            self.enter(commit_id="replayed-dispatch")
        self.assertEqual(self.path.read_bytes(), before)

    def test_deliberate_work_holds_the_slot_against_a_spontaneous_selection(self) -> None:
        self.enter()
        with self.assertRaisesRegex(GhostConflict, "slot"):
            self.select("spontaneous")

    def test_an_unreleased_or_already_answered_action_cannot_enter(self) -> None:
        batch = self.store.begin_batch("other-actions", embed=False)
        deferred = batch.create(ActionSchemaRecord(content=self.instruction, action_type=ActionType.Loopback,
            provenance=Provenance(source_kind=SourceKind.Workflow),
            lineage=RecordLineage(episode_id="dispatch:deferred", operation="action_dispatch",
                parent_record_ids=[self.source.id]),
            metadata={"action_id": "deferred", "workflow_name": "inner_thought", "dispatch_status": "pending",
                      "dispatch_deferred": True}))
        owned = batch.create(ActionSchemaRecord(content=self.instruction, action_type=ActionType.Loopback,
            provenance=Provenance(source_kind=SourceKind.Workflow),
            lineage=RecordLineage(episode_id="dispatch:answered", operation="action_dispatch",
                parent_record_ids=[self.source.id]),
            metadata={"action_id": "answered", "workflow_name": "inner_thought", "dispatch_status": "admitted",
                      "mental_result_id": 91}))
        imagined = batch.create(ActionSchemaRecord(content=self.instruction, action_type=ActionType.Loopback,
            scope=RecordScope(frame=FrameEnum.Meta), provenance=Provenance(source_kind=SourceKind.Workflow),
            lineage=RecordLineage(episode_id="dispatch:imagined", operation="action_dispatch",
                parent_record_ids=[self.source.id]),
            metadata={"action_id": "imagined", "workflow_name": "inner_thought", "dispatch_status": "admitted"}))
        self.commit(batch)
        for record, pattern in [(deferred, "released by main cognition"), (owned, "already owns a mental result"),
                                (imagined, "in its inquiry reality")]:
            with self.assertRaisesRegex(ValueError, pattern):
                self.enter(action_id=record.id, commit_id=f"reject-{record.id}")

    def test_admission_names_its_operation_and_the_first_step_keeps_the_action_identity(self) -> None:
        with self.assertRaisesRegex(KeyError, "unknown thought operation: daydream"):
            self.enter(operation="daydream", commit_id="bad-operation")
        entered = self.enter()
        step = InquiryStep(operation="feasibility_check", thought="The topic faded after a shared setback.",
            evidence_ids=(self.source.id,), assumptions=(), reality="inferred", judgments=self.judgments,
            relations=(), alternatives=(), progress=InquiryProgress(kind="none", inspected_source_ids=(),
                compared_node_ids=(), meaningful_difference=False, reason="Nothing outside was acquired."),
            outcome="continue", reason="Keep examining the withdrawn topic.", information_gap="",
            next_operation="perspective_shift")
        batch = self.store.begin_batch("first-step", embed=False)
        executed = stage_inquiry_step(batch, parent_receipt_id=entered.record.id, step=step, grant=self.grant,
            active_seconds=2, now_utc=self.now, limits=self.limits)
        self.commit(batch)
        self.assertEqual(executed.receipt.state.sequence, 1)
        self.assertEqual(executed.receipt.state.origin_record_id, self.action.id)
        origin = inquiry_origin(self.store.snapshot(capture=False), executed.receipt.state)
        self.assertEqual(origin.kind, "deliberate_action")
        self.assertEqual(origin.record_id, self.action.id)


if __name__ == "__main__":
    unittest.main()
