"""Verify explicit parked admission through actual store and checkpoint boundaries.

The proposals are synthetic storage inputs. These checks execute no attention
model, automatic scheduler, or substitute service.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest

from pm.memory.snapshot import GhostConflict
from pm.memory.store import MemoryStore
from pm.memory.write_batch import WriteBatch
from pm.records.base import Provenance, RecordScope
from pm.records.cycle_state import PersonaConfig
from pm.records.enums import EntityClass, FeatureType, InterlocusType, SourceKind
from pm.records.memory_records import Entity, Feature
from workflows.cognitive_control.schemas import ControlConfig
from workflows.inner_thought.graph_host import PrivateInquiryOwner
from workflows.inner_thought.inquiry import (
    InquiryRecord, inquiry_history, stage_attention, stage_inquiry_resume, stage_inquiry_step,
)
from workflows.inner_thought.nominations import capture_inquiry_grant, extend_inquiry_grant, stage_nominations
from workflows.inner_thought.schemas import (
    InquiryGrant, InquiryProgress, InquiryResumeDecision, InquiryStep, ThoughtAttentionDecision,
    ThoughtJudgments, ThoughtNomination, ThoughtRelation,
)


class InquiryResumeTests(unittest.TestCase):
    """Keep admission distinct from executed progress and preserve finite bounds."""

    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path: Path = Path(directory.name) / "resume.pickle"
        self.store: MemoryStore = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"), checkpoint=self.path)
        self.now: datetime = datetime(2026, 11, 13, 0, 0)
        self.limits: ControlConfig = ControlConfig()
        anchors = self.store.begin_batch("anchors", embed=False)
        self.observer: Entity = anchors.create(Entity(content="Spamuel", aliases=[], entity_class=EntityClass.Self))
        self.partner: Entity = anchors.create(Entity(content="Jesse", aliases=[], entity_class=EntityClass.Human))
        self.source: Feature = anchors.create(Feature(content="The sketch remains unfinished.", causal=True,
            feature_type=FeatureType.Dialogue, entity_id=self.partner.id, addressee_entity_id=self.observer.id,
            interlocus=InterlocusType.Public, provenance=Provenance(source_kind=SourceKind.User)))
        self.commit(anchors)
        self.grant: InquiryGrant = capture_inquiry_grant(self.store.snapshot(capture=False),
            observer_entity_id=self.observer.id, scope=RecordScope(), source_ids=[self.source.id],
            governing_ids=[self.source.id], critical_epoch=2)
        self.judgments: ThoughtJudgments = ThoughtJudgments(interestingness=0.6, character_relevance=0.6,
            plausibility=0.3, consequence=0.4, desirability=0.4, action_suitability=0.2,
            reason="An unfinished sketch leaves a question, not an observed motive.")

    def commit(self, batch: WriteBatch) -> None:
        """Persist append-only proposals through actual clean recovery roots."""
        batch.preserve_on_recovery()
        self.store.commit(batch)

    def select(self, episode_id: str) -> InquiryRecord:
        """Allocate the existing private slot through its actual storage owner."""
        batch = self.store.begin_batch("nomination-" + episode_id, embed=False)
        records = stage_nominations(batch, [ThoughtNomination(text="The sketch could reflect several concerns.",
            question="What remains unknown?", evidence_ids=(self.source.id,), reality="inferred",
            judgments=self.judgments, suggested_operation="future_projection")], grant=self.grant, batch_id=episode_id)
        self.commit(batch)
        attention = self.store.begin_batch("attention-" + episode_id, embed=False)
        selected = stage_attention(attention, [records[0].record.id],
            ThoughtAttentionDecision(nomination_id=records[0].record.id, disposition="select", reason="Inspect this private question."),
            episode_id=episode_id, now_utc=self.now)
        self.commit(attention)
        assert selected.inquiry is not None
        return selected.inquiry

    def parked(self, *, limits: ControlConfig | None = None, active_seconds: float = 1) -> InquiryRecord:
        """Pause a real accepted generative operation without claiming a conclusion."""
        selected = self.select("sketch")
        step = InquiryStep(operation="future_projection", thought="Several motives remain possible.",
            evidence_ids=(self.source.id,), assumptions=(), reality="inferred", judgments=self.judgments,
            relations=(), alternatives=(), progress=InquiryProgress(kind="none", inspected_source_ids=(),
                compared_node_ids=(), meaningful_difference=False, reason="No new outside evidence."),
            outcome="pause", reason="Leave this question until reconsideration.", information_gap="Jesse's motive.", next_operation="")
        batch = self.store.begin_batch("park", embed=False)
        accepted = stage_inquiry_step(batch, parent_receipt_id=selected.record.id, step=step,
            grant=self.grant, active_seconds=active_seconds, now_utc=self.now, limits=limits or self.limits)
        self.commit(batch)
        return accepted

    def stage_resume(self, parent: InquiryRecord, *, grant: InquiryGrant | None = None,
                     trigger_ids: tuple[int, ...] = (), now_utc: datetime | None = None,
                     limits: ControlConfig | None = None, commit_id: str = "resume") -> tuple[WriteBatch, InquiryRecord]:
        """Stage the explicit decision without committing it or inventing a thought."""
        batch = self.store.begin_batch(commit_id, embed=False)
        accepted = stage_inquiry_resume(batch, parent_receipt_id=parent.record.id,
            decision=InquiryResumeDecision(operation="feasibility_check", reason="Reconsider the retained question.",
                trigger_source_ids=trigger_ids), grant=grant or parent.receipt.state.grant,
            now_utc=now_utc or self.now + timedelta(seconds=self.limits.parked_seconds), limits=limits or self.limits)
        return batch, accepted

    def report(self, *, source_kind: SourceKind = SourceKind.User,
               feature_type: FeatureType = FeatureType.Dialogue) -> Feature:
        """Commit an explicit new source with its real stored provenance."""
        batch = self.store.begin_batch("new-source-" + str(len(self.store.all_features)), embed=False)
        report = batch.create(Feature(content="I returned to the sketch because the colours finally made sense.",
            causal=True, feature_type=feature_type, entity_id=self.partner.id, addressee_entity_id=self.observer.id,
            interlocus=InterlocusType.Public, provenance=Provenance(source_kind=source_kind)))
        self.commit(batch)
        return report

    def test_elapsed_deadline_readmits_without_nodes_or_progress(self) -> None:
        parked = self.parked()
        before = parked.receipt.state
        owner = PrivateInquiryOwner(self.store, commit=self.commit, cancel=Event(), limits=self.limits)
        resumed = owner.accept_resume(parked,
            InquiryResumeDecision(operation="feasibility_check", reason="Reconsider the retained question after parking.",
                                  trigger_source_ids=()), grant=before.grant, commit_id="resume",
            now_utc=self.now + timedelta(seconds=self.limits.parked_seconds))
        restored = MemoryStore.load(self.path)
        history = inquiry_history(restored.snapshot(capture=False), "sketch")
        self.assertEqual(tuple(item.receipt.state.sequence for item in history), (0, 1, 1))
        self.assertEqual(history[-1].receipt, resumed.receipt)
        self.assertEqual(resumed.receipt.state.node_ids, before.node_ids)
        self.assertEqual(resumed.receipt.state.pending_generation_ids, before.pending_generation_ids)
        self.assertEqual(resumed.receipt.state.active_elapsed_seconds, before.active_elapsed_seconds)
        self.assertEqual(resumed.receipt.state.inspected_root_ids, before.inspected_root_ids)
        self.assertEqual(resumed.receipt.state.grant, before.grant)
        self.assertEqual(resumed.receipt.state.stalled_steps, 0)
        self.assertEqual(resumed.receipt.state.status, "active")
        self.assertIsNone(resumed.receipt.operation_record_id)
        self.assertIsNone(resumed.receipt.state.resume_after_utc)
        self.assertEqual(resumed.receipt.state.resume_source_ids, ())
        self.assertEqual(restored.gwt_turns, {})

    def test_new_original_report_allows_early_admission_without_claimed_acquisition(self) -> None:
        parked = self.parked()
        report = self.report()
        grant = extend_inquiry_grant(self.store.snapshot(capture=False), parked.receipt.state.grant, [report.id])
        batch, resumed = self.stage_resume(parked, grant=grant, trigger_ids=(report.id,), now_utc=self.now)
        self.commit(batch)
        self.assertEqual(resumed.receipt.state.inspected_root_ids, parked.receipt.state.inspected_root_ids)
        self.assertEqual(resumed.receipt.state.root_source_ids, parked.receipt.state.root_source_ids)
        self.assertIn(report.id, {source.record_id for source in resumed.receipt.state.grant.sources})
        self.assertEqual(resumed.receipt.resume.trigger_source_ids if resumed.receipt.resume is not None else (), (report.id,))

    def test_early_admission_without_new_evidence_is_rejected(self) -> None:
        parked = self.parked()
        before = self.path.read_bytes()
        with self.assertRaises(GhostConflict):
            self.stage_resume(parked, now_utc=self.now)
        self.assertEqual(self.path.read_bytes(), before)

    def test_old_source_and_generated_spoof_cannot_supply_new_outside_evidence(self) -> None:
        parked = self.parked()
        with self.assertRaises(ValueError):
            self.stage_resume(parked, trigger_ids=(self.source.id,), now_utc=self.now)
        for kind, feature in [(SourceKind.Self, FeatureType.Dialogue), (SourceKind.Workflow, FeatureType.Thought),
                              (SourceKind.System, FeatureType.StateUpdate), (SourceKind.Agent, FeatureType.InnerMonologue)]:
            source = self.report(source_kind=kind, feature_type=feature)
            grant = extend_inquiry_grant(self.store.snapshot(capture=False), parked.receipt.state.grant, [source.id])
            with self.assertRaises(ValueError):
                self.stage_resume(parked, grant=grant, trigger_ids=(source.id,), now_utc=self.now)

    def test_step_and_active_time_bounds_are_never_reset(self) -> None:
        parked = self.parked()
        with self.assertRaises(GhostConflict):
            self.stage_resume(parked, limits=ControlConfig(episode_steps=1))
        with self.assertRaises(GhostConflict):
            self.stage_resume(parked, limits=ControlConfig(episode_seconds=1))

    def test_critical_epoch_cannot_be_rebased_on_resume(self) -> None:
        parked = self.parked()
        changed = parked.receipt.state.grant.model_copy(update={"critical_epoch": 3})
        with self.assertRaises(GhostConflict):
            self.stage_resume(parked, grant=changed)

    def test_original_fingerprints_remain_guards_through_resume_commit(self) -> None:
        parked = self.parked()
        batch, _ = self.stage_resume(parked)
        corrected = self.store.begin_batch("correct-original", embed=False)
        source = corrected.view().get(self.source.id).copy_for_edit()
        source.content = "The sketch is finished."
        corrected.replace(source)
        self.store.commit(corrected)
        with self.assertRaises(GhostConflict):
            self.commit(batch)

    def test_competing_resumes_commit_only_one_exact_parent(self) -> None:
        parked = self.parked()
        first, _ = self.stage_resume(parked, commit_id="first")
        second, _ = self.stage_resume(parked, commit_id="second")
        self.commit(first)
        with self.assertRaises(GhostConflict):
            self.commit(second)
        with self.assertRaises(GhostConflict):
            self.stage_resume(parked, commit_id="stale-parent")

    def test_other_active_inquiry_keeps_the_private_slot(self) -> None:
        parked = self.parked()
        self.select("different-question")
        with self.assertRaises(GhostConflict):
            self.stage_resume(parked)

    def test_next_operation_uses_same_graph_after_resume_and_clean_restart(self) -> None:
        parked = self.parked()
        batch, resumed = self.stage_resume(parked)
        self.commit(batch)
        grant = extend_inquiry_grant(self.store.snapshot(capture=False), resumed.receipt.state.grant,
                                    resumed.receipt.state.node_ids)
        node_id = resumed.receipt.state.node_ids[0]
        step = InquiryStep(operation="feasibility_check", thought="The motive still requires an outside answer.",
            evidence_ids=(self.source.id,), assumptions=(), reality="inferred", judgments=self.judgments,
            relations=(ThoughtRelation(target_record_id=node_id, kind="extends", reason="Revisit the retained uncertainty."),),
            alternatives=(), progress=InquiryProgress(kind="none", inspected_source_ids=(), compared_node_ids=(),
                meaningful_difference=False, reason="Admission alone acquired no evidence."), outcome="information_gap",
            reason="Stop with the missing evidence explicit.", information_gap="Jesse's actual motive.", next_operation="")
        executed = self.store.begin_batch("after-resume", embed=False)
        accepted = stage_inquiry_step(executed, parent_receipt_id=resumed.record.id, step=step, grant=grant,
            active_seconds=2, now_utc=self.now, limits=self.limits)
        self.commit(executed)
        restored = MemoryStore.load(self.path)
        history = inquiry_history(restored.snapshot(capture=False), "sketch")
        self.assertEqual(tuple(item.receipt.state.sequence for item in history), (0, 1, 1, 2))
        self.assertEqual(history[-1].receipt, accepted.receipt)
        self.assertEqual(accepted.receipt.state.active_elapsed_seconds, 3)
        self.assertEqual(accepted.receipt.state.origin_record_id, parked.receipt.state.origin_record_id)


if __name__ == "__main__":
    unittest.main()
