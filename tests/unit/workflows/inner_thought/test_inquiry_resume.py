"""Verify explicit parked admission through actual store and checkpoint boundaries.

The proposals are synthetic storage inputs. These checks execute no attention
model, automatic scheduler, or substitute service. They cover what shared
attention may be offered, the durable opportunity identity behind a re-admission,
and the rule that retained parked work never holds the private slot.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest

from pm import vclock
from pm.memory.snapshot import GhostConflict
from pm.memory.store import MemoryStore
from pm.memory.write_batch import WriteBatch
from pm.records.base import Provenance, RecordScope
from pm.records.cycle_state import PersonaConfig
from pm.records.enums import EntityClass, FeatureType, InterlocusType, SourceKind
from pm.records.memory_records import Entity, Feature
from workflows.cognitive_control.attention import CapturedPrivateFocus, PrivateFocusChanged, captured_private_focus
from workflows.cognitive_control.schemas import ControlConfig, FocusState
from workflows.inner_thought.graph_execution import parked_candidate
from workflows.inner_thought.graph_host import PrivateInquiryOwner, active_inquiry, parked_inquiries
from workflows.inner_thought.inquiry import (
    InquiryRecord, inquiry_history, parked_offer, stage_attention, stage_inquiry_resume, stage_inquiry_step,
)
from workflows.inner_thought.nominations import capture_inquiry_grant, extend_inquiry_grant, stage_nominations
from workflows.inner_thought.schemas import (
    InquiryGrant, InquiryProgress, InquiryResumeDecision, InquiryStep, ParkedInquiryOffer, ThoughtAttentionDecision,
    ThoughtAttentionReceipt, ThoughtJudgments, ThoughtNomination, ThoughtRelation,
)


class InquiryResumeTests(unittest.TestCase):
    """Keep admission distinct from executed progress and preserve finite bounds."""

    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path: Path = Path(directory.name) / "resume.pickle"
        self.store: MemoryStore = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"), checkpoint=self.path)
        self.now: datetime = datetime(2026, 11, 13, 0, 0)
        previous_clock = vclock.active()
        self.addCleanup(vclock.install, previous_clock)
        self.clock: vclock.VirtualClock = vclock.VirtualClock(self.now)
        vclock.install(self.clock)
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

    def parked(self, *, limits: ControlConfig | None = None, active_seconds: float = 1,
               episode_id: str = "sketch") -> InquiryRecord:
        """Pause a real accepted generative operation without claiming a conclusion."""
        selected = self.select(episode_id)
        step = InquiryStep(operation="future_projection", thought="Several motives remain possible.",
            evidence_ids=(self.source.id,), assumptions=(), reality="inferred", judgments=self.judgments,
            relations=(), alternatives=(), progress=InquiryProgress(kind="none", inspected_source_ids=(),
                compared_node_ids=(), meaningful_difference=False, reason="No new outside evidence."),
            outcome="pause", reason="Leave this question until reconsideration.", information_gap="Jesse's motive.", next_operation="")
        batch = self.store.begin_batch("park:" + episode_id, embed=False)
        accepted = stage_inquiry_step(batch, parent_receipt_id=selected.record.id, step=step,
            grant=self.grant, active_seconds=active_seconds, now_utc=self.now, limits=limits or self.limits)
        self.commit(batch)
        return accepted

    def admit(self, parent: InquiryRecord, operation: str, reason: str) -> ThoughtAttentionDecision:
        """State the shared attention choice that puts this parked inquiry back in the slot."""
        return ThoughtAttentionDecision(parked_receipt_id=parent.record.id, disposition="select",
            admitted_operation=operation, reason=reason)

    def stage_resume(self, parent: InquiryRecord, *, grant: InquiryGrant | None = None,
                     trigger_ids: tuple[int, ...] = (), now_utc: datetime | None = None,
                     limits: ControlConfig | None = None, commit_id: str = "resume") -> tuple[WriteBatch, InquiryRecord]:
        """Stage the shared attention admission without committing it or inventing a thought."""
        reason = "Reconsider the retained question."
        decision = InquiryResumeDecision(operation="feasibility_check", reason=reason,
            trigger_source_ids=trigger_ids)
        batch = self.store.begin_batch(commit_id, embed=False)
        accepted = stage_inquiry_resume(batch, parent_receipt_id=parent.record.id, decision=decision,
            attention=self.admit(parent, decision.operation, reason),
            grant=grant or parent.receipt.state.grant,
            now_utc=now_utc or self.now + timedelta(seconds=self.limits.parked_seconds), limits=limits or self.limits)
        return batch, accepted

    def owner(self) -> PrivateInquiryOwner:
        """Build the private owner that keeps commit authority over the store."""
        return PrivateInquiryOwner(self.store, commit=self.commit, cancel=Event(), limits=self.limits)

    def offer(self, parked: InquiryRecord, *, grant: InquiryGrant | None = None,
              now_utc: datetime | None = None, limits: ControlConfig | None = None) -> ParkedInquiryOffer | None:
        """Ask what shared attention may be offered for this retained work right now."""
        deadline = self.now + timedelta(seconds=self.limits.parked_seconds)
        return parked_offer(self.store.snapshot(capture=False), parked,
            grant=grant or parked.receipt.state.grant, now_utc=now_utc or deadline, limits=limits or self.limits)

    def admit_opportunity(self, parked: InquiryRecord, *, grant: InquiryGrant | None = None,
                          batch_id: str = "opportunity-one", focus: CapturedPrivateFocus | None = None) -> InquiryRecord:
        """Commit one shared attention decision that re-admits parked work and spends an opportunity."""
        reason = "Revisit the retained question now that its parked deadline passed."
        decision = InquiryResumeDecision(operation="feasibility_check", reason=reason, trigger_source_ids=())
        attention = ThoughtAttentionDecision(parked_receipt_id=parked.record.id, disposition="select",
            admitted_operation=decision.operation, reason=reason)
        deadline = self.now + timedelta(seconds=self.limits.parked_seconds)
        if focus is None:
            focus = captured_private_focus(self.store.snapshot(capture=False), self.now)
        return self.owner().admit_parked_inquiry(parked, decision, attention=attention, nomination_ids=(),
            offered_grant=self.grant, grant=grant or parked.receipt.state.grant, batch_id=batch_id,
            now_utc=deadline, focus=focus)

    def attention_outcomes(self, batch_id: str) -> list[ThoughtAttentionReceipt]:
        """Read the retained attention outcomes recorded for one opportunity identity."""
        view = self.store.snapshot(capture=False)
        return [ThoughtAttentionReceipt.model_validate(record.metadata["thought_attention"])
                for record in view.all_knoxels.values()
                if record.metadata.get("thought_attention_batch") == batch_id]

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
        reason = "Reconsider the retained question after parking."
        resumed = owner.accept_resume(parked,
            InquiryResumeDecision(operation="feasibility_check", reason=reason, trigger_source_ids=()),
            attention=self.admit(parked, "feasibility_check", reason), grant=before.grant, commit_id="resume",
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

    def test_parked_work_is_offered_only_on_its_deadline_or_new_outside_evidence(self) -> None:
        parked = self.parked()
        self.assertIsNone(self.offer(parked, now_utc=self.now))
        late = self.offer(parked)
        assert late is not None
        self.assertEqual(late.receipt_id, parked.record.id)
        self.assertTrue(late.deadline_elapsed)
        self.assertEqual(late.available_evidence_ids, ())
        self.assertEqual(late.question, "What remains unknown?")
        self.assertEqual(late.stop_reason, "Leave this question until reconsideration.")
        report = self.report()
        grant = extend_inquiry_grant(self.store.snapshot(capture=False), parked.receipt.state.grant, [report.id])
        early = self.offer(parked, grant=grant, now_utc=self.now)
        assert early is not None
        self.assertFalse(early.deadline_elapsed)
        self.assertEqual(early.available_evidence_ids, (report.id,))
        self.assertIsNone(self.offer(parked, limits=ControlConfig(episode_steps=1)))
        self.assertIsNone(self.offer(parked, grant=parked.receipt.state.grant.model_copy(update={"critical_epoch": 3})))

    def test_attention_admission_spends_one_opportunity_and_survives_restart(self) -> None:
        parked = self.parked()
        before = parked.receipt.state
        resumed = self.admit_opportunity(parked)
        self.assertEqual(resumed.receipt.state.status, "active")
        self.assertEqual(resumed.receipt.state.sequence, before.sequence)
        self.assertEqual(resumed.receipt.state.grant, before.grant)
        self.assertEqual(resumed.receipt.state.active_elapsed_seconds, before.active_elapsed_seconds)
        self.assertIsNone(resumed.receipt.operation_record_id)
        self.assertEqual(self.store.gwt_turns, {})
        outcomes = self.attention_outcomes("opportunity-one")
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0].decision.parked_receipt_id, parked.record.id)
        self.assertEqual(outcomes[0].decision.admitted_operation, "feasibility_check")
        self.assertEqual(outcomes[0].nomination_ids, ())
        restored = MemoryStore.load(self.path)
        history = inquiry_history(restored.snapshot(capture=False), "sketch")
        self.assertEqual(tuple(item.receipt.state.sequence for item in history), (0, 1, 1))
        self.assertEqual(history[-1].receipt.state.status, "active")
        with self.assertRaisesRegex(ValueError, "exactly one retained outcome"):
            self.admit_opportunity(parked, batch_id="opportunity-one")

    def test_parked_deferral_records_the_opportunity_and_allocates_nothing(self) -> None:
        parked = self.parked()
        owner = self.owner()
        decision = ThoughtAttentionDecision(disposition="defer", reason="Conversation holds attention now.")
        focus = captured_private_focus(self.store.snapshot(capture=False), self.now)
        owner.record_attention_outcome(batch_id="deferred-opportunity", nomination_ids=(), decision=decision,
            grant=self.grant, focus=focus)
        head = inquiry_history(self.store.snapshot(capture=False), "sketch")[-1]
        self.assertEqual(head.record.id, parked.record.id)
        self.assertEqual(head.receipt.state.status, "paused")
        self.assertEqual(self.store.gwt_turns, {})
        self.assertEqual(self.attention_outcomes("deferred-opportunity")[0].decision.disposition, "defer")
        with self.assertRaisesRegex(ValueError, "exactly one retained outcome"):
            owner.record_attention_outcome(batch_id="deferred-opportunity", nomination_ids=(),
                decision=ThoughtAttentionDecision(disposition="defer", reason="Still busy."), grant=self.grant,
                focus=focus)

    def test_retained_parked_work_does_not_hold_the_private_slot(self) -> None:
        parked = self.parked()
        view = self.store.snapshot(capture=False)
        self.assertIsNone(active_inquiry(view, observer_entity_id=self.observer.id))
        self.assertEqual(parked_inquiries(view, observer_entity_id=self.observer.id)[0].record.id, parked.record.id)
        competing = self.select("different-question")
        active = active_inquiry(self.store.snapshot(capture=False), observer_entity_id=self.observer.id)
        assert active is not None
        self.assertEqual(active.record.id, competing.record.id)
        with self.assertRaisesRegex(GhostConflict, "already holds this observer's private slot"):
            self.admit_opportunity(parked)

    def test_parked_heads_follow_receipt_order_instead_of_episode_names(self) -> None:
        older = self.parked(episode_id="z-older")
        newer = self.parked(episode_id="a-newer")
        restored = MemoryStore.load(self.path)
        heads = parked_inquiries(restored.snapshot(capture=False), observer_entity_id=self.observer.id)
        self.assertEqual(tuple(head.record.id for head in heads), (newer.record.id, older.record.id))

    def test_newest_exhausted_inquiry_does_not_hide_eligible_older_work(self) -> None:
        older = self.parked(episode_id="a-older")
        self.parked(episode_id="z-newer", active_seconds=self.limits.episode_seconds)
        self.clock.set(self.now + timedelta(seconds=self.limits.parked_seconds))
        candidate = parked_candidate(self.owner(), observer_entity_id=self.observer.id, grant=self.grant)
        assert candidate is not None
        self.assertEqual(candidate.inquiry.record.id, older.record.id)

    def test_current_critical_epoch_excludes_older_parked_work(self) -> None:
        self.parked()
        self.clock.set(self.now + timedelta(seconds=self.limits.parked_seconds))
        grant = self.grant.model_copy(update={"critical_epoch": self.grant.critical_epoch + 1})
        candidate = parked_candidate(self.owner(), observer_entity_id=self.observer.id, grant=grant)
        self.assertIsNone(candidate)

    def advice(self, instruction: str) -> CapturedPrivateFocus:
        """Commit accepted focus that expires before this inquiry's parked deadline."""
        batch = self.store.begin_batch("focus:" + instruction, embed=False)
        batch.create(Feature(content=instruction, feature_type=FeatureType.StateUpdate,
            interlocus=InterlocusType.PrivateInternal, provenance=Provenance(source_kind=SourceKind.Self),
            metadata={"control_kind": "attention_focus",
                "focus": FocusState(accept=True, instruction=instruction, strength=0.4).model_dump(mode="json"),
                "expires_at": (self.now + timedelta(seconds=10)).isoformat()}))
        self.commit(batch)
        return captured_private_focus(self.store.snapshot(capture=False), self.now)

    def test_expired_focus_rejects_resume_without_spending_the_opportunity(self) -> None:
        parked = self.parked()
        focus = self.advice("Examine this concern")
        self.clock.set(self.now + timedelta(seconds=10))
        with self.assertRaisesRegex(PrivateFocusChanged, "expired"):
            self.admit_opportunity(parked, focus=focus)
        self.assertEqual(self.attention_outcomes("opportunity-one"), [])
        self.assertEqual(inquiry_history(self.store.snapshot(capture=False), "sketch")[-1].record.id, parked.record.id)

    def test_replaced_focus_rejects_resume_and_parked_only_deferral(self) -> None:
        parked = self.parked()
        focus = self.advice("First concern")
        self.advice("Different concern")
        with self.assertRaisesRegex(PrivateFocusChanged, "changed"):
            self.admit_opportunity(parked, focus=focus)
        with self.assertRaisesRegex(PrivateFocusChanged, "changed"):
            self.owner().record_attention_outcome(batch_id="stale-deferral", nomination_ids=(),
                decision=ThoughtAttentionDecision(disposition="defer", reason="Earlier advice held attention."),
                grant=self.grant, focus=focus)
        self.assertEqual(self.attention_outcomes("opportunity-one"), [])
        self.assertEqual(self.attention_outcomes("stale-deferral"), [])

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
