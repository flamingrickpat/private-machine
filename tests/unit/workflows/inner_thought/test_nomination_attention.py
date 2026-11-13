"""Exercise retained nomination stages and attention against actual stores.

The explicit semantic proposals are boundary inputs. These checks establish
storage, provenance, selection limits, and prompt construction, not generation
quality. The connected OpenRouter workflow gate supplies that separate evidence.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pm.memory.snapshot import GhostConflict
from pm.memory.store import MemoryStore
from pm.records.base import Provenance, RecordScope
from pm.records.cycle_state import PersonaConfig
from pm.records.enums import EntityClass, FeatureType, InterlocusType, SourceKind
from pm.records.memory_records import Entity, Feature
from workflows.cognitive_control.attention import validate_private_attention
from workflows.inner_thought.inquiry import stage_attention
from workflows.inner_thought.nominations import (
    capture_inquiry_grant, read_nomination_batch, source_roots, stage_nomination_batch,
)
from workflows.inner_thought.schemas import (
    AttentionCandidate, InnerRequest, NominationBatch, ParkedInquiryOffer, PrivateAttentionContext,
    ThoughtAttentionDecision, ThoughtCandidate, ThoughtJudgments, ThoughtNomination,
)
from workflows.inner_thought.stage_runner import stage_manifest
from workflows.inner_thought.stages.nominate.steps import (
    build_messages as nomination_messages, load_prompts as nomination_prompts, validate_nominations,
)
from workflows.inner_thought.stages.trigger.steps import (
    AttentionStageRequest, build_attention_messages, load_prompts as attention_prompts,
)


class NominationAttentionTests(unittest.TestCase):
    """Keep nomination, selection, execution, and evidence authority separate."""

    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "nominations.pickle"
        self.store = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"), checkpoint=self.path)
        batch = self.store.begin_batch("sources", embed=False)
        self.observer = batch.create(Entity(content="Spamuel", aliases=[], entity_class=EntityClass.Self))
        self.partner = batch.create(Entity(content="Jesse", aliases=[], entity_class=EntityClass.Human))
        self.source = batch.create(Feature(content="I wanted us to finish the story together.",
            feature_type=FeatureType.Dialogue, causal=True, entity_id=self.partner.id,
            addressee_entity_id=self.observer.id, interlocus=InterlocusType.Public,
            provenance=Provenance(source_kind=SourceKind.User)))
        batch.preserve_on_recovery()
        self.store.commit(batch)
        self.grant = capture_inquiry_grant(self.store.snapshot(), observer_entity_id=self.observer.id,
            scope=RecordScope(), source_ids=[self.source.id], critical_epoch=0)
        self.judgments = ThoughtJudgments(interestingness=0.9, character_relevance=0.8,
            plausibility=0.25, consequence=0.6, desirability=0.95, action_suitability=0.2,
            reason="The shared story matters, while the explanation remains uncertain.")
        self.nomination = ThoughtNomination(text="Maybe the unfinished story means something to Jesse.",
            question="What made finishing it together matter?", evidence_ids=(self.source.id,),
            reality="inferred", judgments=self.judgments, suggested_operation="perspective_shift")

    def test_empty_stage_is_durable_and_replay_does_not_invent_nominations(self) -> None:
        batch = self.store.begin_batch("empty-stage", embed=False)
        staged = stage_nomination_batch(batch, NominationBatch(nominations=(), reason="No useful private question now."),
            grant=self.grant, batch_id="empty-opportunity")
        batch.preserve_on_recovery()
        self.store.commit(batch)
        reloaded = MemoryStore.from_checkpoint(self.path)
        retained = read_nomination_batch(reloaded.snapshot(), "empty-opportunity")
        assert retained is not None
        self.assertEqual(retained.record.id, staged.record.id)
        self.assertEqual(retained.nominations, ())
        self.assertEqual(retained.receipt.reason, "No useful private question now.")
        self.assertEqual(list(reloaded.gwt_turns), [])
        repeat = reloaded.begin_batch("repeat-empty", embed=False)
        with self.assertRaisesRegex(ValueError, "read instead of regenerated"):
            stage_nomination_batch(repeat, NominationBatch(nominations=(), reason="No question."),
                grant=self.grant, batch_id="empty-opportunity")

    def test_attention_allocates_only_after_separate_retained_nomination_stage(self) -> None:
        batch = self.store.begin_batch("nomination-stage", embed=False)
        staged = stage_nomination_batch(batch, NominationBatch(nominations=(self.nomination,), reason="An unresolved shared concern."),
            grant=self.grant, batch_id="story-opportunity")
        batch.preserve_on_recovery()
        self.store.commit(batch)
        nomination_id = staged.nominations[0].record.id
        context = PrivateAttentionContext(candidates=(AttentionCandidate(nomination_id=nomination_id,
            nomination=self.nomination, root_source_ids=(self.source.id,)),), pending_input_ids=(),
            accepted_focus=None, active_intention_ids=(), attention_summary="A broad aperture; no urgent competing input.")
        decision = validate_private_attention(context, ThoughtAttentionDecision(nomination_id=nomination_id,
            disposition="select", reason="Explore the uncertainty without treating it as a fact."))
        selection_batch = self.store.begin_batch("attention-stage", embed=False)
        selected = stage_attention(selection_batch, [nomination_id], decision,
            episode_id="story-inquiry", now_utc=datetime(2026, 11, 12, 15, 0))
        selection_batch.preserve_on_recovery()
        self.store.commit(selection_batch)
        assert selected.inquiry is not None
        self.assertEqual(selected.inquiry.receipt.state.sequence, 0)
        self.assertEqual(selected.inquiry.receipt.state.node_ids, ())
        self.assertEqual(list(self.store.gwt_turns), [])
        self.assertEqual(self.nomination.judgments.plausibility, 0.25)
        view = self.store.snapshot()
        roots = source_roots(view, [staged.record.id], scope=RecordScope(),
            permitted_ids=frozenset(view.revisions))
        self.assertEqual(roots, (self.source.id,))

    def test_competing_stage_completion_cannot_duplicate_one_invocation(self) -> None:
        first = self.store.begin_batch("empty-one", embed=False)
        second = self.store.begin_batch("empty-two", embed=False)
        result = NominationBatch(nominations=(), reason="No question attracted attention.")
        stage_nomination_batch(first, result, grant=self.grant, batch_id="same-opportunity")
        stage_nomination_batch(second, result, grant=self.grant, batch_id="same-opportunity")
        first.preserve_on_recovery()
        second.preserve_on_recovery()
        self.store.commit(first)
        with self.assertRaises(GhostConflict):
            self.store.commit(second)

    def test_shared_attention_rejects_unknown_identity_and_blocked_selection(self) -> None:
        context = PrivateAttentionContext(candidates=(AttentionCandidate(nomination_id=100,
            nomination=self.nomination, root_source_ids=(self.source.id,)),), pending_input_ids=(),
            accepted_focus=None, active_intention_ids=(), attention_summary="Accepted wait.", blocked_reason="wait")
        with self.assertRaisesRegex(ValueError, "outside the supplied batch"):
            validate_private_attention(context, ThoughtAttentionDecision(nomination_id=101,
                disposition="select", reason="Choose it."))
        with self.assertRaisesRegex(ValueError, "cannot allocate"):
            validate_private_attention(context, ThoughtAttentionDecision(nomination_id=100,
                disposition="select", reason="Choose it."))
        decision = ThoughtAttentionDecision(nomination_id=None, disposition="defer", reason="The accepted wait remains active.")
        self.assertEqual(validate_private_attention(context, decision), decision)

    def test_parked_work_wins_the_slot_only_on_its_own_offer(self) -> None:
        offer = ParkedInquiryOffer(receipt_id=77, episode_id="story-inquiry",
            question="What made finishing it together matter?", stop_reason="Awaiting Jesse's actual motive.",
            resume_after_utc=datetime(2026, 11, 12, 14, 0), deadline_elapsed=True, available_evidence_ids=())
        context = PrivateAttentionContext(candidates=(), parked=(offer,), pending_input_ids=(),
            accepted_focus=None, active_intention_ids=(), attention_summary="A broad aperture; parked work is due.")
        admitted = validate_private_attention(context, ThoughtAttentionDecision(parked_receipt_id=77,
            disposition="select", admitted_operation="feasibility_check",
            reason="The parked wait has passed and nothing new competes."))
        self.assertEqual(admitted.parked_receipt_id, 77)
        self.assertEqual(admitted.admitted_operation, "feasibility_check")
        declined = validate_private_attention(context,
            ThoughtAttentionDecision(disposition="defer", reason="Ordinary input holds attention now."))
        self.assertIsNone(declined.parked_receipt_id)
        with self.assertRaisesRegex(ValueError, "outside the supplied offers"):
            validate_private_attention(context, ThoughtAttentionDecision(parked_receipt_id=99,
                disposition="select", admitted_operation="feasibility_check", reason="Choose it."))
        with self.assertRaisesRegex(KeyError, "unknown thought operation: daydream"):
            validate_private_attention(context, ThoughtAttentionDecision(parked_receipt_id=77,
                disposition="select", admitted_operation="daydream", reason="Choose it."))
        # Nothing was offered, so no identity can be admitted from this context.
        unoffered = PrivateAttentionContext(candidates=(), pending_input_ids=(), accepted_focus=None,
            active_intention_ids=(), attention_summary="Nothing competes.")
        with self.assertRaisesRegex(ValueError, "outside the supplied offers"):
            validate_private_attention(unoffered, ThoughtAttentionDecision(parked_receipt_id=77,
                disposition="select", admitted_operation="feasibility_check", reason="Choose it."))
        with self.assertRaisesRegex(ValueError, "outside the supplied batch"):
            validate_private_attention(unoffered, ThoughtAttentionDecision(nomination_id=77,
                disposition="select", reason="Choose it."))

    def test_attention_prompt_delivers_offered_parked_work_and_legal_operations(self) -> None:
        request = InnerRequest(companion_name="Spamuel", user_name="Jesse", candidates=[
            ThoughtCandidate(knoxel_id=self.source.id, text=self.source.content, revision=1)])
        offer = ParkedInquiryOffer(receipt_id=77, episode_id="story-inquiry",
            question="What made finishing it together matter?", stop_reason="Stalled without new evidence.",
            resume_after_utc=None, deadline_elapsed=False, available_evidence_ids=(self.source.id,))
        context = PrivateAttentionContext(candidates=(), parked=(offer,), pending_input_ids=(self.source.id,),
            accepted_focus=None, active_intention_ids=(), attention_summary="New input competes for attention.")
        messages = build_attention_messages(AttentionStageRequest(request=request, attention=context),
            attention_prompts())
        system, user = str(messages[0].content), str(messages[1].content)
        self.assertIn("parked", system)
        self.assertIn("parked_receipt_id", user)
        self.assertIn("available_evidence_ids", user)
        self.assertIn("feasibility_check", user)

    def test_nomination_and_attention_prompts_keep_separate_schema_and_source_contracts(self) -> None:
        request = InnerRequest(companion_name="Spamuel", user_name="Jesse", candidates=[
            ThoughtCandidate(knoxel_id=self.source.id, text=self.source.content, revision=1)])
        proposals = NominationBatch(nominations=(self.nomination,), reason="The story raises a question.")
        self.assertEqual(validate_nominations(request, proposals), proposals)
        invalid = proposals.model_copy(update={"nominations": (self.nomination.model_copy(update={"evidence_ids": (999,)}),)})
        with self.assertRaisesRegex(ValueError, "outside the frozen request"):
            validate_nominations(request, invalid)
        self.assertTrue(stage_manifest("nominate").is_file())
        messages = nomination_messages(request, nomination_prompts())
        self.assertIn("separate attention stage", str(messages[0].content))
        self.assertIn(self.source.content, str(messages[1].content))
        context = PrivateAttentionContext(candidates=(), pending_input_ids=(self.source.id,),
            accepted_focus=None, active_intention_ids=(), attention_summary="New input competes for attention.")
        attention = build_attention_messages(AttentionStageRequest(request=request, attention=context), attention_prompts())
        self.assertIn("pending_input_ids", str(attention[1].content))
        self.assertIn("dismiss", str(attention[1].content))


if __name__ == "__main__":
    unittest.main()
