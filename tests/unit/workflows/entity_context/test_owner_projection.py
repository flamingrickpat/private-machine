"""Accepted entity claims and current projections against real stores and files.

These deterministic boundary inputs do not certify generated interpretation.
The separate real workflow gate exercises extraction and character choices.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier
from typing import Literal
import unittest

from pydantic import ValidationError

from pm.memory.dependencies import capture_source
from pm.memory.snapshot import GhostConflict
from pm.memory.store import MemoryStore
from pm.records.base import Provenance, RecordScope
from pm.records.cycle_state import PersonaConfig
from pm.records.enums import EntityClass, FeatureType, FrameEnum, InterlocusType, SourceKind
from pm.records.global_workspace import GwtTurnRecord
from pm.records.memory_records import Entity, Feature
from workflows.entity_context.owner import EntityContextOwner, active_claims, claim_identity
from workflows.entity_context.projection import (
    EntityContextChanged, expect_entity_context, project_entity_context,
    render_entity_context, validate_entity_context,
)
from workflows.entity_context.schemas import (
    EntityClaimDraft, EntityClaimEvidence, EntityContextClaim, EntityContextKey, EntityContextProposal,
    EntityContextView, EntityExtractionDraft, EntityProjectionQuery, EntityUpdateResult, InteractionClaim,
)


class EntityContextStoreTests(unittest.TestCase):
    """Verify durable attribution, current meaning, and scoped acceptance."""

    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path: Path = Path(directory.name) / "entity-store.pickle"
        self.store: MemoryStore = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"), checkpoint=self.path)
        batch = self.store.begin_batch("identities", embed=False)
        self.observer: Entity = batch.create(Entity(content="Spamuel", aliases=[], entity_class=EntityClass.Self))
        self.partner: Entity = batch.create(Entity(content="Jesse", aliases=[], entity_class=EntityClass.Human))
        self.peer: Entity = batch.create(Entity(content="Mara", aliases=[], entity_class=EntityClass.AI))
        batch.preserve_on_recovery()
        self.store.commit(batch)
        self.owner: EntityContextOwner = EntityContextOwner(self.store)
        self.now: datetime = datetime(2026, 11, 11, 10, 0)
        self.key: EntityContextKey = EntityContextKey(subject_entity_id=self.partner.id,
            observer_entity_id=self.observer.id, scope=RecordScope())

    def report(self, text: str, *, when: datetime | None = None,
               scope: RecordScope = RecordScope(), speaker: Entity | None = None) -> Feature:
        """Commit an attributed external utterance through the production store."""
        entity = speaker if speaker is not None else self.partner
        moment = when if when is not None else self.now
        batch = self.store.begin_batch(f"report-{len(self.store.all_features)}", embed=False)
        record = batch.create(Feature(content=text, feature_type=FeatureType.Dialogue, causal=True,
            entity_id=entity.id, addressee_entity_id=self.observer.id, interlocus=InterlocusType.Public,
            scope=scope, timestamp_world_begin=moment, timestamp_world_end=moment,
            provenance=Provenance(source_kind=SourceKind.User if entity is self.partner else SourceKind.Agent)))
        batch.preserve_on_recovery()
        self.store.commit(batch)
        return record

    def proposal(self, source: Feature, proposal_id: str, *, key: EntityContextKey | None = None,
                 claim_key: str = "availability.respond", ability: Literal["can", "cannot", "unknown"] | None = "cannot",
                 statement: str | None = None, review_after_utc: datetime | None = None,
                 applicable_until_utc: datetime | None = None) -> EntityContextProposal:
        """Hydrate explicit boundary input with real source identities and meaning."""
        selected_key = key if key is not None else self.key
        view = self.store.snapshot(capture=False)
        interaction = InteractionClaim(dimension="respond", ability=ability) if ability is not None else None
        evidence = EntityClaimEvidence(source=capture_source(view, source.id, current_head=True), exact_quote=source.content)
        claim = EntityContextClaim(layer="situation", claim_key=claim_key,
            statement=statement if statement is not None else source.content, epistemic_kind="report",
            confidence=0.9, evidence=[evidence], observed_at_utc=source.timestamp_world_begin,
            applicable_from_utc=source.timestamp_world_begin, review_after_utc=review_after_utc,
            applicable_until_utc=applicable_until_utc, interaction=interaction)
        return EntityContextProposal(proposal_id=proposal_id, key=selected_key,
            parent_receipt_id=self.owner.current_head(view, selected_key), producer="immediate",
            input_commit_ids=sorted(view.commit_ids), claims=[claim])

    def accept(self, proposal: EntityContextProposal) -> EntityUpdateResult:
        """Use the actual owner and append-only recovery commit."""
        with self.store.lock:
            batch = self.store.begin_batch(f"accept-{proposal.proposal_id}", embed=False)
            result = self.owner.stage_update(batch, proposal)
            if result.disposition == "staged":
                batch.preserve_on_recovery()
                self.store.commit(batch)
            return result

    def context(self, *, as_of_utc: datetime | None = None,
                claim_keys: list[str] | None = None, key: EntityContextKey | None = None) -> EntityContextView:
        """Project a frozen relevant query rather than every claim in memory."""
        query = EntityProjectionQuery(key=key if key is not None else self.key,
            claim_keys=claim_keys if claim_keys is not None else ["availability.respond"],
            include_interaction=True, token_budget=3000)
        return project_entity_context(self.store.snapshot(capture=False), query=query,
                                      as_of_utc=as_of_utc if as_of_utc is not None else self.now)

    def test_private_append_survives_clean_checkpoint_and_reuses_identity(self) -> None:
        source = self.report("I cannot answer for a while.")
        proposal = self.proposal(source, "absence")
        result = self.accept(proposal)
        self.assertEqual(result.disposition, "staged")
        assert result.staged_receipt_id is not None
        self.assertEqual(self.accept(proposal).staged_receipt_id, result.staged_receipt_id)
        reopened = MemoryStore.from_checkpoint(self.path)
        claims = active_claims(reopened.snapshot(capture=False), key=self.key)
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].claim_id, claim_identity("absence", 0))
        record = reopened.snapshot(capture=False).get(result.staged_receipt_id)
        self.assertIsInstance(record, Feature)
        assert record.lineage is not None
        self.assertEqual(record.lineage.source_event_ids, [source.id])
        self.assertEqual(reopened.snapshot(capture=False).get(source.id).content, source.content)

    def test_correction_preserves_old_receipt_and_invalidates_current_direction(self) -> None:
        original = self.report("I cannot answer for a while.")
        first = self.accept(self.proposal(original, "away"))
        captured = self.context()
        previous = captured.claims[0].claim.claim_id
        returned = self.report("I can answer now.", when=self.now + timedelta(minutes=1))
        proposal = self.proposal(returned, "returned", ability="can")
        proposal = proposal.model_copy(update={"retract_claim_ids": [previous],
            "correction_evidence": proposal.claims[0].evidence})
        self.assertEqual(self.accept(proposal).disposition, "staged")
        current = self.context(as_of_utc=self.now + timedelta(minutes=1))
        self.assertEqual(len(current.claims), 1)
        assert current.claims[0].claim.interaction is not None
        assert first.staged_receipt_id is not None
        self.assertEqual(current.claims[0].claim.interaction.ability, "can")
        self.assertIsNotNone(self.store.get_knoxel_by_id(first.staged_receipt_id))
        self.assertEqual(self.store.snapshot(capture=False).get(original.id).content, original.content)
        with self.assertRaises(EntityContextChanged):
            validate_entity_context(self.store, captured, as_of_utc=self.now + timedelta(minutes=1))

    def test_elapsed_time_weakens_estimate_without_asserting_the_opposite(self) -> None:
        source = self.report("I cannot answer for a while.")
        self.accept(self.proposal(source, "timed", review_after_utc=self.now + timedelta(minutes=10),
            applicable_until_utc=self.now + timedelta(minutes=20)))
        before_count = len(self.store.all_knoxels)
        stale = self.context(as_of_utc=self.now + timedelta(minutes=11))
        expired = self.context(as_of_utc=self.now + timedelta(minutes=21))
        self.assertEqual(stale.claims[0].applicability, "due_for_review")
        self.assertEqual(expired.claims[0].applicability, "expired")
        self.assertIn("Current ability to respond: unknown", render_entity_context(expired))
        assert expired.claims[0].claim.interaction is not None
        self.assertEqual(expired.claims[0].claim.interaction.ability, "cannot")
        self.assertEqual(len(self.store.all_knoxels), before_count)

    def test_equivalent_new_report_preserves_direction_and_original_provenance(self) -> None:
        source = self.report("I cannot answer for a while.")
        self.accept(self.proposal(source, "original-meaning", review_after_utc=self.now + timedelta(hours=1)))
        captured = self.context()
        newer = self.report(source.content, when=self.now + timedelta(minutes=1))
        self.accept(self.proposal(newer, "same-meaning", review_after_utc=self.now + timedelta(hours=2)))
        current = self.context(as_of_utc=self.now + timedelta(minutes=2))
        self.assertNotEqual(current.head_receipt_id, captured.head_receipt_id)
        self.assertEqual(current.projected_meaning_fingerprint, captured.projected_meaning_fingerprint)
        self.assertEqual(len(current.claims), 1)
        with self.store.lock:
            proposal = self.store.begin_batch("reply-with-original-evidence", embed=False)
            expect_entity_context(proposal, captured, as_of_utc=self.now + timedelta(minutes=2))
            self.store.commit(proposal)
        assert self.store.causal_history is not None
        inputs = self.store.causal_history.transactions["reply-with-original-evidence"].input_ids
        self.assertIn(source.id, inputs)
        self.assertNotIn(newer.id, inputs)
        self.assertEqual(captured.source_dependencies[0].record_id, source.id)

    def test_unrelated_same_family_append_does_not_veto_frozen_query(self) -> None:
        source = self.report("I cannot answer for a while.")
        self.accept(self.proposal(source, "availability"))
        captured = self.context()
        background = self.report("My bicycle is blue.")
        self.accept(self.proposal(background, "bicycle", claim_key="bicycle.colour", ability=None))
        validate_entity_context(self.store, captured, as_of_utc=self.now)
        empty = self.context(claim_keys=[])
        self.assertNotIn("bicycle", render_entity_context(empty))

    def test_stale_entity_proposal_does_not_poison_main_appraisal_batch(self) -> None:
        source = self.report("I cannot answer for a while.")
        pending = self.proposal(source, "slow-main")
        main_batch = self.store.begin_batch("valid-main", embed=False, new_tick=True)
        appraisal = main_batch.create(Feature(content="Boundary appraisal receipt", feature_type=FeatureType.StateUpdate))
        self.accept(self.proposal(source, "faster-immediate"))
        with self.store.lock:
            result = self.owner.stage_update(main_batch, pending)
            self.assertEqual(result.disposition, "deferred")
            self.assertEqual(result.reason, "stale_parent")
            appraisal.metadata["entity_context_results"] = [result.model_dump(mode="json")]
            main_batch.set_gwt(GwtTurnRecord(tick_id=main_batch.tick_id, knoxel_ids=[appraisal.id]))
            self.store.commit(main_batch)
        reopened = MemoryStore.from_checkpoint(self.path)
        persisted = reopened.snapshot(capture=False).get(appraisal.id)
        self.assertEqual(persisted.metadata["entity_context_results"][0]["disposition"], "deferred")
        self.assertEqual(len(active_claims(reopened.snapshot(capture=False), key=self.key)), 1)

    def test_competing_family_writes_have_one_winner_under_real_commit_lock(self) -> None:
        source = self.report("I cannot answer for a while.")
        proposals = [self.proposal(source, f"concurrent-{index}") for index in range(2)]
        ready = Barrier(2)

        def worker(index: int) -> bool:
            batch = self.store.begin_batch(f"writer-{index}", embed=False)
            result = self.owner.stage_update(batch, proposals[index])
            self.assertEqual(result.disposition, "staged")
            batch.preserve_on_recovery()
            ready.wait(timeout=10)
            try:
                self.store.commit(batch)
                return True
            except GhostConflict:
                return False

        with ThreadPoolExecutor(max_workers=2) as workers:
            results = list(workers.map(worker, range(2)))
        self.assertEqual(sum(results), 1)
        reopened = MemoryStore.from_checkpoint(self.path)
        self.assertEqual(len(active_claims(reopened.snapshot(capture=False), key=self.key)), 1)

    def test_foreign_observer_and_wrong_reality_are_rejected(self) -> None:
        source = self.report("I cannot answer for a while.")
        foreign = self.key.model_copy(update={"observer_entity_id": self.peer.id})
        self.assertEqual(self.accept(self.proposal(source, "wrong-observer", key=foreign)).disposition, "rejected")
        wrong_reality = self.key.model_copy(update={"scope": RecordScope(reality_id=99)})
        self.assertEqual(self.accept(self.proposal(source, "wrong-reality", key=wrong_reality)).disposition, "rejected")
        meta_key = self.key.model_copy(update={"scope": RecordScope(frame=FrameEnum.Meta)})
        meta_source = self.report("This is a hypothetical situation.", scope=meta_key.scope)
        self.assertEqual(self.accept(self.proposal(meta_source, "scoped-meta", key=meta_key)).disposition, "staged")
        self.assertEqual(self.context().claims, [])

    def test_report_cannot_promote_itself_to_observed_external_state(self) -> None:
        source = self.report("I am probably asleep.")
        proposal = self.proposal(source, "unsupported-observation")
        claim = proposal.claims[0].model_copy(update={"epistemic_kind": "observation"})
        result = self.accept(proposal.model_copy(update={"claims": [claim]}))
        self.assertEqual(result.disposition, "rejected")
        self.assertEqual(self.context().claims, [])

    def test_changed_source_head_defers_without_staging_guards(self) -> None:
        source = self.report("I cannot answer for a while.")
        proposal = self.proposal(source, "original-source")
        correction = self.store.begin_batch("source-successor", embed=False)
        successor = source.copy_for_edit()
        successor.content = "Corrected original content."
        correction.replace(successor)
        self.store.commit(correction)
        batch = self.store.begin_batch("unchanged-main", embed=False)
        result = self.owner.stage_update(batch, proposal)
        self.assertEqual(result.reason, "changed_source")
        self.store.commit(batch)
        self.assertEqual(self.context().claims, [])

    def test_reports_preserve_actual_speaker_in_another_subjects_model(self) -> None:
        source = self.report("Mara cannot answer right now.")
        peer_key = self.key.model_copy(update={"subject_entity_id": self.peer.id})
        self.accept(self.proposal(source, "reported-peer", key=peer_key))
        context = self.context(key=peer_key)
        self.assertEqual(context.claims[0].attributions[0].speaker_entity_id, self.partner.id)
        self.assertIn("source Jesse", render_entity_context(context))
        self.assertEqual(self.context().claims, [])

    def test_conflicting_current_claims_remain_uncertain(self) -> None:
        source = self.report("I cannot answer right now.")
        self.accept(self.proposal(source, "conflict-first"))
        other = self.report("I can answer right now.")
        self.accept(self.proposal(other, "conflict-second", ability="can"))
        context = self.context()
        self.assertEqual(len(context.claims), 2)
        self.assertTrue(all(item.contested for item in context.claims))
        self.assertIn("Current ability to respond: unknown", render_entity_context(context))

    def test_original_freshness_deadline_is_conservative_after_unseen_refresh(self) -> None:
        source = self.report("I cannot answer for a while.")
        self.accept(self.proposal(source, "first-deadline", review_after_utc=self.now + timedelta(minutes=5)))
        captured = self.context()
        repeated = self.report(source.content, when=self.now + timedelta(minutes=1))
        self.accept(self.proposal(repeated, "later-deadline", review_after_utc=self.now + timedelta(hours=1)))
        with self.assertRaises(EntityContextChanged):
            validate_entity_context(self.store, captured, as_of_utc=self.now + timedelta(minutes=6))

    def test_draft_boundary_rejects_invented_host_fields(self) -> None:
        with self.assertRaises(ValidationError):
            EntityExtractionDraft.model_validate({"claims": [], "proposal_id": "model-invented"})
        proposal = self.proposal(self.report("I cannot answer now."), "forged-claim")
        claim = proposal.claims[0].model_copy(update={"claim_id": "model-invented"})
        with self.assertRaises(ValueError):
            self.accept(proposal.model_copy(update={"claims": [claim]}))

    def test_new_receipt_can_project_before_commit_without_a_fabricated_revision(self) -> None:
        source = self.report("I cannot answer for a while.")
        batch = self.store.begin_batch("staged-entity-view", embed=False)
        result = self.owner.stage_update(batch, self.proposal(source, "stage-before-project"))
        assert result.staged_receipt_id is not None
        view = batch.view()
        self.assertNotIn(result.staged_receipt_id, view.revisions)
        context = project_entity_context(view,
            query=EntityProjectionQuery(key=self.key, claim_keys=["availability.respond"], token_budget=3000),
            as_of_utc=self.now)
        self.assertEqual(context.head_receipt_id, result.staged_receipt_id)
        self.assertEqual(context.claims[0].claim.statement, source.content)
        self.assertNotIn(result.staged_receipt_id, [item.record_id for item in context.source_dependencies])
        batch.preserve_on_recovery()
        self.store.commit(batch)

    def test_private_dialogue_to_another_observer_cannot_support_inference(self) -> None:
        batch = self.store.begin_batch("other-observers-dialogue", embed=False)
        source = batch.create(Feature(content="I am occupied.", feature_type=FeatureType.Dialogue, causal=True,
            entity_id=self.partner.id, addressee_entity_id=self.peer.id, interlocus=InterlocusType.Public,
            timestamp_world_begin=self.now, timestamp_world_end=self.now,
            provenance=Provenance(source_kind=SourceKind.User)))
        batch.preserve_on_recovery()
        self.store.commit(batch)
        proposal = self.proposal(source, "wrong-dialogue-perspective")
        claim = proposal.claims[0].model_copy(update={"epistemic_kind": "inference"})
        result = self.accept(proposal.model_copy(update={"claims": [claim]}))
        self.assertEqual(result.disposition, "rejected")
        self.assertEqual(self.context().claims, [])

    def test_repeated_inference_cannot_raise_confidence_without_new_evidence(self) -> None:
        source = self.report("I am occupied.")
        original = self.proposal(source, "original-inference")
        inferred = original.claims[0].model_copy(update={"epistemic_kind": "inference"})
        self.accept(original.model_copy(update={"claims": [inferred]}))
        repeated = self.proposal(source, "inflated-inference")
        inflated = repeated.claims[0].model_copy(update={"epistemic_kind": "inference", "confidence": 0.99})
        result = self.accept(repeated.model_copy(update={"claims": [inflated]}))
        self.assertEqual(result.disposition, "rejected")
        self.assertEqual(len(self.context().claims), 1)

    def test_actual_runtime_observation_requires_provenance_and_execution_role(self) -> None:
        batch = self.store.begin_batch("runtime-observation", embed=False)
        source = batch.create(Feature(content="The partner connection is closed.", feature_type=FeatureType.WorldEvent,
            causal=True, timestamp_world_begin=self.now, timestamp_world_end=self.now,
            provenance=Provenance(source_kind=SourceKind.Workflow, source_ref="transport"),
            metadata={"memory_kind": "executed_event"}))
        batch.preserve_on_recovery()
        self.store.commit(batch)
        proposal = self.proposal(source, "observed-transport")
        observed = proposal.claims[0].model_copy(update={"epistemic_kind": "observation"})
        self.assertEqual(self.accept(proposal.model_copy(update={"claims": [observed]})).disposition, "staged")
        self.assertEqual(self.context().claims[0].claim.epistemic_kind, "observation")
        unmarked_batch = self.store.begin_batch("unmarked-world-text", embed=False)
        unmarked = unmarked_batch.create(Feature(content="A generated world interpretation.", feature_type=FeatureType.WorldEvent,
            causal=True, timestamp_world_begin=self.now, timestamp_world_end=self.now,
            provenance=Provenance(source_kind=SourceKind.Workflow, source_ref="interpretation")))
        unmarked_batch.preserve_on_recovery()
        self.store.commit(unmarked_batch)
        unsupported = self.proposal(unmarked, "not-an-observation")
        inferred = unsupported.claims[0].model_copy(update={"epistemic_kind": "observation"})
        self.assertEqual(self.accept(unsupported.model_copy(update={"claims": [inferred]})).disposition, "rejected")

    def test_equivalent_reports_keep_tight_budget_admission_stable(self) -> None:
        source = self.report("I cannot answer for a while.")
        self.accept(self.proposal(source, "tight-budget-original"))
        view = self.store.snapshot(capture=False)
        captured: EntityContextView | None = None
        for token_budget in range(100, 401):
            context = project_entity_context(view,
                query=EntityProjectionQuery(key=self.key, claim_keys=["availability.respond"], token_budget=token_budget),
                as_of_utc=self.now)
            if context.claims:
                captured = context
                break
        assert captured is not None
        repeated = self.report(source.content, when=self.now + timedelta(minutes=1))
        self.accept(self.proposal(repeated, "tight-budget-different-hash"))
        validate_entity_context(self.store, captured, as_of_utc=self.now + timedelta(minutes=1))
        current = project_entity_context(self.store.snapshot(capture=False), query=captured.query,
                                        as_of_utc=self.now + timedelta(minutes=1))
        self.assertEqual(current.projected_meaning_fingerprint, captured.projected_meaning_fingerprint)
        self.assertEqual(current.claims[0].claim.claim_id, captured.claims[0].claim.claim_id)

    def test_malformed_report_and_hypothetical_estimate_fail_at_model_boundary(self) -> None:
        source = self.report("I cannot answer right now.")
        with self.assertRaises(ValidationError):
            EntityClaimDraft(subject_entity_id=self.partner.id, source_ids=[source.id], layer="situation",
                claim_key="availability.respond", statement=source.content, epistemic_kind="report", confidence=0.9)
        with self.assertRaises(ValidationError):
            EntityClaimDraft(subject_entity_id=self.partner.id, source_ids=[source.id], layer="situation",
                claim_key="availability.respond", statement="An imagined possibility", epistemic_kind="hypothetical",
                confidence=0.3, interaction=InteractionClaim(dimension="respond", ability="cannot"))
