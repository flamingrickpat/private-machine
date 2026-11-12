"""Immediate entity maintenance through actual intake, triage, and checkpoint files.

Semantic drafts and classification values are explicit boundary inputs. These
checks do not certify generated interpretation or substitute for the real LLM
gate. All collaborators below are production stores, bus modules, and adapters.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest

from pm.bus.bus import BusSystem
from pm.llm.structured import ModelFormatError
from pm.memory.snapshot import GhostConflict
from pm.memory.store import MemoryStore
from pm.records.base import RecordScope
from pm.records.cycle_state import PersonaConfig
from pm.records.enums import EntityClass
from pm.records.memory_records import Entity, Feature
from workflows.cognitive_control.host import CognitiveControl
from workflows.entity_context.owner import EntityContextOwner, active_claims
from workflows.entity_context.schemas import (
    EntityClaimDraft, EntityContextKey, EntityContextProposal,
    EntityExtractionDraft, InteractionClaim,
)
from workflows.input_triage.entity_updates import capture_entity_extraction, hydrate_entity_extraction
from workflows.input_triage.schemas import EarlyTriageAssessment, EntityExtractionInput, TriageEvidence
from workflows.main_cognition.schemas import TriageItem, TriageResult
from workflows.public_speech.publication_control import ReplySuperseded
from workflows.public_speech.user_input import UserInputProxy


class ImmediateEntityUpdateTests(unittest.TestCase):
    """Exercise atomic maintenance independently of a main cycle or ToM action."""

    def test_foreign_iso_offsets_normalize_before_strict_internal_claim_hydration(self) -> None:
        source = self.submit("I cannot answer until 12:10 UTC.")
        payload = self.report_draft(source).model_dump(mode="json")
        payload["applicable_from_utc"] = "2026-11-12T13:00:00+01:00"
        payload["review_after_utc"] = "2026-11-12T12:05:00Z"
        payload["applicable_until_utc"] = "2026-11-12T12:10:00"
        draft = EntityClaimDraft.model_validate(payload)
        self.assertEqual(draft.applicable_from_utc, datetime(2026, 11, 12, 12, 0))
        self.assertEqual(draft.review_after_utc, datetime(2026, 11, 12, 12, 5))
        self.assertEqual(draft.applicable_until_utc, datetime(2026, 11, 12, 12, 10))
        proposal = hydrate_entity_extraction(self.capture(source), EntityExtractionDraft(claims=[draft]),
                                             producer="immediate")[0]
        review_time = proposal.claims[0].review_after_utc
        assert review_time is not None
        self.assertEqual(review_time, datetime(2026, 11, 12, 12, 5))
        self.assertIsNone(review_time.tzinfo)

    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path: Path = Path(directory.name) / "immediate-entity.pickle"
        self.bus: BusSystem = BusSystem()
        self.store: MemoryStore = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"), checkpoint=self.path)
        self.observer: Entity = Entity(content="Spamuel", aliases=[], entity_class=EntityClass.Self)
        self.partner: Entity = Entity(content="Jesse", aliases=[], entity_class=EntityClass.Human)
        self.peer: Entity = Entity(content="Mara", aliases=[], entity_class=EntityClass.AI)
        for identity in (self.observer, self.partner, self.peer):
            self.store.store_knoxel(identity)
        self.store.accept_import()
        self.owner: EntityContextOwner = EntityContextOwner(self.store)
        self.control: CognitiveControl = self.controller(self.store)
        self.replies: list[str] = []
        self.input: UserInputProxy = UserInputProxy("input", self.bus, self.runtime_error, time.monotonic,
            self.store, self.replies.append)
        self.input.register_triage_input = self.control.register_input

    def runtime_error(self, error: BaseException) -> None:
        """Preserve any actual bus-module error as a failing boundary check."""
        raise error

    def controller(self, store: MemoryStore) -> CognitiveControl:
        """Construct the real control owner and rebuild durable assessments."""
        control = CognitiveControl("control", self.bus, self.runtime_error, time.monotonic, None, ghost=store)
        control.entity_context_owner = EntityContextOwner(store)
        control.enable_input_triage()
        return control

    def submit(self, text: str, *, speaker: str | None = None) -> Feature:
        """Use durable input and its real classification fence, without inference."""
        correlation = self.input.submit(text, speaker=speaker)
        return next(record for record in self.store.all_features
                    if record.metadata.get("exchange_correlation") == correlation)

    def capture(self, source: Feature, *, subject_ids: list[int] | None = None) -> EntityExtractionInput:
        """Freeze actual source meaning and the accepted claims supplied to extraction."""
        view = self.store.recovery_snapshot(capture=False)
        return capture_entity_extraction(self.owner, view,
            subject_ids=subject_ids if subject_ids is not None else [self.partner.id],
            observer_entity_id=self.observer.id, source_ids=[source.id], evidence_ids=[source.id],
            as_of_utc=source.timestamp_world_begin, token_budget=3000)

    def report_draft(self, source: Feature, *, subject_entity_id: int | None = None) -> EntityClaimDraft:
        """Declare a deterministic report boundary input with a known source citation."""
        return EntityClaimDraft(subject_entity_id=subject_entity_id if subject_entity_id is not None else self.partner.id,
            source_ids=[source.id], exact_quote=source.content, layer="situation", claim_key="availability.respond",
            statement=source.content, epistemic_kind="report", confidence=0.9,
            review_after_utc=source.timestamp_world_begin + timedelta(hours=1),
            interaction=InteractionClaim(dimension="respond", ability="cannot"))

    def assessment(self, extraction: EntityExtractionInput,
                   proposals: list[EntityContextProposal]) -> EarlyTriageAssessment:
        """Keep every supplied source fingerprint in the actual control contract."""
        fingerprints = {source.dependency.record_id: source.dependency.fingerprint for source in extraction.sources}
        heads = {source.dependency.record_id for source in extraction.sources if source.dependency.current_head}
        for family in extraction.families:
            for dependency in [family.context.identity_source, *family.context.source_dependencies]:
                fingerprints[dependency.record_id] = dependency.fingerprint
                if dependency.current_head:
                    heads.add(dependency.record_id)
        source_fingerprints = {record_id: fingerprints[record_id] for record_id in extraction.source_ids}
        return EarlyTriageAssessment(source_ids=extraction.source_ids, source_fingerprints=source_fingerprints,
            guidance_tick=None, result=TriageResult(items=[TriageItem(knoxel_id=source_id, category="Moderate",
                reasoning="Explicit boundary input; no generated classification is claimed.")
                for source_id in extraction.source_ids]),
            evidence=TriageEvidence(record_fingerprints=fingerprints, commit_ids=extraction.input_commit_ids,
                                    current_head_ids=sorted(heads)), entity_proposals=proposals)

    def key(self, subject_entity_id: int) -> EntityContextKey:
        """Name an exact runtime observer family for result inspection."""
        return EntityContextKey(subject_entity_id=subject_entity_id, observer_entity_id=self.observer.id, scope=RecordScope())

    def test_hydration_derives_source_meaning_clocks_and_ids_on_the_host(self) -> None:
        source = self.submit("I cannot answer for a while.")
        extraction = self.capture(source)
        draft = EntityExtractionDraft(claims=[self.report_draft(source)])
        proposals = hydrate_entity_extraction(extraction, draft, producer="immediate")
        self.assertEqual(len(proposals), 1)
        proposal = proposals[0]
        self.assertEqual(proposal.key, self.key(self.partner.id))
        self.assertIsNone(proposal.parent_receipt_id)
        self.assertEqual(proposal.input_commit_ids, extraction.input_commit_ids)
        self.assertEqual(proposal.claims[0].evidence[0].source, extraction.sources[0].dependency)
        self.assertEqual(proposal.claims[0].observed_at_utc, source.timestamp_world_begin)
        self.assertEqual(proposal.claims[0].applicable_from_utc, source.timestamp_world_begin)
        self.assertEqual(proposal.claims[0].claim_id, "")
        self.assertEqual(hydrate_entity_extraction(extraction, draft, producer="immediate"), proposals)
        self.assertNotIn('"fingerprint"', draft.model_dump_json())
        self.assertNotIn('"proposal_id"', draft.model_dump_json())
        self.assertFalse(self.control.has_assessment(source.id))
        self.assertEqual(self.store.states, [])

    def test_triage_and_entity_update_commit_atomically_and_reload_before_main(self) -> None:
        epoch = self.control.publication_epoch()
        source = self.submit("I cannot answer for a while.")
        with self.assertRaises(ReplySuperseded):
            self.control.validate_publication(epoch, None)
        extraction = self.capture(source)
        proposals = hydrate_entity_extraction(extraction,
            EntityExtractionDraft(claims=[self.report_draft(source)]), producer="immediate")
        accepted = self.control.accept_triage(self.assessment(extraction, proposals))
        assert accepted.entity_outcomes is not None
        self.assertEqual(len(accepted.entity_outcomes), 1)
        outcome = accepted.entity_outcomes[0]
        self.assertEqual(outcome.disposition, "staged")
        assert outcome.staged_receipt_id is not None
        assessment_record = next(record for record in self.store.all_features
            if record.metadata.get("control_kind") == "input_triage")
        history = self.store.causal_history
        assert history is not None
        atomic_commit = next(transaction for transaction in history.transactions.values()
            if outcome.staged_receipt_id in transaction.created_ids)
        self.assertTrue(atomic_commit.recovery_root)
        self.assertEqual(set(atomic_commit.created_ids), {outcome.staged_receipt_id, assessment_record.id})
        self.assertEqual(EarlyTriageAssessment.model_validate(assessment_record.metadata["assessment"]), accepted)
        self.assertTrue(self.control.has_assessment(source.id))
        self.control.validate_publication(epoch, None)
        self.assertEqual(self.store.states, [])
        restored = MemoryStore.from_checkpoint(self.path)
        rebuilt = self.controller(restored)
        self.assertTrue(rebuilt.has_assessment(source.id))
        rebuilt.validate_publication(epoch, None)
        claims = active_claims(restored.snapshot(capture=False), key=self.key(self.partner.id))
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].statement, source.content)
        self.assertEqual(restored.states, [])
        self.assertEqual(dict(restored.gwt_turns), {})

    def test_stale_second_family_retains_hold_without_partial_first_family_commit(self) -> None:
        epoch = self.control.publication_epoch()
        source = self.submit("I am occupied, and Mara is occupied too.")
        frozen = self.capture(source, subject_ids=[self.partner.id, self.peer.id])
        semantic = EntityExtractionDraft(claims=[self.report_draft(source),
            self.report_draft(source, subject_entity_id=self.peer.id)])
        pending_proposals = hydrate_entity_extraction(frozen, semantic, producer="immediate")
        pending = self.assessment(frozen, pending_proposals)
        newer = self.submit("My current situation changed.", speaker="Mara")
        fresh_peer = self.capture(newer, subject_ids=[self.peer.id])
        peer_proposals = hydrate_entity_extraction(fresh_peer,
            EntityExtractionDraft(claims=[self.report_draft(newer, subject_entity_id=self.peer.id)]), producer="immediate")
        self.control.accept_triage(self.assessment(fresh_peer, peer_proposals))
        before_ids = set(self.store.snapshot(capture=False).all_knoxels)
        before_checkpoint = self.path.read_bytes()
        with self.assertRaisesRegex(GhostConflict, "stale_parent"):
            self.control.accept_triage(pending)
        self.assertEqual(set(self.store.snapshot(capture=False).all_knoxels), before_ids)
        self.assertEqual(self.path.read_bytes(), before_checkpoint)
        self.assertFalse(self.control.has_assessment(source.id))
        self.assertTrue(self.control.has_assessment(newer.id))
        self.assertEqual(active_claims(self.store.snapshot(capture=False), key=self.key(self.partner.id)), [])
        self.assertEqual(len(active_claims(self.store.snapshot(capture=False), key=self.key(self.peer.id))), 1)
        with self.assertRaises(ReplySuperseded):
            self.control.validate_publication(epoch, None)
        restored = MemoryStore.from_checkpoint(self.path)
        rebuilt = self.controller(restored)
        self.assertFalse(rebuilt.has_assessment(source.id))
        with self.assertRaises(ReplySuperseded):
            rebuilt.publication_epoch()
        self.assertEqual(active_claims(restored.snapshot(capture=False), key=self.key(self.partner.id)), [])
        refreshed = self.capture(source, subject_ids=[self.partner.id, self.peer.id])
        refreshed_proposals = hydrate_entity_extraction(refreshed, semantic, producer="immediate")
        self.assertNotEqual(refreshed_proposals[1].proposal_id, pending_proposals[1].proposal_id)
        self.assertIsNotNone(refreshed_proposals[1].parent_receipt_id)
        self.control.accept_triage(self.assessment(refreshed, refreshed_proposals))
        self.assertTrue(self.control.has_assessment(source.id))
        self.control.validate_publication(epoch, None)

    def test_bad_quote_and_outside_source_or_subject_ids_reject_before_acceptance(self) -> None:
        source = self.submit("I cannot answer for a while.")
        extraction = self.capture(source)
        outside = self.submit("This source was not in the captured evidence.", speaker="Mara")
        original = self.report_draft(source)
        bad_quotes = original.model_copy(update={"exact_quote": "Words nobody said."})
        outside_source = original.model_copy(update={"source_ids": [source.id, outside.id]})
        outside_subject = original.model_copy(update={"subject_entity_id": self.peer.id})
        before_ids = set(self.store.snapshot(capture=False).all_knoxels)
        for draft in (bad_quotes, outside_source, outside_subject):
            with self.subTest(statement=draft.exact_quote, sources=draft.source_ids, subject=draft.subject_entity_id):
                with self.assertRaises(ModelFormatError):
                    hydrate_entity_extraction(extraction, EntityExtractionDraft(claims=[draft]), producer="immediate")
        self.assertEqual(set(self.store.snapshot(capture=False).all_knoxels), before_ids)
        self.assertFalse(self.control.has_assessment(source.id))

    def test_background_correction_uses_an_existing_claim_from_the_frozen_family(self) -> None:
        original = self.submit("My occupation is librarian.")
        first_capture = self.capture(original)
        first_draft = self.report_draft(original).model_copy(update={"layer": "background", "claim_key": "occupation",
                                                                  "interaction": None, "review_after_utc": None})
        initial = hydrate_entity_extraction(first_capture, EntityExtractionDraft(claims=[first_draft]), producer="immediate")
        self.control.accept_triage(self.assessment(first_capture, initial))
        old_claim = active_claims(self.store.snapshot(capture=False), key=self.key(self.partner.id))[0]
        correction = self.submit("Correction: my occupation is museum guide.")
        captured = self.capture(correction)
        self.assertIn(old_claim.claim_id, [item.claim.claim_id for item in captured.families[0].context.claims])
        corrected_draft = self.report_draft(correction).model_copy(update={"layer": "background", "claim_key": "occupation",
            "interaction": None, "review_after_utc": None, "correction_target_ids": [old_claim.claim_id]})
        proposals = hydrate_entity_extraction(captured, EntityExtractionDraft(claims=[corrected_draft]), producer="immediate")
        self.assertEqual(proposals[0].retract_claim_ids, [old_claim.claim_id])
        self.assertEqual(proposals[0].parent_receipt_id, captured.families[0].parent_receipt_id)
        self.control.accept_triage(self.assessment(captured, proposals))
        current = active_claims(self.store.snapshot(capture=False), key=self.key(self.partner.id))
        self.assertEqual(len(current), 1)
        self.assertEqual(current[0].statement, correction.content)
        self.assertEqual(self.store.snapshot(capture=False).get(original.id).content, original.content)

    def test_existing_claim_id_from_another_family_cannot_be_a_correction_target(self) -> None:
        source = self.submit("I cannot answer for a while.")
        captured = self.capture(source)
        proposals = hydrate_entity_extraction(captured,
            EntityExtractionDraft(claims=[self.report_draft(source)]), producer="immediate")
        self.control.accept_triage(self.assessment(captured, proposals))
        foreign_claim = active_claims(self.store.snapshot(capture=False), key=self.key(self.partner.id))[0]
        peer_source = self.submit("My current situation changed.", speaker="Mara")
        peer_capture = self.capture(peer_source, subject_ids=[self.peer.id])
        draft = self.report_draft(peer_source, subject_entity_id=self.peer.id).model_copy(
            update={"correction_target_ids": [foreign_claim.claim_id]})
        with self.assertRaisesRegex(ModelFormatError, "existing claim ID"):
            hydrate_entity_extraction(peer_capture, EntityExtractionDraft(claims=[draft]), producer="immediate")
        self.assertFalse(self.control.has_assessment(peer_source.id))
