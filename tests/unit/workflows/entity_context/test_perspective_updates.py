"""Main cognition stages a perspective revision through the real entity owner.

These boundary checks use a real MemoryStore and the real EntityContextOwner:
a private finding's stored revision becomes a ``tom`` proposal that the owner
either stages or records as rejected. They make no claim about generated
thought quality; the recorded ``perspective-morning`` scenario and the live
gate cover the real child kernels.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pm.memory.dependencies import capture_source
from pm.memory.store import MemoryStore
from pm.memory.write_batch import WriteBatch
from pm.records.base import Provenance, RecordScope
from pm.records.cycle_state import PersonaConfig
from pm.records.enums import EntityClass, FeatureType, FrameEnum, InterlocusType, SourceKind
from pm.records.memory_records import Entity, Feature
from workflows.entity_context.owner import EntityContextOwner, active_claims, receipt_chain
from workflows.entity_context.schemas import (
    EntityClaimDraft, EntityClaimEvidence, EntityContextClaim, EntityContextKey, EntityContextProposal,
    EntityExtractionDraft, PerspectiveRevision,
)
from pm.records.enums import ActionType
from workflows.main_cognition.entity_updates import prepare_perspective_revisions


class PerspectiveUpdatesFixture(unittest.TestCase):
    """A real store with an observer, a subject, one message, and the owner."""

    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path: Path = Path(directory.name) / "perspective-updates.pickle"
        self.store: MemoryStore = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"),
                                              checkpoint=self.path)
        self.now: datetime = datetime(2026, 11, 14, 9, 0)
        batch = self.store.begin_batch("anchors", embed=False)
        self.observer: Entity = batch.create(Entity(content="Spamuel", aliases=[], entity_class=EntityClass.Self))
        self.partner: Entity = batch.create(Entity(content="Jesse", aliases=[], entity_class=EntityClass.Human))
        self.message: Feature = batch.create(Feature(
            content="I am usually up around six on workdays.", feature_type=FeatureType.Dialogue, causal=True,
            entity_id=self.partner.id, addressee_entity_id=self.observer.id, interlocus=InterlocusType.Public,
            timestamp_world_begin=self.now, timestamp_world_end=self.now,
            provenance=Provenance(source_kind=SourceKind.User)))
        batch.preserve_on_recovery()
        self.store.commit(batch)
        self.owner: EntityContextOwner = EntityContextOwner(self.store)
        self.key: EntityContextKey = EntityContextKey(subject_entity_id=self.partner.id,
            observer_entity_id=self.observer.id, scope=RecordScope())

    def commit(self, changes: WriteBatch) -> None:
        """Use actual durable production acceptance and recovery roots."""
        self.store.commit(changes)

    def stage(self, proposal: EntityContextProposal) -> object:
        """Run the real owner through a real locked batch and commit when staged."""
        with self.store.lock:
            changes = self.store.begin_batch(f"stage-{proposal.proposal_id}", embed=False)
            outcome = self.owner.stage_update(changes, proposal)
            if outcome.disposition == "staged":
                changes.preserve_on_recovery()
                self.store.commit(changes)
            return outcome

    def accepted_report(self, proposal_id: str, *, claim_key: str = "routine.workday",
                        statement: str | None = None, epistemic_kind: str = "report",
                        confidence: float = 0.9) -> EntityContextClaim:
        """Stage one accepted claim so a later revision has a real family."""
        view = self.store.snapshot(capture=False)
        evidence = EntityClaimEvidence(source=capture_source(view, self.message.id, current_head=True),
                                       exact_quote=self.message.content)
        claim = EntityContextClaim(layer="pattern", claim_key=claim_key,
            statement=statement if statement is not None else self.message.content,
            epistemic_kind=epistemic_kind, confidence=confidence, evidence=[evidence],
            observed_at_utc=self.message.timestamp_world_begin,
            applicable_from_utc=self.message.timestamp_world_begin)
        proposal = EntityContextProposal(proposal_id=proposal_id, key=self.key,
            parent_receipt_id=self.owner.current_head(view, self.key), producer="main",
            input_commit_ids=sorted(view.commit_ids), claims=[claim])
        outcome = self.stage(proposal)
        assert outcome.disposition == "staged", outcome
        assert outcome.staged_receipt_id is not None
        return active_claims(self.store.snapshot(capture=False), key=self.key)[-1]

    def finding(self, revision: PerspectiveRevision, *, finding_id: int = 700) -> Feature:
        """Commit one private finding with its stored revision, like the graph loop."""
        batch = self.store.begin_batch(f"finding-{finding_id}", embed=False)
        record = batch.create(Feature(content="The morning pattern is an estimate.",
            feature_type=FeatureType.Thought, entity_id=self.observer.id, scope=RecordScope(),
            provenance=Provenance(source_kind=SourceKind.Self),
            metadata={"perspective_revision": revision.model_dump(mode="json")}))
        batch.preserve_on_recovery()
        self.store.commit(batch)
        return record

    def draft(self, *, claim_key: str = "routine.morning", statement: str | None = None,
              epistemic_kind: str = "inference", confidence: float = 0.6,
              challenge: list[str] | None = None) -> EntityClaimDraft:
        return EntityClaimDraft(subject_entity_id=self.partner.id, source_ids=[self.message.id],
            exact_quote=None, layer="pattern", claim_key=claim_key,
            statement=statement if statement is not None else "Jesse is usually reachable after seven.",
            epistemic_kind=epistemic_kind, confidence=confidence,
            challenge_target_ids=challenge or [])


class PerspectiveUpdatesTests(PerspectiveUpdatesFixture):
    """The owner accepts a tom revision, or records an explicit rejection."""

    def test_valid_revision_stages_tom_proposal_over_the_message(self) -> None:
        self.accepted_report("morning-workday")
        revision = PerspectiveRevision(subject_entity_id=self.partner.id, claims=(self.draft(),))
        finding = self.finding(revision)
        updates = prepare_perspective_revisions(self.owner, self.store.snapshot(capture=False),
            [finding.id], observer_entity_id=self.observer.id)
        self.assertEqual(len(updates.proposals), 1)
        self.assertEqual(updates.rejected, [])
        outcome = self.stage(updates.proposals[0])
        self.assertEqual(outcome.disposition, "staged")
        assert outcome.staged_receipt_id is not None
        chain = receipt_chain(self.store.snapshot(capture=False), self.key)
        self.assertEqual(chain[-1].receipt.proposal.producer, "tom")
        claim = chain[-1].receipt.proposal.claims[0]
        self.assertEqual(claim.claim_key, "routine.morning")
        self.assertEqual(claim.evidence[0].source.record_id, self.message.id)

    def test_retracted_challenge_target_is_a_recorded_rejection(self) -> None:
        accepted = self.accepted_report("morning-workday")
        view = self.store.snapshot(capture=False)
        evidence = EntityClaimEvidence(source=capture_source(view, self.message.id, current_head=True),
                                       exact_quote=self.message.content)
        retraction = EntityContextProposal(proposal_id="retract-workday", key=self.key,
            parent_receipt_id=self.owner.current_head(view, self.key), producer="main",
            input_commit_ids=sorted(view.commit_ids), retract_claim_ids=[accepted.claim_id],
            correction_evidence=[evidence])
        outcome = self.stage(retraction)
        self.assertEqual(outcome.disposition, "staged")
        revision = PerspectiveRevision(subject_entity_id=self.partner.id,
            claims=(self.draft(challenge=[accepted.claim_id]),))
        finding = self.finding(revision)
        updates = prepare_perspective_revisions(self.owner, self.store.snapshot(capture=False),
            [finding.id], observer_entity_id=self.observer.id)
        self.assertEqual(updates.proposals, [])
        self.assertEqual(len(updates.rejected), 1)
        self.assertEqual(updates.rejected[0]["source_id"], finding.id)
        self.assertIn("existing claim ID", updates.rejected[0]["reason"])

    def test_staging_the_same_proposal_is_idempotent(self) -> None:
        self.accepted_report("morning-workday")
        revision = PerspectiveRevision(subject_entity_id=self.partner.id, claims=(self.draft(),))
        finding = self.finding(revision)
        updates = prepare_perspective_revisions(self.owner, self.store.snapshot(capture=False),
            [finding.id], observer_entity_id=self.observer.id)
        first = self.stage(updates.proposals[0])
        second = self.stage(updates.proposals[0])
        self.assertEqual(first.disposition, "staged")
        self.assertEqual(second.disposition, "staged")
        self.assertEqual(second.staged_receipt_id, first.staged_receipt_id)

    def test_repeated_inference_with_higher_confidence_is_rejected_by_the_owner(self) -> None:
        accepted = self.accepted_report("the-same", claim_key="routine.same",
            statement="Jesse prefers quiet mornings.", epistemic_kind="inference", confidence=0.5)
        self.assertIsNotNone(accepted.claim_id)
        revision = PerspectiveRevision(subject_entity_id=self.partner.id,
            claims=(self.draft(claim_key="routine.same", statement="Jesse prefers quiet mornings.",
                epistemic_kind="inference", confidence=0.9),))
        finding = self.finding(revision)
        updates = prepare_perspective_revisions(self.owner, self.store.snapshot(capture=False),
            [finding.id], observer_entity_id=self.observer.id)
        self.assertEqual(len(updates.proposals), 1)
        outcome = self.stage(updates.proposals[0])
        self.assertEqual(outcome.disposition, "rejected")
        self.assertEqual(outcome.reason, "unsupported_interpretation")
        self.assertIsNone(outcome.staged_receipt_id)


if __name__ == "__main__":
    unittest.main()
