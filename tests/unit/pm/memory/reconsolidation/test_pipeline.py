"""Real source selection and entity filing at a frozen Ghost frontier, without model inference."""
from __future__ import annotations

from datetime import datetime, timedelta
import unittest

from pm.app.character_cli import build_models
from pm.memory.cluster_membership import ClusterMembership
from pm.memory.dependencies import capture_source
from pm.memory.entity_history import EntityClaimEvidence, EntityContextClaim, EntityContextKey, EntityContextProposal
from pm.memory.reconsolidation.facts import EpisodeFacts
from pm.memory.reconsolidation.membership import eligible_external_features
from pm.memory.reconsolidation.pipeline import ReconsolidationPass, reconsolidate
from pm.memory.reconsolidation.schemas import FactClassificationValidation, FactFiling, ReconsolidationConfig
from pm.memory.store import MemoryStore
from pm.records.base import Provenance, RecordScope
from pm.records.cycle_state import PersonaConfig
from pm.records.enums import ClusterType, EntityClass, FeatureType, FrameEnum, InterlocusType, SourceKind
from pm.records.memory_records import Entity, Feature, MemoryClusterRecord
from pm.records.taxonomy import EpistemicStatus, PmCategory
from workflows.entity_context.owner import EntityContextOwner

NOW = datetime(2026, 11, 20, 9)


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"))
        batch = self.store.begin_batch("identities")
        self.observer = batch.create(Entity(content="Spamuel", entity_class=EntityClass.Self, aliases=[]))
        self.peer = batch.create(Entity(content="Mara", entity_class=EntityClass.AI, aliases=[]))
        self.partner = batch.create(Entity(content="Jesse", entity_class=EntityClass.Human, aliases=[]))
        self.store.commit(batch)

    def add_feature(self, text: str, when: datetime, speaker: Entity, recipient: Entity,
                    kind: FeatureType = FeatureType.Dialogue, scope: RecordScope = RecordScope(),
                    retained: bool = False) -> Feature:
        batch = self.store.begin_batch(f"source-{self.store.current_knoxel_id + 1}")
        row = Feature(content=text, feature_type=kind, entity_id=speaker.id, addressee_entity_id=recipient.id,
                      causal=not retained, scope=scope, timestamp_world_begin=when, timestamp_world_end=when,
                      interlocus=InterlocusType.Public if kind == FeatureType.Dialogue else InterlocusType.PrivateInternal,
                      provenance=Provenance(source_kind=SourceKind.Agent))
        if retained:
            row.metadata["private_inquiry_finding"] = True
        batch.create(row)
        self.store.commit(batch)
        return row

    def test_episode_selection_separates_participation_citation_and_membership(self) -> None:
        old = NOW - timedelta(days=2)
        speech = self.add_feature("Talk to me after lunch.", old, self.peer, self.partner)
        routine = self.add_feature("Maybe later.", old + timedelta(minutes=1), self.peer, self.peer, FeatureType.Thought)
        finding = self.add_feature("I worked out where the keys went.", old + timedelta(hours=9), self.observer,
                                   self.observer, FeatureType.InnerMonologue, retained=True)
        self.add_feature("A future dialogue.", NOW + timedelta(hours=1), self.observer, self.partner)
        self.store.accept_import()
        models = build_models("config.yaml")
        work = ReconsolidationPass(self.store, complete=models.stage, embeddings=models.embeddings, force=True, now=NOW)
        episodes = list(work.episode_sources(work.frontier.features))
        self.assertEqual(len(episodes), 2)
        self.assertEqual(episodes[0].members, (speech, routine))
        self.assertEqual(episodes[0].evidence, (speech,))
        self.assertEqual(set(episodes[0].participant_ids), {self.partner.id, self.peer.id})
        self.assertEqual(episodes[1].evidence, (finding,))
        self.assertEqual(episodes[1].participant_ids, (self.observer.id,))

    def test_scope_mismatch_and_future_inputs_never_hide_an_external_event(self) -> None:
        feature = self.add_feature("Our boat is blue.", NOW - timedelta(days=1), self.partner, self.observer)
        future = self.add_feature("Tomorrow's news.", NOW + timedelta(days=1), self.partner, self.observer)
        batch = self.store.begin_batch("malformed-legacy-cluster")
        legacy = batch.create(MemoryClusterRecord(content="Imported summary.", level=100, cluster_type=ClusterType.Topical,
            scope=RecordScope(frame=FrameEnum.Meta), included_event_ids=str(feature.id)))
        self.store.commit(batch)
        self.store.accept_import()
        view = self.store.recovery_snapshot(capture=False)
        due = eligible_external_features([feature, future], [legacy], NOW, ReconsolidationConfig(), ClusterMembership(view.all_knoxels))
        self.assertEqual([item.id for item in due], [feature.id])

    def test_newest_open_episode_makes_no_model_call(self) -> None:
        self.add_feature("This conversation is ongoing.", NOW, self.partner, self.observer)
        self.store.accept_import()
        models = build_models("config.yaml")
        report = reconsolidate(self.store, complete=models.stage, embeddings=models.embeddings, force=True, now=NOW)
        self.assertEqual(report.created_ids, [])
        self.assertEqual(report.model_calls, 0)
        self.assertIn(f"reconsolidation:{report.run_id}", self.store.consolidations)

    def test_two_claims_and_an_independent_report_can_share_one_line(self) -> None:
        source = self.add_feature("Jesse likes tea, plays guitar and starts at four.", NOW, self.peer, self.observer)
        self.store.accept_import()
        view = self.store.recovery_snapshot(capture=False)
        key = EntityContextKey(subject_entity_id=self.partner.id, observer_entity_id=self.observer.id, scope=source.scope)
        claims: list[EntityContextClaim] = []
        for name, quote in (("drink", "likes tea"), ("instrument", "plays guitar")):
            evidence = EntityClaimEvidence(source=capture_source(view, source.id, current_head=True), exact_quote=quote)
            claims.append(EntityContextClaim(layer="background", claim_key=name, statement=f"Mara said Jesse {quote}.",
                epistemic_kind="report", confidence=.8, evidence=[evidence], observed_at_utc=NOW, applicable_from_utc=NOW))
        owner = EntityContextOwner(self.store)
        append = self.store.begin_batch("two-claims")
        proposal = EntityContextProposal(proposal_id="preferences", key=key, parent_receipt_id=None, producer="immediate", claims=claims)
        outcome = owner.stage_update(append, proposal)
        self.assertEqual(outcome.disposition, "staged")
        append.preserve_on_recovery()
        self.store.commit(append)
        view = self.store.recovery_snapshot(capture=False)
        facts = EpisodeFacts(view, self.observer.id, NOW)
        options = facts.claim_options((source,))
        self.assertEqual({item.statement for item in options}, {item.statement for item in claims})
        subject = f"entity:{self.partner.id}"
        validation = FactClassificationValidation(fact_count=1, allowed_subjects=(subject,), claim_options=(options,))
        filing = FactFiling.model_validate({"about": subject, "essence": "Jesse starts at four.",  # pm-lint: allow PM005 -- model response wire validation
            "category": [PmCategory.PEOPLE_PREFERENCES]}, context=validation)
        independent = facts.attribute("Jesse starts at four.", filing, (source,), (self.peer.id, self.observer.id))
        self.assertEqual(len(independent), 1)
        self.assertEqual(independent[0].text, "Mara reported: Jesse starts at four.")
        self.assertEqual(independent[0].filing.epistemic_status, EpistemicStatus.ReportedByOther)
        self.assertIsNone(independent[0].claim)
        filing.entity_claim = options[0].claim_id
        matched = facts.attribute("Jesse enjoys tea.", filing, (source,), (self.peer.id, self.observer.id))
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0].text, claims[0].statement)
        self.assertEqual(matched[0].subject_id, self.partner.id)
        self.assertEqual(matched[0].sources, (source,))
        self.assertEqual(matched[0].claim.receipt_id if matched[0].claim else None, outcome.staged_receipt_id)
