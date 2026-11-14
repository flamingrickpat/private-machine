"""Rate two waiting displaced questions with the real embedding endpoint.

Prerequisite: the embedding endpoint configured in ``config.yaml`` must answer.
The check makes four embedding requests and no chat completion.

The store, the two inquiries, their displacement, and the private owner are
real. The source texts and the cue texts are synthetic inputs. The check
measures one thing a deterministic test cannot: whether the configured
embedding model rates related text as similar enough that a displaced question
comes into focus from something merely related. The outcome of the weighted
draw is not asserted, because it legitimately offers the weaker question
sometimes; the measured ratings are.
"""
from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest

from pydantic import BaseModel, ConfigDict, Field

from pm import vclock
from pm.llm.config import load_config
from pm.memory.embeddings import EmbeddingService
from pm.memory.store import MemoryStore
from pm.memory.write_batch import WriteBatch
from pm.records.base import Provenance, RecordScope
from pm.records.cycle_state import PersonaConfig
from pm.records.enums import EntityClass, FeatureType, InterlocusType, SourceKind
from pm.records.memory_records import Entity, Feature
from pm.records.mental_state import FullMentalState
from pm.utils.operating_system import OperatingSystem
from workflows.cognitive_control.schemas import ControlConfig
from workflows.inner_thought.graph_execution import DisplacedCandidate, displaced_candidate
from workflows.inner_thought.graph_host import PrivateInquiryOwner, displaced_inquiries
from workflows.inner_thought.inquiry import InquiryRecord, stage_attention
from workflows.inner_thought.nominations import capture_inquiry_grant, stage_nominations
from workflows.inner_thought.request_views import RecentThoughtView
from workflows.inner_thought.schemas import (
    InnerRequest, InquiryGrant, RetainedOfferSelection, RetainedQuestionRating, ThoughtAttentionDecision,
    ThoughtCandidate, ThoughtJudgments, ThoughtNomination,
)


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = ROOT / ".runtime/beta-b05/offer-salience"

#: A related cue must rate this much above an unrelated one to count as a cue.
MINIMUM_SIMILARITY_MARGIN = 0.1


class OpportunityReceipt(BaseModel):
    """One measured opportunity: what the mind held and how each question rated."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Which situation this opportunity states.")
    cue_text: str = Field(description="The one cue text the private request held.")
    selection: RetainedOfferSelection = Field(description="The full offer receipt the coordinator produced.")


class SalienceEvidence(BaseModel):
    """What this live check ran against and what it measured."""

    model_config = ConfigDict(extra="forbid")

    started_at_utc: datetime
    revision: str = Field(description="Git revision, marked dirty when the tree has local changes.")
    embedding_model: str
    embedding_endpoint: str = Field(description="Endpoint base URL, without credentials.")
    embedding_requests: int = Field(ge=0, description="Requests this check sent through the embedding service.")
    sketch_receipt_id: int
    tap_receipt_id: int
    opportunities: tuple[OpportunityReceipt, ...]


class DisplacedOfferSalienceTests(unittest.TestCase):
    """Something related, or a held source, brings the matching displaced question into focus."""

    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store: MemoryStore = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"),
                                              checkpoint=Path(directory.name) / "salience.pickle")
        self.now: datetime = datetime(2026, 11, 14, 9, 0)
        previous_clock = vclock.active()
        self.addCleanup(vclock.install, previous_clock)
        vclock.install(vclock.VirtualClock(self.now))
        anchors = self.store.begin_batch("anchors", embed=False)
        self.observer: Entity = anchors.create(Entity(content="Spamuel", aliases=[], entity_class=EntityClass.Self))
        self.partner: Entity = anchors.create(Entity(content="Jesse", aliases=[], entity_class=EntityClass.Human))
        self.sketch_source: Feature = self.report(anchors,
            "I left the watercolour sketch of the harbour unfinished again.")
        self.tap_source: Feature = self.report(anchors, "The kitchen tap is dripping again.")
        self.commit_root(anchors)
        self.owner: PrivateInquiryOwner = PrivateInquiryOwner(self.store, commit=self.commit, cancel=Event(),
                                                              limits=ControlConfig())
        config = load_config(ROOT / "config.yaml")
        self.embeddings: EmbeddingService = EmbeddingService(config.embedding, live_test=True)
        self.embedding_requests: int = 0

    def report(self, batch: WriteBatch, text: str) -> Feature:
        """Stage one synthetic outside report addressed to the observer."""
        return batch.create(Feature(content=text, causal=True, feature_type=FeatureType.Dialogue,
            entity_id=self.partner.id, addressee_entity_id=self.observer.id, interlocus=InterlocusType.Public,
            provenance=Provenance(source_kind=SourceKind.User)))

    def commit(self, batch: WriteBatch) -> None:
        """Use actual durable acceptance; the owner marks its own recovery roots."""
        self.store.commit(batch)

    def commit_root(self, batch: WriteBatch) -> None:
        """Persist an append-only test input as a clean recovery root."""
        batch.preserve_on_recovery()
        self.store.commit(batch)

    def capture(self, critical_epoch: int) -> InquiryGrant:
        """Capture the grant a private opportunity holds at this accepted Critical epoch."""
        source_ids = [self.sketch_source.id, self.tap_source.id]
        return capture_inquiry_grant(self.store.snapshot(capture=False), observer_entity_id=self.observer.id,
            scope=RecordScope(), source_ids=source_ids, governing_ids=source_ids, critical_epoch=critical_epoch)

    def admit(self, episode_id: str, *, text: str, question: str, source: Feature, critical_epoch: int) -> InquiryRecord:
        """Admit one synthetic nomination into the private slot through its storage owner."""
        grant = self.capture(critical_epoch)
        judgments = ThoughtJudgments(interestingness=0.6, character_relevance=0.6, plausibility=0.3,
            consequence=0.4, desirability=0.4, action_suitability=0.2, reason="A synthetic unresolved question.")
        batch = self.store.begin_batch("nomination-" + episode_id, embed=False)
        records = stage_nominations(batch, [ThoughtNomination(text=text, question=question,
            evidence_ids=(source.id,), reality="inferred", judgments=judgments,
            suggested_operation="future_projection")], grant=grant, batch_id=episode_id)
        self.commit_root(batch)
        attention = self.store.begin_batch("attention-" + episode_id, embed=False)
        selected = stage_attention(attention, [records[0].record.id], ThoughtAttentionDecision(
            nomination_id=records[0].record.id, disposition="select", reason="Inspect this private question."),
            episode_id=episode_id, now_utc=self.now)
        self.commit_root(attention)
        assert selected.inquiry is not None
        return selected.inquiry

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed through the real configured endpoint, exactly as the private host does."""
        self.embedding_requests += 1
        return asyncio.run(self.embeddings.embed_many(list(texts)))

    def offer(self, request: InnerRequest, *, critical_epoch: int, seed: str) -> DisplacedCandidate:
        """Ask the real coordinator which displaced question this opportunity offers."""
        candidate = displaced_candidate(self.owner, observer_entity_id=self.observer.id,
            grant=self.capture(critical_epoch), request=request, mental_state=FullMentalState(),
            selection_seed=seed, embed_texts=self.embed_texts)
        assert candidate is not None
        return candidate

    def rating(self, selection: RetainedOfferSelection, receipt_id: int) -> RetainedQuestionRating:
        """Read one question's rating from an offer receipt."""
        return next(item for item in selection.ratings if item.receipt_id == receipt_id)

    def test_related_text_and_a_held_source_each_cue_the_matching_question(self) -> None:
        started = datetime.now(timezone.utc)
        self.admit("sketch", text="The harbour sketch keeps being abandoned halfway.",
            question="Why does Jesse leave the watercolour sketch unfinished?", source=self.sketch_source,
            critical_epoch=2)
        sketch = self.owner.displace_superseded(observer_entity_id=self.observer.id, critical_epoch=3)[0]
        self.admit("tap", text="The dripping tap might point to a larger plumbing problem.",
            question="Is the dripping kitchen tap a sign of a bigger leak?", source=self.tap_source,
            critical_epoch=3)
        tap = self.owner.displace_superseded(observer_entity_id=self.observer.id, critical_epoch=4)[0]
        waiting = displaced_inquiries(self.store.snapshot(capture=False), observer_entity_id=self.observer.id)
        self.assertEqual({head.record.id for head in waiting}, {sketch.record.id, tap.record.id})

        # A new outside remark that is a source of neither question.
        heard = self.store.begin_batch("heard-remark", embed=False)
        remark = self.report(heard, "A pipe under the sink has started leaking water onto the floor.")
        self.commit_root(heard)

        # A private thought of the observer that is no node of either closed inquiry.
        thought_text = "The colours in that half-painted harbour picture never felt settled to me."
        musing = self.store.begin_batch("adjacent-thought", embed=False)
        thought = musing.create(Feature(content=thought_text, feature_type=FeatureType.InnerMonologue,
            entity_id=self.observer.id, interlocus=InterlocusType.PrivateInternal,
            provenance=Provenance(source_kind=SourceKind.Self)))
        self.commit_root(musing)

        heard_text = remark.content
        opportunities = (
            ("heard something related", heard_text,
             InnerRequest(candidates=[ThoughtCandidate(knoxel_id=remark.id, text=heard_text)])),
            ("adjacent private thought", thought_text,
             InnerRequest(recent_thoughts=[RecentThoughtView(knoxel_id=thought.id, text=thought_text)])),
            ("original source held again", self.tap_source.content,
             InnerRequest(candidates=[ThoughtCandidate(knoxel_id=self.tap_source.id,
                                                       text=self.tap_source.content)])),
        )
        receipts: list[OpportunityReceipt] = []
        selections: dict[str, RetainedOfferSelection] = {}
        for name, cue_text, request in opportunities:
            candidate = self.offer(request, critical_epoch=4, seed="live:" + name)
            selections[name] = candidate.selection
            receipts.append(OpportunityReceipt(name=name, cue_text=cue_text, selection=candidate.selection))
        repeated = self.offer(opportunities[0][2], critical_epoch=4, seed="live:heard something related")

        directory = EVIDENCE_ROOT / started.strftime("%Y%m%dT%H%M%SZ")
        directory.mkdir(parents=True, exist_ok=False)
        head = OperatingSystem.output_text(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).strip()
        dirty = OperatingSystem.output_text(["git", "status", "--porcelain"], cwd=ROOT).strip()
        evidence = SalienceEvidence(started_at_utc=started, revision=head + ("+dirty" if dirty else ""),
            embedding_model=self.embeddings.config.model, embedding_endpoint=self.embeddings.config.base_url,
            embedding_requests=self.embedding_requests, sketch_receipt_id=sketch.record.id,
            tap_receipt_id=tap.record.id, opportunities=tuple(receipts))
        (directory / "result.json").write_text(evidence.model_dump_json(indent=2), encoding="utf-8")
        print(f"Displaced offer salience evidence: {directory}", flush=True)

        heard_selection = selections["heard something related"]
        heard_tap = self.rating(heard_selection, tap.record.id)
        heard_sketch = self.rating(heard_selection, sketch.record.id)
        self.assertEqual((heard_tap.source_overlap, heard_sketch.source_overlap), (0.0, 0.0),
                         "The remark is a new record, so it shares no source with either question.")
        self.assertGreater(heard_tap.similarity, heard_sketch.similarity + MINIMUM_SIMILARITY_MARGIN,
                           "A remark about a leaking pipe must cue the tap question by meaning alone.")
        self.assertEqual(heard_tap.most_similar_cue_id, remark.id)

        thought_selection = selections["adjacent private thought"]
        self.assertGreater(self.rating(thought_selection, sketch.record.id).similarity,
                           self.rating(thought_selection, tap.record.id).similarity + MINIMUM_SIMILARITY_MARGIN,
                           "A neighbouring thought about the painting must cue the sketch question.")

        held_selection = selections["original source held again"]
        held_tap = self.rating(held_selection, tap.record.id)
        self.assertEqual(held_tap.shared_source_ids, (self.tap_source.id,))
        self.assertEqual(held_tap.source_overlap, 1.0)
        self.assertEqual(self.rating(held_selection, sketch.record.id).source_overlap, 0.0)
        self.assertGreater(held_tap.relevance, self.rating(held_selection, sketch.record.id).relevance)

        for selection in selections.values():
            self.assertEqual(selection.mode, "weighted_random")
            self.assertIn(selection.offered_receipt_id, {sketch.record.id, tap.record.id})
        self.assertEqual(repeated.selection.offered_receipt_id, heard_selection.offered_receipt_id,
                         "The same opportunity seed must offer the same question again.")
        self.assertEqual(self.embedding_requests, 4)


if __name__ == "__main__":
    unittest.main()
