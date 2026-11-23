"""One claim mentioned in three commit groups keeps one live semantic record.

The real store, the real imported events, the real commit path, and the real
facts staging run. Only the model transport is unused: staging sends no request,
so this test needs no endpoint. A real :class:`LazyStageModel` is built so
:class:`_Run` holds the contract it always holds, and nothing asks it anything.

A shared ``(subject, essence)`` claim is appended to by every later commit group,
and :meth:`WriteBatch.replace` gives each appended version a new id while guarding
the id it replaced. The dedup map must follow that new id. Left on the guarded id,
the next group appends to a record the store no longer holds as a head, and its
commit raises ``GhostConflict: Stale memory revisions``. One real run died that way
with ``[9625]`` on 2026-11-22 after an hour of fact extraction.

The run drives :meth:`_Run._stage_facts` directly because that is the staging step.
The model answer that normally feeds it is staged in by hand, and the whole
:meth:`_Run.facts` stage with a live model is exercised by the real run it fixes.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
import unittest

from pm.app.chat_import import ChatRow, group_exchanges, import_exchanges
from pm.llm.activation import ModelActivation
from pm.llm.config import Connection, EmbeddingConfig
from pm.llm.stage_model import LazyStageModel
from pm.memory.embeddings import EmbeddingService
from pm.memory.reconsolidation import reconsolidation_due, unclustered_events
from pm.memory.reconsolidation.model_calls import ModelCalls
from pm.memory.reconsolidation.pipeline import FiledFact, ReconsolidationProgress, _Run
from pm.memory.reconsolidation.schemas import (
    FactClassificationValidation, FactFiling, ReconsolidationConfig, ReconsolidationReport, load_prompts)
from pm.memory.store import MemoryStore
from pm.records.cycle_state import PersonaConfig
from pm.records.enums import ClusterType, FeatureType
from pm.records.memory_records import DeclarativeFactRecord, Feature, MemoryClusterRecord, SemanticRecord
from pm.records.mental_state import PersonalityProfile
from pm.records.taxonomy import MemoryTaxonomy, PmCategory

CLAIM = "Bob drinks coffee at his desk every morning."
ESSENCE = "Bob drinks coffee at his desk"
GROUPS = 3
START = datetime(2026, 11, 16, 9, 0)


def _rows() -> list[ChatRow]:
    """Return a short morning-routine exchange per day."""
    rows: list[ChatRow] = []
    number = 0
    for day in range(GROUPS * 2):
        for step in range(2):
            number += 1
            rows.append(ChatRow(row=number, speaker="user" if step == 0 else "companion",
                                message=f"morning chat {number} about the coffee Bob drinks at his desk",
                                timestamp=START + timedelta(days=day, minutes=2 * step)))
    return rows


class FactDedupChainTests(unittest.TestCase):
    """Three clusters, one shared claim, three commit groups."""

    def setUp(self) -> None:
        self.store = MemoryStore(PersonaConfig(user_name="Bob", companion_name="Ava",
                                               character_card_story="Ava."))
        self.now = datetime(2026, 11, 22, 7, 0, tzinfo=UTC).replace(tzinfo=None)
        import_exchanges(self.store, group_exchanges(_rows()), import_id="dedup-chain", source_sha256="abc",
                         personality=PersonalityProfile(), imported_at=self.now)
        self.config = ReconsolidationConfig()
        self.events = unclustered_events(self.store, self.config)
        self.bob_id = self.store.get_entity_id("Bob")
        assert self.bob_id is not None, "the imported exchange must name the user"
        self.run = _Run(
            store=self.store,
            calls=ModelCalls(LazyStageModel(Connection(base_url="http://127.0.0.1:9/v1", model="scripted",
                                                       context_window=8192), ModelActivation())),
            embeddings=EmbeddingService(EmbeddingConfig(base_url="http://127.0.0.1:9/v1",
                                                        model="scripted-embedding", dimensions=1024)),
            config=self.config,
            prompts=load_prompts(),
            report=ReconsolidationReport(run_id="dedup-chain"),
            now=self.now,
            cancel=None,
            progress=ReconsolidationProgress())
        self.clusters = [self._cluster(event, index) for index, event in enumerate(self.events[:GROUPS])]
        self.dedup_key = MemoryTaxonomy.dedup_key(self.bob_id, ESSENCE)

    def _cluster(self, event: Feature, index: int) -> MemoryClusterRecord:
        """Store one topical cluster that still awaits fact extraction.

        Each setup batch needs its own commit id: the store replays a commit id it
        has already applied, and a repeated id would drop the later cluster.
        """
        batch = self.run.begin_batch(f"setup-cluster-{index}")
        # ``included_event_ids`` is stored as a comma-separated string.
        record = batch.create(MemoryClusterRecord(cluster_type=ClusterType.Topical, content="Bob's mornings",
                                                 level=1, included_event_ids=str(event.id), facts_extracted=False))
        self.run.finish_batch(batch)
        return record

    def _filed_fact(self, evidence: Feature) -> FiledFact:
        """Build one fact filing the way the pipeline builds a model answer."""
        answer = {"index": 0, "about": "Bob", "record_type": "fact", "epistemic_status": "observed",
                  "confidence": 0.9, "object_kind": "event", "object_value": CLAIM,
                  "topic_card": "Bob's mornings", "essence": ESSENCE,
                  "category": [PmCategory.PEOPLE_PREFERENCES]}
        context = FactClassificationValidation(fact_count=1, allowed_subjects=("Bob",))
        classification = FactFiling.model_validate(answer, context=context)
        return FiledFact(text=CLAIM, filing=classification, sources=[evidence])

    def _stage_one_group(self, group_index: int, existing: dict[str, int]) -> None:
        """Commit one group's mention of the shared claim."""
        subjects = {"Bob": self.bob_id}
        batch = self.run.begin_batch(f"facts-{group_index}")
        self.run._stage_facts(batch, [self.clusters[group_index]],
                              [[self._filed_fact(self.events[group_index])]], subjects, existing)
        self.run.finish_batch(batch)

    def test_each_group_extends_the_current_head(self) -> None:
        """Every group after the first must supersede the live head, not a guarded id."""
        existing: dict[str, int] = {}
        head_id_by_group: list[int] = []
        for group_index in range(GROUPS):
            self._stage_one_group(group_index, existing)
            head_id_by_group.append(existing[self.dedup_key])

        live_semantics = [record for record in self.store.all_knoxels.values() if isinstance(record, SemanticRecord)]
        self.assertEqual(len(live_semantics), 1, "one claim must keep exactly one live semantic record")
        head = live_semantics[0]
        self.assertEqual(head.subject_entity_id, self.bob_id, "the claim must stay attributed to its subject")
        self.assertEqual(head.essence, ESSENCE, "the claim must keep the identity the dedup key is built from")
        self.assertEqual(existing[self.dedup_key], head.id,
                         "the dedup map must point at the head the store still holds, not a guarded id")

        facts = sorted((record for record in self.store.all_knoxels.values()
                        if isinstance(record, DeclarativeFactRecord)), key=lambda record: record.id)
        self.assertEqual(len(facts), GROUPS, "every group's mention must keep its own fact record")
        citations = [fact.semantic_record_id for fact in facts]
        self.assertEqual(citations, head_id_by_group, "each group must cite the head its own commit left live")
        self.assertEqual(citations[-1], head.id, "the newest mention must cite the live head")
        self.assertEqual(head_id_by_group, sorted(set(head_id_by_group)), "each append must move the head to a new id")

    def test_the_live_head_holds_every_mention(self) -> None:
        """The supersession chain must not drop the earlier facts' evidence."""
        existing: dict[str, int] = {}
        for group_index in range(GROUPS):
            self._stage_one_group(group_index, existing)

        head_ids = [record.id for record in self.store.all_knoxels.values() if isinstance(record, SemanticRecord)]
        self.assertEqual(len(head_ids), 1)
        head = self.store.all_knoxels[head_ids[0]]
        self.assertEqual(head.content, CLAIM, "the one live claim must keep the wording")
        self.assertEqual(sorted(head.source_knoxel_ids), sorted(self.run.report.fact_ids),
                         "the one live claim must carry the fact ids of all three mentions")

    def test_the_setup_leaves_real_work_for_the_stages(self) -> None:
        """The fixture must be a store the pipeline itself would reconsolidate."""
        self.assertEqual(len(self.events), 12, "each imported message must contribute its own story event")
        self.assertGreaterEqual(len(self.events), self.config.min_events)
        self.assertTrue(all(isinstance(event, Feature) and event.feature_type in self.config.story_feature_types
                            and event.causal for event in self.events))
        self.assertFalse(reconsolidation_due(self.store, self.config),
                         f"a store this small stays under the run trigger of {self.config.trigger_events} events; "
                         "a run reaches these events only because --reconsolidate forces one")


if __name__ == "__main__":
    unittest.main()
