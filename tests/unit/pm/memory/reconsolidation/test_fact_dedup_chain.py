"""Exercise episode filing, supersession and publication against a real recovered Ghost."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest

from pm.app.chat_import import ChatRow, group_exchanges, import_exchanges
from pm.memory.reconsolidation import reconsolidation_due, unclustered_events
from pm.memory.reconsolidation.facts import EpisodeFacts, FiledFact
from pm.memory.reconsolidation.publication import MemoryPreparation, MemoryUnitReceipt, PreparedMemoryUnit
from pm.memory.reconsolidation.schemas import FactClassificationValidation, FactFiling
from pm.memory.retrieval import candidate, scoped_records
from pm.memory.store import MemoryStore
from pm.records.base import Provenance
from pm.records.cycle_state import PersonaConfig
from pm.records.enums import ClusterType, SourceKind
from pm.records.memory_records import DeclarativeFactRecord, MemoryClusterRecord, SemanticRecord
from pm.records.mental_state import PersonalityProfile
from pm.records.taxonomy import PmCategory

CLAIM = "Bob drinks coffee at his desk every morning."
ESSENCE = "Bob drinks coffee at his desk"
START = datetime(2026, 11, 16, 9)


class FactDedupChainTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)  # pm-lint: allow PM003 -- unittest owns temporary file cleanup
        self.path = Path(temporary.name) / "ghost.pickle"
        self.store = MemoryStore(PersonaConfig(user_name="Bob", companion_name="Ava", character_card_story="Ava."),
                                 checkpoint=self.path)
        rows: list[ChatRow] = []
        for day in range(3):
            for minute, speaker in enumerate(("user", "companion")):
                rows.append(ChatRow(row=len(rows) + 1, speaker=speaker, message=CLAIM,
                                    timestamp=START + timedelta(days=day, minutes=minute)))
        import_exchanges(self.store, group_exchanges(rows), import_id="three-mornings", source_sha256="abc",
                         personality=PersonalityProfile.model_validate_json("{}"), imported_at=START + timedelta(days=3))
        self.store.save()
        self.view = self.store.recovery_snapshot(capture=False)
        bob_id = self.store.get_entity_id("Bob")
        assert bob_id is not None
        self.bob: int = bob_id
        self.events = [row for row in unclustered_events(self.store) if row.entity_id == self.bob]
        self.stop = Event()
        self.preparation = MemoryPreparation(self.store, self.view, "three-mornings", self.stop)
        self.filer = EpisodeFacts(self.view, self.store.get_entity_id("Ava"), START + timedelta(days=3))

    def prepare_episode(self, index: int, wording: str = CLAIM, essence: str = ESSENCE,
                        canonical_id: int | None = None) -> PreparedMemoryUnit:
        evidence = self.events[index]
        who = f"entity:{self.bob}"
        validation = FactClassificationValidation(fact_count=1, allowed_subjects=(who,))
        # This is a typed input to deterministic filing, independent of a model response.
        filing = FactFiling.model_validate({"about": who, "index": 1, "essence": essence, "canonical_id": canonical_id,  # pm-lint: allow PM005 -- typed model answer validation
            "category": [PmCategory.PEOPLE_PREFERENCES], "epistemic_status": "asserted"}, context=validation)
        facts: list[FiledFact] = self.filer.attribute(wording, filing, (evidence,), (self.bob,))
        receipt = MemoryUnitReceipt(run_id="three-mornings", unit=f"topic-{index}")
        batch = self.preparation.begin(receipt.unit, {evidence.id} | self.filer.prerequisites(facts))
        cluster = batch.create(MemoryClusterRecord(level=100, cluster_type=ClusterType.Topical,
            content=CLAIM, included_event_ids=str(evidence.id), facts_extracted=True))
        receipt.topical_ids.append(cluster.id)
        self.filer.stage(batch, cluster, facts, receipt, Provenance(source_kind=SourceKind.System))
        return self.preparation.finish(batch, receipt)

    def test_three_unpublished_units_extend_the_same_head_after_restart(self) -> None:
        units = [self.prepare_episode(index) for index in range(3)]
        self.assertFalse(any(isinstance(row, SemanticRecord) for row in self.store.all_knoxels.values()))
        self.stop.set()
        for unit in units:
            self.assertTrue(self.preparation.publish(unit))
        self.store.save()
        reopened = MemoryStore.load(self.path)
        facts = [row for row in reopened.all_knoxels.values() if isinstance(row, DeclarativeFactRecord)]
        heads = [row for row in reopened.all_knoxels.values() if isinstance(row, SemanticRecord)]
        self.assertEqual(len(heads), 1)
        self.assertEqual(len(facts), 3)
        self.assertEqual(heads[0].source_knoxel_ids, sorted(row.id for row in facts))
        self.assertEqual(heads[0].subject_entity_id, self.bob)
        self.assertEqual({row.lineage.source_event_ids[0] for row in facts if row.lineage}, {row.id for row in self.events})
        self.assertEqual(len({row.semantic_record_id for row in facts}), 3)
        view = reopened.recovery_snapshot()
        visible, _ = scoped_records(view)
        self.assertIn(heads[0].id, visible)
        self.assertTrue(all(row.id in visible for row in facts))
        historical_heads = set(view.historical_records)
        self.assertTrue(historical_heads.isdisjoint(visible))
        citations = candidate(view, heads[0].id, heads[0]).source_ids
        self.assertEqual(set(citations), {row.id for row in facts} | {row.id for row in self.events})
        self.assertTrue(historical_heads.isdisjoint(citations))

    def test_recovery_keeps_only_the_published_episode(self) -> None:
        first, second = self.prepare_episode(0), self.prepare_episode(1)
        self.assertTrue(self.preparation.publish(first))
        self.store.save()
        reopened = MemoryStore.load(self.path)
        self.assertIn(first.batch.commit_id, reopened.consolidations)
        self.assertNotIn(second.batch.commit_id, reopened.consolidations)
        uncovered = {row.id for row in unclustered_events(reopened)}
        self.assertNotIn(self.events[0].id, uncovered)
        self.assertIn(self.events[1].id, uncovered)
        self.assertFalse(reconsolidation_due(reopened))

    def test_paraphrase_extends_dated_support_only_for_the_same_source_authority(self) -> None:
        first = self.prepare_episode(0)
        canonical_id = first.receipt.semantic_ids[0]
        paraphrase = "Every morning Bob has coffee at his desk."
        next_event = self.events[1]
        subjects = self.filer.subjects((next_event,), (self.bob,))
        options = self.filer.canonical_options(paraphrase, (next_event,), subjects, (self.bob,))
        self.assertEqual([option.head_id for option in options], [canonical_id])
        opposite = self.filer.canonical_options("Bob never drinks coffee at his desk.", (next_event,), subjects, (self.bob,))
        self.assertEqual(opposite, ())
        other = next(row for row in unclustered_events(self.store) if row.entity_id != self.bob)
        self.assertEqual(self.filer.canonical_options(paraphrase, (other,), subjects, (self.bob,)), ())
        second = self.prepare_episode(1, paraphrase, "Coffee at work is Bob's regular morning practice.", canonical_id)
        self.assertTrue(self.preparation.publish(first))
        self.assertTrue(self.preparation.publish(second))
        heads = [row for row in self.store.all_knoxels.values() if isinstance(row, SemanticRecord)]
        self.assertEqual(len(heads), 1)
        self.assertEqual(heads[0].timestamp_world_begin, self.events[0].timestamp_world_begin)
        self.assertEqual(heads[0].timestamp_world_end, next_event.timestamp_world_end)
        self.assertEqual(heads[0].source_knoxel_ids, first.receipt.fact_ids + second.receipt.fact_ids)
        self.assertIn(paraphrase, heads[0].source_texts)

    def test_source_changed_before_publication_invalidates_dependents(self) -> None:
        first, dependent = self.prepare_episode(0), self.prepare_episode(1)
        correction = self.store.begin_batch("correct-source")
        original = self.events[0].copy_for_edit()
        original.content = "Bob has corrected this entry."
        correction.replace(original)
        self.store.commit(correction)
        self.assertFalse(self.preparation.publish(first))
        self.assertFalse(self.preparation.publish(dependent))
        self.assertFalse(any(isinstance(row, SemanticRecord) for row in self.store.all_knoxels.values()))
