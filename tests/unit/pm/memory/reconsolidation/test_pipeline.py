"""A reconsolidation run over a small imported store, with a scripted model.

The real store, the real commit path, and the real stages run. Only the
model and the embeddings are replaced: the scripted model answers by the
task phrase of each prompt, and the embeddings hash topic words into small
vectors. A second run over the same store must make no model call.
"""
import re
import unittest
from datetime import datetime, timedelta

from pm.app.chat_import import ChatRow, add_transfer_records, group_exchanges, import_exchanges
from pm.context_render.turns import Role
from pm.memory.reconsolidation import reconsolidate, unclustered_events
from pm.memory.reconsolidation.pipeline import OPERATION_NARRATIVE
from pm.memory.store import MemoryStore
from pm.llm.prompt_encoding import normalize_turns
from pm.records.cycle_state import PersonaConfig
from pm.records.enums import ClusterType
from pm.records.memory_records import DeclarativeFactRecord, MemoryClusterRecord, Narrative, SemanticRecord
from pm.records.mental_state import PersonalityProfile

TOPICS = {"pizza": ["pizza", "cheese", "oven", "dough"], "train": ["train", "station", "ticket", "delay"]}


class FakeEmbeddings:
    """Hash topic words into a four-dimensional vector."""

    async def embed(self, text, cancel=None):
        words = text.casefold()
        vector = [0.05, 0.05, 0.05, 0.05]
        for index, topic in enumerate(TOPICS.values()):
            vector[index] += sum(words.count(word) for word in topic)
        return vector


class ScriptedModel:
    """Answer each prompt by its task phrase. ``calls`` lists the phrases."""

    def __init__(self):
        self.calls: list[str] = []

    def __call__(self, turns, **generation):
        text = normalize_turns(turns)[-1].content.text
        if normalize_turns(turns)[-1].role != Role.User:
            raise AssertionError("The last turn must be the task.")
        if "keeps this scene as one episode" in text:
            return self.answer("topic", '{"summary": "Bob and Ava talk about one topic in detail."}')
        if "keeps how this episode felt" in text:
            return self.answer("state", '{"description": "Ava seems relaxed; read from the words."}')
        if "keeps this period as one memory" in text or "joins these memories" in text:
            return self.answer("temporal", '{"summary": "Between the start and the end, Bob and Ava talked."}')
        if "turns this scene into lasting knowledge" in text:
            return self.answer("extract", '{"facts": [{"text": "Bob says that he likes pizza.", "lines": [1]}]}')
        if "files each fact" in text:
            count = len(re.findall(r"^\d+\. ", text.split("### BEGIN FACTS")[1].split("### END FACTS")[0], re.M))
            items = ",".join(
                f'{{"index": {i}, "about": "Bob", "record_type": "preference", "epistemic_status": "asserted", '
                f'"confidence": 0.9, "object_kind": "food", "object_value": "pizza", "topic_card": "food", '
                f'"essence": "Bob likes pizza.", "category": ["people_preferences"], "importance": 0.6, '
                f'"time_dependent": 0.2, "reason": "A stated preference."}}' for i in range(1, count + 1))
            return self.answer("file", f'{{"facts": [{items}]}}')
        if "narrative" in text:
            return self.answer("narrative", '{"narrative": "The character is steady and curious, and the story '
                                            'shows a calm, teasing rhythm with Bob over food and travel."}')
        raise AssertionError(f"Unknown prompt: {text[-300:]}")

    def answer(self, phase: str, text: str) -> str:
        self.calls.append(phase)
        return text


def _rows() -> list[ChatRow]:
    start = datetime(2026, 5, 1, 19, 0)
    rows, number = [], 0
    for day, topic in ((0, "pizza"), (1, "train")):
        for step in range(14):
            number += 1
            words = " ".join(TOPICS[topic])
            rows.append(ChatRow(row=number, speaker="user" if step % 2 == 0 else "companion",
                                message=f"message {number} about {words}",
                                timestamp=start + timedelta(days=day, minutes=2 * step)))
    return rows


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.store = MemoryStore(PersonaConfig(user_name="Bob", companion_name="Ava", character_card_story="Ava."))
        self.now = datetime(2026, 6, 10, 12, 0)
        import_exchanges(self.store, group_exchanges(_rows()), import_id="import-t", source_sha256="abc",
                         personality=PersonalityProfile(), imported_at=self.now)
        self.model = ScriptedModel()

    def run_once(self):
        return reconsolidate(self.store, complete=self.model, embeddings=FakeEmbeddings(), force=True, now=self.now)

    def test_run_cites_sources_and_a_second_run_does_nothing(self):
        report = self.run_once()
        records = {key: self.store.get_knoxel_by_id(key) for key in report.created_ids}
        topical = [records[key] for key in report.topical_ids]
        temporal = [records[key] for key in report.temporal_ids]
        self.assertGreaterEqual(len(topical), 2)
        self.assertTrue(all(isinstance(item, MemoryClusterRecord) and item.cluster_type == ClusterType.Topical
                            and item.included_event_ids and item.lineage.source_event_ids for item in topical))
        self.assertEqual(unclustered_events(self.store), [])
        keys = sorted(item.temporal_key for item in temporal)
        self.assertIn("2026-05-01-EVENING", keys)
        self.assertIn("2026-05-01", keys)
        self.assertIn("2026-W17", keys)
        self.assertIn("2026-05", keys)
        self.assertTrue(all(item.included_event_ids or item.included_cluster_ids for item in temporal))
        facts = [records[key] for key in report.fact_ids]
        self.assertTrue(facts and all(isinstance(item, DeclarativeFactRecord) and item.source_cluster_id
                                      and item.lineage.source_event_ids for item in facts))
        self.assertEqual(len(report.semantic_ids), 1, "Equal claims merge into one semantic record.")
        semantic = records[report.semantic_ids[0]]
        self.assertIsInstance(semantic, SemanticRecord)
        self.assertEqual(sorted(semantic.source_knoxel_ids), sorted(report.fact_ids))
        self.assertEqual(semantic.lineage.parent_record_ids, [min(report.fact_ids)])
        self.assertTrue(all(item.semantic_record_id == semantic.id for item in facts))
        self.assertTrue(all(self.store.get_knoxel_by_id(key).facts_extracted for key in report.topical_ids))
        narratives = [records[key] for key in report.narrative_ids]
        self.assertEqual(len(narratives), 9)
        self.assertTrue(all(isinstance(item, Narrative) and item.lineage.operation == OPERATION_NARRATIVE
                            and item.lineage.source_event_ids for item in narratives))
        self.assertIn(f"reconsolidation:{report.run_id}", self.store.consolidations)

        add_transfer_records(self.store, import_id="import-t", imported_at=self.now)
        calls = len(self.model.calls)
        second = self.run_once()
        self.assertEqual(len(self.model.calls), calls)
        self.assertEqual(second.model_calls, 0)
        self.assertEqual(second.created_ids, [])


if __name__ == "__main__":
    unittest.main()
