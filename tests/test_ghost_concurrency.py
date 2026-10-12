"""Actual concurrent commits against one Ghost and one checkpoint."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier
import unittest

from pm.memory.store import Ghost, GhostConflict
from pm.records.memory_records import Feature
from pm.records.enums import FeatureType
from tests.test_ghost_commits import small_ghost


class GhostConcurrencyTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "ghost.json"
        self.ghost = small_ghost(self.path)

    def together(self, prepare):
        ready = Barrier(2)

        def run(index):
            batch = self.ghost.begin_batch(f"worker-{index}", new_tick=True)
            prepare(batch, index)
            ready.wait(timeout=10)
            try:
                return self.ghost.commit(batch)
            except GhostConflict as error:
                return error

        with ThreadPoolExecutor(max_workers=2) as workers:
            return list(workers.map(run, range(2)))

    def test_unrelated_appends_both_commit(self):
        receipts = self.together(lambda batch, index: batch.create(
            Feature(feature_type=FeatureType.Dialogue, content=f"worker {index}")))
        self.assertTrue(all(not isinstance(item, Exception) for item in receipts))
        reopened = Ghost.from_checkpoint(self.path)
        self.assertEqual({record.content for record in reopened.all_knoxels.values()}, {"worker 0", "worker 1"})
        self.assertEqual(len({receipt.added_ids[0] for receipt in receipts}), 2)
        self.assertEqual(len({receipt.tick_id for receipt in receipts}), 2)

    def test_same_record_has_one_winner_without_partial_append(self):
        original = self.ghost.store_knoxel(Feature(feature_type=FeatureType.Dialogue, content="original"))

        def prepare(batch, index):
            record = batch.view().get(original.id)
            record.content = f"winner {index}"
            batch.replace(record)
            batch.create(Feature(feature_type=FeatureType.Dialogue, content=f"append {index}"))

        results = self.together(prepare)
        self.assertEqual(sum(isinstance(result, GhostConflict) for result in results), 1)
        reopened = Ghost.from_checkpoint(self.path)
        winner = reopened.get_knoxel_by_id(original.id).content[-1]
        self.assertEqual({record.content for record in reopened.all_knoxels.values()}, {f"winner {winner}", f"append {winner}"})
        self.assertEqual(reopened.revisions[original.id], 2)

    def test_same_run_receipt_has_one_winner(self):
        results = self.together(lambda batch, index: batch.set_run_receipt(
            "job", {"state": "completed", "result": index}))
        self.assertEqual(sum(isinstance(result, GhostConflict) for result in results), 1)
        reopened = Ghost.from_checkpoint(self.path)
        self.assertEqual(reopened.managed_runs, self.ghost.managed_runs)
