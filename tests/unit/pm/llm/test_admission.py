"""Real admission synchronization; no model or fabricated endpoint."""
import asyncio
from threading import Barrier, Event
import unittest
from pm.llm.concurrency import InferenceAdmission, InferenceRole, inference_role, background_inference, ordered_parallel_map

class AdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_configured_slots_can_be_occupied(self) -> None:
        admission = InferenceAdmission(8)
        entered = 0
        all_entered = asyncio.Event()
        release = asyncio.Event()
        async def use() -> None:
            nonlocal entered
            async with admission.slot():
                entered += 1
                if entered == 8:
                    all_entered.set()
                await release.wait()
        tasks = [asyncio.create_task(use()) for _ in range(8)]
        await asyncio.wait_for(all_entered.wait(), 2)
        release.set()
        await asyncio.gather(*tasks)
        self.assertEqual(entered, 8)

    async def test_triage_gets_next_free_slot_before_queued_private_work(self) -> None:
        admission = InferenceAdmission(1)
        order: list[str] = []
        async def use(role: InferenceRole) -> None:
            with inference_role(role):
                async with admission.slot():
                    order.append(role.value)
        async with admission.slot():
            private = asyncio.create_task(use(InferenceRole.Private))
            await asyncio.sleep(0)
            triage = asyncio.create_task(use(InferenceRole.Triage))
            await asyncio.sleep(0)
        await asyncio.gather(private, triage)
        self.assertEqual(order, ['triage', 'private'])

    async def test_unlimited_settings_admit_without_waiting_for_active_calls(self) -> None:
        for limit in (-1, 0, 9999):
            admission = InferenceAdmission(limit)
            async with admission.slot():
                async with admission.slot():
                    self.assertFalse(admission.background_active)

    async def test_cancelled_waiter_releases_queue_position(self) -> None:
        admission = InferenceAdmission(1)
        async def wait() -> None:
            async with admission.slot():
                self.fail('Cancelled queued work was admitted')
        async with admission.slot():
            task = asyncio.create_task(wait())
            await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        async with admission.slot():
            pass

class WorkerContextTests(unittest.TestCase):
    def test_workers_overlap_across_event_loops_and_preserve_role(self) -> None:
        admission = InferenceAdmission(2)
        barrier = Barrier(2)
        async def use(index: int) -> int:
            async with admission.slot():
                self.assertTrue(admission.background_active)
                await asyncio.to_thread(barrier.wait, 2)
                return index
        def operation(index: int) -> int:
            return asyncio.run(use(index))
        with background_inference():
            self.assertEqual(ordered_parallel_map(operation, [0, 1], workers=2, cancel=Event()), [0, 1])
        self.assertFalse(admission.background_active)
