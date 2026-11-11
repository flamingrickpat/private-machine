"""Exercise real endpoint admission synchronization without model inference."""
import asyncio
from threading import Barrier, Event
import unittest

from pm.llm.concurrency import (
    InferenceAdmission,
    InferenceRole,
    background_inference,
    inference_role,
    ordered_parallel_map,
)


class AdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_current_private_call_finishes_before_critical_chain(self) -> None:
        admission = InferenceAdmission()
        order: list[str] = []
        started = asyncio.Event()
        finish = asyncio.Event()

        async def private() -> None:
            with background_inference():
                async with admission.slot():
                    self.assertTrue(admission.background_active)
                    started.set()
                    await finish.wait()
                    order.append("private-current")
                async with admission.slot():
                    order.append("private-next")

        task = asyncio.create_task(private())
        await started.wait()
        admission.reserve("critical", role=InferenceRole.Critical)
        finish.set()
        with inference_role(InferenceRole.Critical):
            async with admission.slot():
                order.append("cognition")
            await asyncio.sleep(0.02)
            self.assertEqual(order, ["private-current", "cognition"])
            async with admission.slot():
                order.append("dialogue")
        admission.release("critical")
        await asyncio.wait_for(task, 1)
        self.assertEqual(order, ["private-current", "cognition", "dialogue", "private-next"])

    async def test_cancelled_waiter_releases_its_queue_position(self) -> None:
        admission = InferenceAdmission()
        cancel = Event()
        admission.reserve("critical")

        async def private() -> None:
            with background_inference():
                async with admission.slot(cancel):
                    self.fail("Cancelled private work entered the slot.")

        task = asyncio.create_task(private())
        await asyncio.sleep(0)
        cancel.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        admission.release("critical")
        async with admission.slot():
            pass

    async def test_four_urgent_calls_then_oldest_normal_share_fifo(self) -> None:
        admission = InferenceAdmission()
        order: list[str] = []

        async def call(name: str, role: InferenceRole) -> None:
            with inference_role(role):
                async with admission.slot():
                    order.append(name)

        async with admission.slot():
            tasks: list[asyncio.Task[None]] = []
            for index in range(2):
                tasks.append(asyncio.create_task(call(f"interaction-{index}", InferenceRole.Interaction)))
            tasks.append(asyncio.create_task(call("private", InferenceRole.Private)))
            tasks.append(asyncio.create_task(call("cognition", InferenceRole.Cognition)))
            for index in range(6):
                tasks.append(asyncio.create_task(call(f"triage-{index}", InferenceRole.Triage)))
            await asyncio.sleep(0)
        await asyncio.wait_for(asyncio.gather(*tasks), 2)
        self.assertEqual(order, ["triage-0", "triage-1", "triage-2", "triage-3", "private",
                                 "triage-4", "triage-5", "interaction-0", "interaction-1", "cognition"])

    async def test_critical_reservation_blocks_interaction_and_normal_but_not_triage(self) -> None:
        admission = InferenceAdmission()
        order: list[str] = []

        async def call(name: str, role: InferenceRole) -> None:
            with inference_role(role):
                async with admission.slot():
                    order.append(name)

        admission.reserve("critical", role=InferenceRole.Critical)
        tasks = [asyncio.create_task(call(role.value, role)) for role in
                 (InferenceRole.Private, InferenceRole.Cognition, InferenceRole.Interaction)]
        await asyncio.sleep(0)
        with inference_role(InferenceRole.Critical):
            async with admission.slot():
                triage = asyncio.create_task(call("triage", InferenceRole.Triage))
                critical = asyncio.create_task(call("critical", InferenceRole.Critical))
                await asyncio.sleep(0)
        await asyncio.wait_for(asyncio.gather(triage, critical), 1)
        self.assertEqual(order, ["triage", "critical"])
        admission.release("critical")
        await asyncio.wait_for(asyncio.gather(*tasks), 1)
        self.assertEqual(order, ["triage", "critical", "interaction", "private", "cognition"])

    async def test_task_cancellation_and_body_failure_release_slot(self) -> None:
        admission = InferenceAdmission()
        admission.reserve("critical")

        async def waiting() -> None:
            async with admission.slot():
                self.fail("Normal request crossed critical reservation.")

        task = asyncio.create_task(waiting())
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        admission.release("critical")
        with self.assertRaisesRegex(ValueError, "body failed"):
            async with admission.slot():
                raise ValueError("body failed")
        async with admission.slot():
            self.assertFalse(admission.background_active)

    async def test_noncritical_reservation_does_not_block_private(self) -> None:
        admission = InferenceAdmission()
        admission.reserve("ordinary", role=InferenceRole.Cognition)
        with background_inference():
            async with admission.slot():
                self.assertTrue(admission.background_active)
        admission.release("ordinary")


class WorkerContextTests(unittest.TestCase):
    def test_parallel_workers_keep_private_role_and_serialize_independent_loops(self) -> None:
        admission = InferenceAdmission()
        barrier = Barrier(2)
        active_count = 0
        maximum_active_count = 0

        async def local_operation(item: int) -> int:
            nonlocal active_count, maximum_active_count
            async with admission.slot():
                self.assertTrue(admission.background_active)
                active_count += 1
                maximum_active_count = max(maximum_active_count, active_count)
                await asyncio.sleep(0.02)
                active_count -= 1
                return item

        def operation(item: int) -> int:
            barrier.wait(timeout=5)
            return asyncio.run(local_operation(item))

        with background_inference():
            results = ordered_parallel_map(operation, [0, 1], workers=2, cancel=Event())
        self.assertEqual(results, [0, 1])
        self.assertEqual(maximum_active_count, 1)
        self.assertFalse(admission.background_active)

    def test_role_scope_restores_outer_context(self) -> None:
        admission = InferenceAdmission()

        async def check() -> None:
            with background_inference():
                with inference_role(InferenceRole.Interaction):
                    async with admission.slot():
                        self.assertFalse(admission.background_active)
                async with admission.slot():
                    self.assertTrue(admission.background_active)
            async with admission.slot():
                self.assertFalse(admission.background_active)

        asyncio.run(check())
