"""Use the real process-local admission gate to verify memory yields every slot."""
import asyncio
import unittest

from pm.llm.concurrency import InferenceAdmission, InferenceRole, inference_role


class MemoryAdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def take_slot(self, admission: InferenceAdmission, role: InferenceRole,
                        order: list[InferenceRole]) -> None:
        with inference_role(role):
            async with admission.slot():
                order.append(role)

    async def test_foreground_and_private_calls_queued_after_memory_receive_the_next_slot(self) -> None:
        admission = InferenceAdmission(max_concurrent_requests=1)
        order: list[InferenceRole] = []
        async with admission.slot():
            pending: list[asyncio.Task[None]] = []
            for role in (InferenceRole.Reconsolidation, InferenceRole.Private, InferenceRole.Cognition,
                         InferenceRole.Interaction, InferenceRole.Critical, InferenceRole.Triage):
                pending.append(asyncio.create_task(self.take_slot(admission, role, order)))
                await asyncio.sleep(0)
        for task in pending:
            await task
        self.assertEqual(order, [InferenceRole.Triage, InferenceRole.Critical, InferenceRole.Interaction,
                                 InferenceRole.Private, InferenceRole.Cognition, InferenceRole.Reconsolidation])
