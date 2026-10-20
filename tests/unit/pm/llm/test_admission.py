"""Check cooperative admission with events instead of model inference."""
import asyncio
from threading import Event
import unittest

from pm.llm.concurrency import InferenceAdmission, background_inference


class AdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_current_private_call_finishes_then_the_reply_chain_runs(self):
        admission, order = InferenceAdmission(), []
        started, finish = asyncio.Event(), asyncio.Event()

        async def private():
            with background_inference():
                async with admission.slot():
                    started.set()
                    await finish.wait()
                    order.append("private-current")
                async with admission.slot():
                    order.append("private-next")

        task = asyncio.create_task(private())
        await started.wait()
        admission.reserve("user")
        finish.set()
        async with admission.slot():
            order.append("cognition")
        await asyncio.sleep(0.02)
        self.assertEqual(order, ["private-current", "cognition"])
        async with admission.slot():
            order.append("dialogue")
        admission.release("user")
        await asyncio.wait_for(task, 1)
        self.assertEqual(order, ["private-current", "cognition", "dialogue", "private-next"])

    async def test_cancelled_waiter_releases_its_queue_position(self):
        admission, cancel = InferenceAdmission(), Event()
        admission.reserve("foreground")

        async def private():
            with background_inference():
                async with admission.slot(cancel):
                    self.fail("Cancelled private work entered the slot.")

        task = asyncio.create_task(private())
        await asyncio.sleep(0)
        cancel.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        admission.release("foreground")
        async with admission.slot():
            pass
