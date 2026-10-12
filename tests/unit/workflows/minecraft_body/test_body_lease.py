"""Exercise real body locks in an isolated registry."""
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest

from pm.host.durable_files import BodyBusy
from workflows.minecraft_body.lease import BodyLeases


class BodyLeaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.leases = BodyLeases(Path(self.tmp.name))

    def test_second_actor_is_rejected_until_the_owner_finishes(self):
        leases = self.leases
        with leases.acquire("http://127.0.0.1:6767/mcp/", "first"):
            with self.assertRaises(BodyBusy):
                with leases.acquire("http://127.0.0.1:6767/mcp", "second"):
                    self.fail("A second actor obtained the avatar.")
            self.assertEqual(leases.owner("http://127.0.0.1:6767/mcp"), "first")
        with leases.acquire("http://127.0.0.1:6767/mcp", "second"):
            self.assertEqual(leases.owner("http://127.0.0.1:6767/mcp"), "second")

    def test_error_releases_ownership(self):
        leases = self.leases
        with self.assertRaises(ValueError):
            with leases.acquire("avatar", "first"):
                raise ValueError("The actor stopped before another action.")
        self.assertIsNone(leases.owner("avatar"))

    def test_same_owner_cannot_start_an_overlapping_session(self):
        leases = self.leases
        with leases.acquire("avatar", "first"):
            with self.assertRaises(BodyBusy):
                with leases.acquire("avatar", "first"):
                    self.fail("The session overlapped itself.")

    def test_other_thread_cannot_take_the_active_body(self):
        leases = self.leases
        denied = threading.Event()
        def contender():
            try:
                with leases.acquire("avatar", "second"):
                    pass
            except BodyBusy:
                denied.set()
        with leases.acquire("avatar", "first"):
            thread = threading.Thread(target=contender)
            thread.start()
            thread.join(1)
            self.assertTrue(denied.is_set())

    def test_read_waits_between_commands_without_taking_ownership(self):
        leases = BodyLeases()
        entered = threading.Event()
        waiting = threading.Event()
        def read():
            waiting.set()
            with leases.command("avatar"):
                entered.set()
        with leases.acquire("avatar", "actor"):
            with leases.command("avatar"):
                thread = threading.Thread(target=read)
                thread.start()
                self.assertTrue(waiting.wait(1))
                self.assertFalse(entered.wait(0.1))
            thread.join(1)
            self.assertTrue(entered.is_set())
            self.assertEqual(leases.owner("avatar"), "actor")

    def test_cancelled_waiter_sends_no_command(self):
        from pm_workflows.runtime import WorkflowTerminated
        leases = BodyLeases()
        cancel, terminated = threading.Event(), threading.Event()
        def actor():
            try:
                with leases.command("avatar", cancel):
                    self.fail("A cancelled actor reached dispatch.")
            except WorkflowTerminated:
                terminated.set()
        with leases.command("avatar"):
            thread = threading.Thread(target=actor)
            thread.start()
            cancel.set()
            thread.join(1)
            self.assertTrue(terminated.is_set())
