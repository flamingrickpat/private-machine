"""Small real-component checks for the design boundaries found during audit."""
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from fastmcp import Client
import yaml

from pm_core.ghost.service import Ghost
from ghost_mcp import build_mcp, build_tools
from main import build_app
from pm_core.model.ghost_state import GhostConfig
from runtime.shell import Application, ShellModule
from bus.bus import BusSystem
from bus.envelopes import Envelope
from tests.support import ManualClock, Recorder, store_action
from working_target import GhostBashEnvironment, GhostBashWrapper, FolderTarget
from workflow_adapter import ShellModuleWorkflow


def small_ghost():
    return Ghost(GhostConfig(user_name="Jesse", companion_name="Spamuel"))


class DesignContracts(unittest.TestCase):
    def test_cleanup_exception_reaches_application(self):
        class BrokenCleanup(ShellModule):
            def handle_batch(self, batch):
                pass

            def on_terminate(self):
                raise ValueError("cleanup failed")

        app = Application(BusSystem(), [])
        module = BrokenCleanup("broken", app.bus, app.on_error, ManualClock())
        app.modules = [module]
        module.stop()
        # Exercise the actual thread boundary synchronously in the debugger.
        module._run_loop()
        with self.assertRaisesRegex(ValueError, "cleanup failed"):
            app.raise_if_error()

    def test_virtual_files_are_discoverable_and_read_only(self):
        record = {"__type__": "Feature", "data": {"id": 1, "content": "visible memory"}}
        env = GhostBashEnvironment({1: record}, {}, "read")
        self.assertEqual(env.list_files("ghost/knoxels"), ["ghost/knoxels/1.yaml"])
        self.assertIn("visible memory", env.exec("grep -r 'visible memory' /work/ghost").stdout)
        self.assertNotEqual(env.exec("echo changed > /work/ghost/knoxels/1.yaml").exit_code, 0)
        self.assertEqual(yaml.safe_load(env.read_text("ghost/knoxels/1.yaml")), record)

    def test_virtual_text_preserves_whitespace(self):
        env = GhostBashEnvironment({}, {}, "read")
        text = "  indented markdown\n\n"
        env.write_text("scratch/text.md", text)
        env.move("scratch/text.md", "outbox/text.md")
        self.assertEqual(env.read_text("outbox/text.md"), text)

    def test_real_mcp_roundtrip_and_scope(self):
        record = {"__type__": "Feature", "data": {"id": 1, "content": "Jesse"}}
        env = GhostBashEnvironment({1: record}, {}, "read")

        async def exercise():
            async with Client(build_mcp(env)) as client:
                result = await client.call_tool("ghost_get", {"knoxel_id": 1})
                self.assertFalse(result.is_error)
                self.assertIn("Jesse", str(result))
                hidden = await client.call_tool("ghost_get", {"knoxel_id": 2}, raise_on_error=False)
                self.assertTrue(hidden.is_error)

        asyncio.run(exercise())

    def test_tool_results_cannot_mutate_the_snapshot(self):
        env = GhostBashEnvironment({1: {"__type__": "Feature", "data": {"id": 1, "content": "original"}}}, {}, "read")
        tools = build_tools(env)
        tools["ghost_get"](1)["data"]["content"] = "changed"
        self.assertEqual(tools["ghost_get"](1)["data"]["content"], "original")
        self.assertIn("original", env.read_text("ghost/knoxels/1.yaml"))

    def test_workflow_input_cannot_bypass_ghost_scope(self):
        with TemporaryDirectory() as tmp:
            ghost = small_ghost()
            action = store_action(ghost, content="hidden instruction")
            app = build_app(None, tmp, ghost=ghost, clock=ManualClock())
            app.dialogue.working_target = GhostBashWrapper(ghost, visible_ids=())
            app.bus.put(action)
            with self.assertRaises(KeyError):
                app.dialogue.step()
            self.assertIsNone(app.dialogue.last_kernel_summary)
            self.assertEqual(app.dialogue.last_environment.list_files("inbox"), [])

    def test_generic_workflow_needs_no_ghost(self):
        class PlainWorkflow(ShellModuleWorkflow):
            def prepare_input(self, batch):
                return {"text": batch[0].payload["text"]}

            def publish_output(self, report, batch):
                self.report = report

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            role = root / "role.py"
            role.write_text(
                "import yaml\ndef run(context):\n"
                "    env = context.runtime.environment\n"
                "    request = yaml.safe_load(env.read_text('inbox/request.yaml'))\n"
                "    env.write_text('outbox/report.yaml', yaml.safe_dump(request))\n"
                "    return {'status': 'done', 'summary': 'copied'}\n",
                encoding="utf-8",
            )
            manifest = root / "plain.workflow.md"
            manifest.write_text(
                "---\nname: plain\ndriver: {kind: python}\nroles:\n"
                f"  work:\n    skill: {role.as_posix()}\n"
                "    result_contract:\n      schema: {status: {enum: [done]}, summary: string}\n"
                "phases:\n  - name: work\n    kind: role\n    role: work\n"
                "    on_status: {done: stop}\n    on_invalid: {action: stop}\n---\n",
                encoding="utf-8",
            )
            module = PlainWorkflow(
                "plain", BusSystem(), self.fail, ManualClock(), None,
                working_target=FolderTarget(root / "target", "bash"),
                workflow=manifest, tools_factory=lambda env, batch: {}, runtime_root=root / "runtime",
            )
            module.mailbox.put(Envelope("one", "test", "data", "plain", "one", {"text": "hello"}))
            self.assertTrue(module.step())
            self.assertEqual(module.report, {"text": "hello"})
            self.assertIsNone(module._thread)


if __name__ == "__main__":
    unittest.main()
