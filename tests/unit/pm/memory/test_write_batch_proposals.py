"""Role-scoped MemoryStore proposals over a real write batch."""
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest

from pm.bus.bus import BusSystem
from pm.bus.envelopes import Envelope
from pm.memory.store import MemoryStore
from pm.memory.store_tools import build_tools
from pm.records.cycle_state import PersonaConfig
from pm.records.memory_records import Feature
from pm.records.enums import FeatureType
from pm.memory.bash_view import MemoryBashEnvironment, MemoryBashTarget
from pm.runs.workflow_module import WorkflowModule


def _ghost() -> MemoryStore:
    return MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"))


def _record(content: str) -> dict:
    return (Feature(feature_type=FeatureType.Dialogue, content=content)).as_payload()


class GhostProposalTests(unittest.TestCase):
    def test_read_only_tools_expose_no_proposal_tools(self):
        env = MemoryBashEnvironment({}, {}, "read")
        keys = set(build_tools(env))
        self.assertEqual(keys, {"ghost_get", "ghost_search", "ghost_gwt"})

    def test_create_is_staged_in_the_batch_then_becomes_canonical(self):
        ghost = _ghost()
        env = MemoryBashEnvironment({}, {}, "read")
        batch = ghost.begin_batch("turn")
        tools = build_tools(env, batch)

        created = tools["ghost_propose_create"](_record("a new memory"))
        new_id = created["data"]["id"]
        self.assertIsNone(ghost.get_knoxel_by_id(new_id))
        self.assertEqual(tools["ghost_get"](new_id)["data"]["content"], "a new memory")
        self.assertIn(new_id, [r["data"]["id"] for r in tools["ghost_search"]("new memory")])

        ghost.commit(batch)

        self.assertEqual(ghost.get_knoxel_by_id(new_id).content, "a new memory")
        view = ghost.snapshot()
        committed_env = MemoryBashEnvironment(view.serialize_knoxels(), view.serialize_gwt(), "read")
        self.assertEqual(build_tools(committed_env)["ghost_get"](new_id)["data"]["content"], "a new memory")

    def test_proposal_is_invisible_to_an_uncommitted_other_run(self):
        ghost = _ghost()
        env = MemoryBashEnvironment({}, {}, "read")
        first = build_tools(env, ghost.begin_batch("first"))
        second = build_tools(env, ghost.begin_batch("second"))

        created = first["ghost_propose_create"](_record("first run only"))
        new_id = created["data"]["id"]

        with self.assertRaises(KeyError):
            second["ghost_get"](new_id)
        self.assertEqual(second["ghost_search"]("first run only"), [])

    def test_replace_updates_the_batch_view_before_commit(self):
        ghost = _ghost()
        original = ghost.store_knoxel(
            Feature(feature_type=FeatureType.Dialogue, content="before")
        )
        env = MemoryBashEnvironment({}, {}, "read")
        batch = ghost.begin_batch("update")
        tools = build_tools(env, batch)

        replacement = _record("after")
        replacement["data"]["id"] = original.id
        replacement["data"]["tick_id"] = original.tick_id
        tools["ghost_propose_replace"](replacement)

        self.assertEqual(tools["ghost_get"](original.id)["data"]["content"], "after")
        self.assertEqual(ghost.get_knoxel_by_id(original.id).content, "before")

        ghost.commit(batch)
        self.assertEqual(ghost.get_knoxel_by_id(original.id).content, "after")

    def test_untrusted_record_is_rejected(self):
        ghost = _ghost()
        env = MemoryBashEnvironment({}, {}, "read")
        tools = build_tools(env, ghost.begin_batch("bad"))
        with self.assertRaises(Exception):
            tools["ghost_propose_create"]({"__type__": "NotAKnoxel", "data": {}})

    def test_role_proposal_commits_through_the_module(self):
        ghost = _ghost()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            role = root / "role.py"
            role.write_text(
                "import yaml\n"
                "def run(context):\n"
                "    env = context.runtime.environment\n"
                "    record = yaml.safe_load(env.read_text('inbox/request.yaml'))['record']\n"
                "    created = context.runtime.tools['ghost_propose_create'](record)\n"
                "    env.write_text('outbox/report.yaml',\n"
                "        yaml.safe_dump({'new_id': created['data']['id']}))\n"
                "    return {'status': 'done', 'summary': 'created'}\n",
                encoding="utf-8",
            )
            manifest = root / "commit.workflow.md"
            manifest.write_text(
                "---\nname: commit\ndriver: {kind: python}\nroles:\n"
                f"  work:\n    skill: {role.as_posix()}\n"
                "    result_contract:\n      schema: {status: {enum: [done]}, summary: string}\n"
                "phases:\n  - name: work\n    kind: role\n    role: work\n"
                "    on_status: {done: stop}\n    on_invalid: {action: stop}\n---\n",
                encoding="utf-8",
            )

            class CommitWorkflow(WorkflowModule):
                DATA_TOPICS = frozenset({"start"})

                def prepare_input(self, batch):
                    return {"record": self.record}

                def publish_output(self, report, batch):
                    self.report = report
                    self.receipt = ghost.commit(self.last_proposal_batch)

            module = CommitWorkflow(
                "commit", BusSystem(), self.fail, time.monotonic, None,
                working_target=MemoryBashTarget(ghost),
                workflow=manifest, tools_factory=build_tools,
                runtime_root=root / "runtime",
                batch_provider=lambda: ghost.begin_batch("run"),
            )
            module.record = _record("proposed memory")
            module.mailbox.put(Envelope("start", "test", "data", "start", "start", {}))

            self.assertTrue(module.step())

            new_id = module.report["new_id"]
            self.assertEqual(ghost.get_knoxel_by_id(new_id).content, "proposed memory")
            self.assertEqual(module.receipt.added_ids, (new_id,))


if __name__ == "__main__":
    unittest.main()