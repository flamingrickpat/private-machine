"""W8 live Minecraft body acceptance.

When ``MINECRAFT_MCP_URL`` is unset (or the default) and that endpoint is
not running, the test starts a fresh server and the pm-minecraft MCP from the
local checkout. Set ``PM_MINECRAFT_AUTOSTART=0`` to require an external
endpoint instead. An explicitly configured external endpoint is used as-is.

``PM_MINECRAFT_REPO`` selects the pm-minecraft checkout (default
``C:/source/pm/pm-minecraft``) and ``PM_MINECRAFT_ENV`` selects its env file
(default ``.env_tdd``).

The test demonstrates observation, one real action, status, an instruction
update, termination, and memory recovery from a reopened checkpoint.
"""
from __future__ import annotations
from runtime.body_lease import BODY_LEASES

import json
import os
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from pm.memory.store import MemoryStore
from pm.records.cycle_state import PersonaConfig
from pm.llm.stage_model import LazyStageModel
from pm.llm.activation import ModelActivation
from pm.llm.config import load_config, resolve_connection
from runtime.consolidation import Consolidator
from pm.runs.supervisor import ManagedRunSupervisor, WorkflowSpec
from runtime.minecraft_tools import DEFAULT_MCP_URL, build_minecraft_tools
from tests.live.minecraft_stack import MinecraftStack, autostart_enabled
from pm.runs.workflow_module import PROJECT_ROOT

EVIDENCE_ROOT = (
    Path(".runtime/w8/live-minecraft")
    / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:8])
)
MANIFEST = PROJECT_ROOT / "workflows" / "minecraft" / "minecraft.workflow.md"


def _wait_for(predicate, timeout: float | None = 180.0):
    deadline = time.monotonic() + timeout if timeout is not None else None
    while deadline is None or time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def _save(name: str, payload) -> None:
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    (EVIDENCE_ROOT / name).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8",
    )


class LiveMinecraftBodyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mcp_url = os.environ.get("MINECRAFT_MCP_URL", DEFAULT_MCP_URL)
        cls.stack = None
        if autostart_enabled(cls.mcp_url):
            # No endpoint is running. Start a fresh server and the MCP so the
            # gate is self-contained instead of failing on a missing endpoint.
            cls.stack = MinecraftStack(
                repo=Path(os.environ.get("PM_MINECRAFT_REPO", "C:/source/pm/pm-minecraft")),
                env_file=os.environ.get("PM_MINECRAFT_ENV", ".env_tdd"),
            )
            cls.stack.start()

    @classmethod
    def tearDownClass(cls):
        if cls.stack is not None:
            cls.stack.stop()
            cls.stack = None

    def setUp(self):
        self.mcp_url = type(self).mcp_url
        tools = build_minecraft_tools(self.mcp_url)
        try:
            names = tools["minecraft_list_tools"]()
        except Exception as error:  # a required endpoint is missing
            self.fail(f"The Minecraft MCP is not reachable at {self.mcp_url}: {error!r}")
        self.assertIn("minecraft_observe", names)
        self.tools = tools
        config_path = os.environ.get("PM_MODEL_CONFIG", "config.local.yaml" if Path("config.local.yaml").is_file() else "config.example.yaml")
        self.model = LazyStageModel(resolve_connection(load_config(config_path)), ModelActivation(), live_test=True)

    def _target(self) -> dict:
        observation = self.tools["minecraft_call"]("minecraft_observe", {"include_image": False})
        self.assertTrue(observation["ok"], observation)
        blocks = [
            block for block in (observation["result"].get("nearby_blocks") or [])
            if block.get("block_name") in {"dirt", "grass_block", "stone", "oak_log"}
        ]
        self.assertTrue(blocks, "the isolated world has no minable block near the player")
        target = min(blocks, key=lambda block: block.get("distance", 999))
        return {"position": target["nearest"], "block": target["block_name"]}

    def test_body_lifecycle_and_memory_recovery(self):
        root = EVIDENCE_ROOT / "state"
        root.mkdir(parents=True, exist_ok=True)
        ghost = MemoryStore(
            PersonaConfig(user_name="Jesse", companion_name="Spamuel"),
            checkpoint=root / "ghost.json",
        )
        consolidator = Consolidator(ghost, clock=time.monotonic)
        supervisor = ManagedRunSupervisor(
            ghost,
            runtime_root=root / "runtime",
            workflows={
                "minecraft": WorkflowSpec(
                    MANIFEST,
                    namespace="minecraft",
                    tools_factory=lambda run_id, environment: {
                        **build_minecraft_tools(self.mcp_url, plan_model=self.model.complete),
                        "minecraft_endpoint": lambda: self.mcp_url,
                        "minecraft_lease": lambda: BODY_LEASES.acquire(self.mcp_url, run_id),
                    },
                ),
            },
            clock=time.monotonic,
            task_root=root / "tasks",
            on_terminal=lambda receipt: consolidator.consolidate(
                receipt["run_id"], receipt["artifact_path"],
            ),
        )
        def stop_workers():
            for run_id in supervisor.receipts():
                supervisor.terminate(run_id)
            supervisor.join_all()
        self.addCleanup(stop_workers)
        target = self._target()

        # Observation, one real action, and completion.
        run = supervisor.start(
            action_id="mc-action-1", workflow_name="minecraft",
            instructions=f"mine one {target['block']}",
            arguments={
                "goal": f"mine one {target['block']}",
                "body_id": "tdd_player",
                "action_tool": "minecraft_mine_block",
                "action_arguments": {"position": target["position"]},
            },
        )
        run_id = run["run_id"]
        self.assertTrue(
            _wait_for(lambda: supervisor.status(run_id)["state"] in {"completed", "failed"}),
            "the Minecraft body run never finished",
        )
        status = supervisor.status(run_id)
        self.assertEqual(status["state"], "completed", status["latest_event"])
        self.assertIn("outbox/world.yaml", status["evidence"])
        world = (Path(status["artifact_path"]) / "outbox" / "world.yaml").read_text(encoding="utf-8")
        self.assertIn("places", world)
        report = json.loads(json.dumps(__import__("yaml").safe_load(
            (Path(status["artifact_path"]) / "outbox/report.yaml").read_text(encoding="utf-8"))))
        self.assertTrue(report["accepted"], report)
        self.assertTrue(report["action"]["result"]["ok"], report["action"])
        after = self.tools["minecraft_call"]("minecraft_inspect_block", {"position": target["position"]})
        # The MCP reports an emptied cell as ``ok=False`` with ``not_found`` and
        # ``block_name="air"``. The removal is confirmed by the block name.
        self.assertIs(after["result"]["ok"], False, after)
        self.assertEqual(after["result"]["block_name"], "air", after)
        screenshots = list((Path(status["artifact_path"]) / "screenshots").glob("*")) \
            if (Path(status["artifact_path"]) / "screenshots").is_dir() else []
        self.assertTrue(screenshots, "the body run wrote no screenshot evidence")

        # Memory recovery from a reopened checkpoint.
        self.assertTrue(_wait_for(lambda: ghost.consolidations, timeout=30))
        reopened = MemoryStore.from_checkpoint(root / "ghost.json")
        self.assertTrue(reopened.consolidations)
        consolidation = next(iter(reopened.consolidations.values()))
        self.assertEqual(consolidation["run_id"], run_id)
        self.assertTrue(consolidation["image_evidence"])

        # Status, update, and termination on a held run.
        held = supervisor.start(
            action_id="mc-action-2", workflow_name="minecraft",
            instructions="hold at the boundary",
            arguments={"body_id": "tdd_player", "wait_for_update": True, "wait_for_cancel": True,
                       "allowed_tools": ["minecraft_observe"]},
        )
        held_id = held["run_id"]
        held_path = Path(held["artifact_path"])
        self.assertTrue(
            _wait_for(lambda: (held_path / "outbox" / "world.yaml").is_file()),
            "the held run never reached its artifact boundary",
        )
        updated = supervisor.update(held_id, "Observe the world once without changing any block.", request_id="w8-update")
        self.assertEqual(updated["revision"], 1)
        self.assertTrue(_wait_for(lambda: (held_path / "outbox/report.yaml").is_file()
                                 or supervisor.status(held_id)["state"] == "failed", timeout=None))
        self.assertNotEqual(supervisor.status(held_id)["state"], "failed", supervisor.status(held_id))
        held_report = __import__("yaml").safe_load((held_path / "outbox/report.yaml").read_text(encoding="utf-8"))
        self.assertEqual(held_report["goal"], updated["instructions"])
        self.assertEqual(held_report["action"]["tool"], "minecraft_observe")
        self.assertEqual(supervisor.status(held_id)["applied_revision"], 1)
        terminated = supervisor.terminate(held_id)
        self.assertEqual(terminated["state"], "cancelled")
        held_status = supervisor.status(held_id)
        self.assertTrue((Path(held_status["artifact_path"]) / "outbox" / "world.yaml").is_file(),
                        "termination did not preserve the world artifact")

        _save("minecraft.json", {
            "mcp_url": self.mcp_url,
            "target": target,
            "action_report": report,
            "post_action_block": after,
            "updated_report": held_report,
            "run": status,
            "consolidation": consolidation,
            "held_run": held_status,
            "screenshots": [path.name for path in screenshots],
        })
        supervisor.join_all(timeout=60)
        print(f"run root: {EVIDENCE_ROOT}")
        print(json.dumps({
            "state": status["state"], "target": target,
            "consolidation_memories": len(consolidation["memory_ids"]),
            "held_state": held_status["state"], "revision": updated["revision"],
        }, indent=2))

if __name__ == "__main__":
    unittest.main()
