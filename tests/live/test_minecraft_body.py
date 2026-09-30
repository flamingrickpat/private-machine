"""W8 live Minecraft body acceptance.

Requires the real pm-minecraft MCP and an isolated checkpoint server. The
endpoint comes from ``MINECRAFT_MCP_URL`` (default
``http://127.0.0.1:6767/mcp``). A missing endpoint fails the test; it is never
skipped silently.

The test demonstrates observation, one real action, status, an instruction
update, termination, and memory recovery from a reopened checkpoint.
"""
from __future__ import annotations

import json
import os
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from pm_core.ghost.service import Ghost
from pm_core.model.ghost_state import GhostConfig
from runtime.consolidation import Consolidator
from runtime.managed_runs import ManagedRunSupervisor, WorkflowSpec
from runtime.minecraft_tools import DEFAULT_MCP_URL, build_minecraft_tools
from workflow_adapter import PROJECT_ROOT

EVIDENCE_ROOT = (
    Path(".runtime/w8/live-minecraft")
    / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:8])
)
MANIFEST = PROJECT_ROOT / "workflows" / "minecraft" / "minecraft.workflow.md"


def _wait_for(predicate, timeout: float = 180.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
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
    def setUp(self):
        self.mcp_url = os.environ.get("MINECRAFT_MCP_URL", DEFAULT_MCP_URL)
        tools = build_minecraft_tools(self.mcp_url)
        try:
            names = tools["minecraft_list_tools"]()
        except Exception as error:  # a required endpoint is missing
            self.fail(f"The Minecraft MCP is not reachable at {self.mcp_url}: {error!r}")
        self.assertIn("minecraft_observe", names)
        self.tools = tools

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
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            ghost = Ghost(
                GhostConfig(user_name="Jesse", companion_name="Spamuel"),
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
                        tools_factory=lambda run_id, environment: build_minecraft_tools(self.mcp_url),
                    ),
                },
                clock=time.monotonic,
                task_root=root / "tasks",
                on_terminal=lambda receipt: consolidator.consolidate(
                    receipt["run_id"], receipt["artifact_path"],
                ),
            )
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
            self.assertIn(target["block"], world)
            screenshots = list((Path(status["artifact_path"]) / "screenshots").glob("*")) \
                if (Path(status["artifact_path"]) / "screenshots").is_dir() else []
            self.assertTrue(screenshots, "the body run wrote no screenshot evidence")

            # Memory recovery from a reopened checkpoint.
            self.assertTrue(_wait_for(lambda: ghost.consolidations, timeout=30))
            reopened = Ghost.from_checkpoint(root / "ghost.json")
            self.assertTrue(reopened.consolidations)
            consolidation = next(iter(reopened.consolidations.values()))
            self.assertEqual(consolidation["run_id"], run_id)
            self.assertTrue(consolidation["image_evidence"])

            # Status, update, and termination on a held run.
            held = supervisor.start(
                action_id="mc-action-2", workflow_name="minecraft",
                instructions="hold at the boundary",
                arguments={"goal": "hold", "body_id": "tdd_player", "wait_for_cancel": True},
            )
            held_id = held["run_id"]
            held_path = Path(held["artifact_path"])
            self.assertTrue(
                _wait_for(lambda: (held_path / "outbox" / "world.yaml").is_file()),
                "the held run never reached its artifact boundary",
            )
            updated = supervisor.update(held_id, "shorten the hold", request_id="w8-update")
            self.assertEqual(updated["revision"], 1)
            terminated = supervisor.terminate(held_id)
            self.assertEqual(terminated["state"], "cancelled")
            held_status = supervisor.status(held_id)
            self.assertTrue((Path(held_status["artifact_path"]) / "outbox" / "world.yaml").is_file(),
                            "termination did not preserve the world artifact")

            _save("minecraft.json", {
                "mcp_url": self.mcp_url,
                "target": target,
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
