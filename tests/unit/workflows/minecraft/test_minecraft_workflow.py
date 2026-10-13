"""Minecraft body workflow tests with a deterministic MCP boundary.

The MCP call is injected; the workflow contract, evidence files, and
consolidation are real. A live pm-minecraft MCP is covered separately.
"""
from __future__ import annotations

import base64
import threading
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pm_workflows.runtime import WorkflowRuntime

from pm.memory.store import MemoryStore
from pm.records.cycle_state import PersonaConfig
from workflows.memory_consolidation.artifact_consolidation import Consolidator
from pm.runs.workflow_runner import PROJECT_ROOT, WorkflowRequest, WorkflowRunner, WorkflowStatus
from workflows.minecraft import steps
from workflows.minecraft.schemas import MinecraftReport, MinecraftRequest
from pm.host.folder_target import FolderEnvironment
from workflows.minecraft_body.lease import BODY_LEASES
from pm.records.base import persisted_type_name

MANIFEST = PROJECT_ROOT / "workflows" / "minecraft" / "minecraft.workflow.md"
PNG = b"\x89PNG\r\n\x1a\n" + b"deterministic-minecraft-frame"
PNG_B64 = base64.b64encode(PNG).decode("ascii")


def observation(frame: str = "f1") -> dict:
    return {
        "tool": "minecraft_observe", "arguments": {"include_image": True}, "ok": True,
        "result": {
            "player": {"biome": "plains", "dimension": "overworld", "health": 20, "food": 20},
            "camera": {"feet_position": {"x": 0.5, "y": 1.0, "z": 0.5}},
            "nearby_blocks": [{
                "block_name": "oak_log", "display_name": "Oak Log",
                "count": 3, "nearest": {"x": 5, "y": 1, "z": 3}, "distance": 4.0,
            }],
            "image": {"frame_id": frame, "width": 64, "height": 64},
        },
        "images": [{"media_type": "image/png", "base64": PNG_B64}],
    }


def action(ok: bool = True) -> dict:
    return {
        "tool": "minecraft_mine_block", "arguments": {"position": {"x": 5, "y": 1, "z": 3}},
        "ok": ok,
        "result": {"player": {"biome": "plains", "dimension": "overworld"}, "camera": {}},
        "images": [],
    }


class FakeMinecraft:
    def __init__(self, *, action_ok: bool = True):
        self.action_ok = action_ok
        self.calls: list[tuple] = []

    def __call__(self, name, arguments=None):
        self.calls.append((name, arguments))
        if name == "minecraft_observe":
            return observation()
        return action(self.action_ok)


class MinecraftTestBase(unittest.TestCase):
    def _run(self, request: dict, minecraft, root: Path):
        request = {"mode": "boundary_check", **request}
        tools = {
            "minecraft_endpoint": lambda: "fixture://boundary",
            "minecraft_lease": lambda: BODY_LEASES.acquire("fixture://boundary", "minecraft-1"),
            "minecraft_call": minecraft,
            "minecraft_plan": lambda observation, allowed_tools, goal: (allowed_tools[0]["name"] if allowed_tools else ""),
            "minecraft_list_tools": lambda: ["minecraft_observe", "minecraft_mine_block"],
            "minecraft_tool_schemas": lambda: [{"name":"minecraft_mine_block", "parameters":{"type":"object"}}],
            "managed_instructions": lambda **kwargs: {"instruction": request.get("instruction", ""), "revision":0},
        }
        environment = FolderEnvironment(root / "task", "bash")
        runtime = WorkflowRuntime(
            environment=environment, stop_event=threading.Event(),
            tools=tools, propagate_python_errors=True,
        )
        runner = WorkflowRunner(MANIFEST, root / "runtime", namespace="minecraft")
        result = runner.invoke(WorkflowRequest("minecraft-1", request), runtime)
        self.assertEqual(result.status, WorkflowStatus.Completed)
        return environment, MinecraftReport.model_validate(result.report)


class MinecraftWorkflowTests(MinecraftTestBase):
    def test_observe_and_action_write_real_evidence(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            minecraft = FakeMinecraft()
            request = {
                "goal": "chop one oak log",
                "body_id": "big_floppa",
                "action_tool": "minecraft_mine_block",
                "action_arguments": {"position": {"x": 5, "y": 1, "z": 3}},
            }
            environment, report = self._run(request, minecraft, root)
            self.assertTrue(report.accepted)
            self.assertEqual([call[0] for call in minecraft.calls],
                             ["minecraft_observe", "minecraft_mine_block", "minecraft_observe"])
            # Real screenshot bytes are on disk.
            screenshot = root / "task" / report.images[0]["path"]
            self.assertTrue(screenshot.is_file())
            self.assertEqual(screenshot.read_bytes(), PNG)
            world_text = environment.read_text("outbox/world.yaml")
            self.assertIn("places", world_text)
            self.assertIn("oak_log", world_text)
            self.assertIn("episodes", world_text)
            observations = environment.read_text("scratch/observations.md")
            self.assertIn("biome=plains", observations)

    def test_a_failed_action_is_not_accepted(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            minecraft = FakeMinecraft(action_ok=False)
            request = {"goal": "mine", "action_tool": "minecraft_mine_block"}
            _environment, report = self._run(request, minecraft, root)
            self.assertFalse(report.accepted)
            self.assertIn("failure", report.rejection_reason)

    def test_planner_selects_an_allowed_tool(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            minecraft = FakeMinecraft()
            request = {"goal": "mine", "allowed_tools": ["minecraft_mine_block"]}
            _environment, report = self._run(request, minecraft, root)
            self.assertEqual([call[0] for call in minecraft.calls],
                             ["minecraft_observe", "minecraft_mine_block", "minecraft_observe"])
            self.assertTrue(report.accepted)


class MinecraftConsolidationTests(MinecraftTestBase):
    def test_world_and_screenshot_become_spatial_memory(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            minecraft = FakeMinecraft()
            request = {"goal": "chop one oak log", "action_tool": "minecraft_mine_block"}
            environment, report = self._run(request, minecraft, root)

            ghost = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"))
            consolidator = Consolidator(ghost, clock=lambda: 10.0)
            receipt = consolidator.consolidate("minecraft-1", root / "task")
            self.assertTrue(receipt["memory_ids"])
            knoxels = [ghost.get_knoxel_by_id(mid) for mid in receipt["memory_ids"]]
            kinds = {persisted_type_name(type(k)) for k in knoxels}
            self.assertIn("WorldPlaceKnoxel", kinds)
            self.assertIn("WorldObjectKnoxel", kinds)
            self.assertIn("BodyEpisodeKnoxel", kinds)
            episode = next(k for k in knoxels if persisted_type_name(type(k)) == "BodyEpisodeKnoxel")
            self.assertEqual(episode.evidence["images"][0]["media_type"], "image/png")
            self.assertEqual(
                base64.b64decode(episode.evidence["images"][0]["base64"]), PNG,
            )
            # The task folder is preserved after durable storage succeeds.
            self.assertTrue((root / "task" / "outbox" / "world.yaml").is_file())


class PlanParsingTests(unittest.TestCase):
    def test_request_ignores_unknown_model_arguments(self):
        # A model action can add a helpful but undeclared argument. The boundary
        # request keeps the declared fields and drops the hint.
        request = MinecraftRequest.model_validate({
            "instruction": "Observe only.", "observe_only": True, "fields": ["position"],
        })
        self.assertEqual(request.instruction, "Observe only.")
        self.assertFalse(hasattr(request, "observe_only"))

    def test_parse_plan_splits_tool_and_json(self):
        self.assertEqual(
            steps.parse_plan('minecraft_mine_block {"position": {"x": 1}}'),
            ("minecraft_mine_block", {"position": {"x": 1}}),
        )
        self.assertEqual(steps.parse_plan("minecraft_observe"), ("minecraft_observe", {}))

    def test_parse_plan_rejects_empty_or_non_object(self):
        with self.assertRaises(ValueError):
            steps.parse_plan("")
        with self.assertRaises(ValueError):
            steps.parse_plan('minecraft_observe [1, 2]')

    def test_summarize_is_bounded_text(self):
        line = steps.summarize(observation())
        self.assertIn("biome=plains", line)
        self.assertIn("oak_log", line)
        self.assertNotIn(PNG_B64, line)


if __name__ == "__main__":
    unittest.main()
