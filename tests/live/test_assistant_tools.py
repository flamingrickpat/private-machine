"""Live Assistant tool-calling check against the configured model.

Real model, real MCP management tools, real local managed run. The model is
asked to read a run's status and must actually execute the granted tool. The
reply is committed only when a receipt supports its claim.
"""
import json
import os
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import yaml

from pm.bus.bus import BusSystem
from pm.bus.envelopes import Envelope
from modules import AssistantDialogue
from pm_core.ghost.service import Ghost
from pm_core.model.ghost_state import GhostConfig
from runtime.cognitive_model import LazyStageModel
from runtime.managed_runs import ManagedRunSupervisor, WorkflowSpec
from runtime.model_activation import ModelActivation
from runtime.models import LlmPreset, load_config, resolve_connection
from tests.support import Recorder
from workflows.dialogue_assistant.module_tools import (
    build_assistant_tools,
)
from pm.host.workflow_module import PROJECT_ROOT
from pm.memory.bash_view import GhostBashWrapper

EVIDENCE_ROOT = Path(".runtime/w6/live-assistant") / (
    datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:8]
)
ASSISTANT_MANIFEST = (
    PROJECT_ROOT / "workflows" / "dialogue_assistant" / "dialogue_assistant.workflow.md"
)
LOCAL_TASK = PROJECT_ROOT / "workflows" / "local_task"


def _wait_for(predicate, timeout=20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class LiveAssistantTests(unittest.TestCase):
    def setUp(self):
        default = "config.local.yaml" if Path("config.local.yaml").is_file() else "config.example.yaml"
        self.config = load_config(os.environ.get("PM_MODEL_CONFIG", default))

    def test_assistant_executes_a_real_management_tool(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            ghost = Ghost(GhostConfig(user_name="Jesse", companion_name="Spamuel"))
            supervisor = ManagedRunSupervisor(
                ghost, runtime_root=root / "runtime",
                workflows={
                    "local_task": WorkflowSpec(
                        LOCAL_TASK / "local_task.workflow.md", namespace="local_task",
                    ),
                },
                clock=time.monotonic, task_root=root / "tasks",
            )
            started = supervisor.start(
                action_id="action-0", workflow_name="local_task",
                instructions="live assistant check",
            )
            run_id = started["run_id"]
            self.assertTrue(_wait_for(lambda: supervisor.status(run_id)["state"] == "completed"))

            model = LazyStageModel(resolve_connection(self.config), ModelActivation())
            bus = BusSystem()
            module = AssistantDialogue(
                "assistant", bus, self.fail, time.monotonic, None,
                working_target=GhostBashWrapper(ghost, mode="read"),
                workflow=ASSISTANT_MANIFEST,
                tools_factory=build_assistant_tools(model, supervisor),
                runtime_root=root / "runtime", ghost=ghost,
            )
            bus.subscribe(module.module_id, module.accepts, module.deliver)
            replies = Recorder(bus, "dialogue.reply")
            receipts = Recorder(bus, "action.receipt")
            prompt = (
                f"Please call the managed_status tool for run {run_id}, "
                "then tell me exactly what state the record reports."
            )
            module.mailbox.put(Envelope(
                "req", "test", "data", "action.request", "corr-live",
                {
                    "action_id": "action-1",
                    "content": prompt,
                    "arguments": {},
                    "stimuli": [{"knoxel_id": 0, "text": prompt}],
                },
                target="dialogue_assistant",
            ))
            self.assertTrue(module.step(), "assistant module did not process the action")

            report = yaml.safe_load(
                module.last_environment.read_text("outbox/report.yaml")
            )
            executed = [
                item for item in report["tool_receipts"]
                if item["executed"] and item["name"] == "managed_status"
            ]
            EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
            (EVIDENCE_ROOT / "assistant.json").write_text(
                json.dumps({
                    "model": self.config.chat[LlmPreset.Default].model,
                    "run_id": run_id,
                    "final_text": report["final_text"],
                    "tool_receipts": report["tool_receipts"],
                    "dialogue_replies": len(replies.events),
                    "action_receipts": len(receipts.events),
                }, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            self.assertTrue(executed, "the assistant did not execute managed_status")
            self.assertTrue(report["final_text"].strip())
            self.assertTrue(replies.events, "no dialogue.reply was published")
            supervisor.join_all(timeout=10)


if __name__ == "__main__":
    unittest.main()
