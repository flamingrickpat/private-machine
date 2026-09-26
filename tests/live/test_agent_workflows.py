"""Real agent reports on application targets, with retained runtime evidence."""
import asyncio
import json
import os
from pathlib import Path
from threading import Event
import unittest
from uuid import uuid4

import yaml
from pm_workflows.runtime import AgentModel, WorkflowRuntime

from runtime.model_activation import ModelActivation
from runtime.models import load_config, resolve_connection
from runtime.workflow_runner import WorkflowRequest, WorkflowRunner, WorkflowStatus
from working_target import FolderTarget, GhostBashEnvironment


class LiveAgentWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        default = "config.local.yaml" if Path("config.local.yaml").is_file() else "config.example.yaml"
        cls.connection = resolve_connection(load_config(os.environ.get("PM_MODEL_CONFIG", default)))
        cls.owner = ModelActivation()
        asyncio.run(cls.owner.activate(cls.connection, live_test=True))

    def run_target(self, virtual):
        root = (Path(".runtime/w3/agent-targets") / uuid4().hex).resolve()
        root.mkdir(parents=True)
        marker = "TARGET_" + uuid4().hex
        record = {"__type__": "Feature", "data": {"id": 1, "content": marker}}
        environment = (
            GhostBashEnvironment({1: record}, {}, "read") if virtual
            else FolderTarget(root / "target", "bash").open("probe")
        )
        memory = "ghost/knoxels/1.yaml"
        if not virtual:
            environment.write_text(memory, yaml.safe_dump(record))
        skill = root / "role.md"
        skill.write_text(
            "Use only the read and write tools with relative paths. Do not use shell commands or subagents. "
            "Read inbox/request.yaml, then read the file at its memory_path. "
            "Write outbox/report.yaml as a YAML mapping with marker equal to that memory's data.content. "
            "Do not change the memory. Then return JSON with status done and summary copied.", encoding="utf-8")
        manifest = root / "probe.workflow.md"
        manifest.write_text("---\n" + yaml.safe_dump({
            "name": "target-probe", "driver": {"kind": "pm-coder"},
            "roles": {"work": {"skill": str(skill), "result_contract": {"schema": {
                "status": {"enum": ["done"]}, "summary": "string",
            }}}},
            "phases": [{"name": "work", "kind": "role", "role": "work",
                        "on_status": {"done": "stop"}, "on_invalid": {"action": "stop"}}],
        }) + "---\n", encoding="utf-8")
        connection = self.connection
        runtime = WorkflowRuntime(environment, Event(), {}, agent_model=AgentModel(
            connection.base_url, connection.model, connection.api_key.get_secret_value(),
            context_window=96000, enable_thinking=False, live_test=True,
        ))
        runner = WorkflowRunner(manifest, root / "runtime", base_dir=root)
        with self.owner.use(connection):
            result = runner.invoke(WorkflowRequest("probe", {"memory_path": memory}), runtime)
        self.assertEqual(result.status, WorkflowStatus.Completed)
        self.assertEqual(result.report, {"marker": marker})
        self.assertEqual(yaml.safe_load(environment.read_text(memory)), record)
        self.assertFalse((root / "runtime/standalone/probe/workspace/outbox/report.yaml").exists())
        sessions = list((result.artifact_dir / "pm-coder").glob("*/messages.json"))
        self.assertEqual(len(sessions), 1)
        messages = json.loads(sessions[0].read_text(encoding="utf-8"))
        calls = {part["tool_name"] for message in messages for part in message["parts"] if part["part_kind"] == "tool-call"}
        self.assertTrue({"read", "write"} <= calls)
        (root / "evidence.json").write_text(json.dumps({
            "target": "virtual" if virtual else "folder", "model": connection.model,
            "status": result.status, "report": result.report, "tools": sorted(calls),
            "memory_unchanged": True, "artifact_dir": str(result.artifact_dir),
        }, indent=2), encoding="utf-8")

    def test_virtual_agent_report(self):
        self.run_target(True)

    def test_folder_agent_report(self):
        self.run_target(False)


if __name__ == "__main__":
    unittest.main()
