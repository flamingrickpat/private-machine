"""Real agent reports on application targets, with retained runtime evidence."""
import asyncio
import json
from pathlib import Path
from threading import Event
import unittest
from uuid import uuid4

import yaml
from pydantic import TypeAdapter
from pm_workflows.runtime import AgentModel, WorkflowRuntime

from pm.utils.operating_system import OperatingSystem
from pm.host.mcp_listeners import ScopedMcpRegistry
from pm.llm.activation import ModelActivation
from pm.llm.config import load_config, resolve_connection
from pm.llm.session import ModelSession
from pm.llm.stage_model import LazyStageModel
from pm.runs.role_runtime import PMWorkflowRuntime
from pm.runs.workflow_runner import PROJECT_ROOT, WorkflowRequest, WorkflowRunner, WorkflowStatus
from pm.memory.store_tools import build_mcp
from pm.host.folder_target import FolderEnvironment, FolderTarget
from pm.memory.bash_view import MemoryBashEnvironment
from workflows.note_task.steps import NoteRejectionReport, NoteRequest


class LiveAgentWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        default = "config.local.yaml" if Path("config.local.yaml").is_file() else "config.example.yaml"
        cls.connection = resolve_connection(load_config(OperatingSystem.environment.get("PM_MODEL_CONFIG", default)))
        cls.owner = ModelActivation()
        asyncio.run(cls.owner.activate(cls.connection, live_test=True))

    def test_note_worker_rejects_a_request_without_a_note_write(self) -> None:
        root = (Path(".runtime/note-task") / uuid4().hex).resolve()
        model = LazyStageModel(self.connection, self.owner, live_test=True)
        environment = FolderEnvironment(root / "task", "bash")
        cancel = Event()
        runtime = PMWorkflowRuntime(environment=environment, stop_event=cancel, tools={},  # pm-lint: allow PM005 -- pm-workflows requires an external tool map for each role
            model_session=ModelSession(model, cancel), propagate_python_errors=True)
        manifest = PROJECT_ROOT / "workflows/note_task/note_task.workflow.md"
        runner = WorkflowRunner(manifest, root / "runtime")
        note = NoteRequest(instruction="Thanks for listening. I feel better.")
        payload = note.model_dump(by_alias=True)  # pm-lint: allow PM007 -- kernel writes the request as inbox YAML
        result = runner.invoke(WorkflowRequest("note-selection", payload), runtime)
        self.assertEqual(result.status, WorkflowStatus.Completed)
        self.assertEqual(environment.list_files("outbox"), ["outbox/report.yaml"])
        report = TypeAdapter(NoteRejectionReport).validate_python(result.report)
        self.assertIs(report["accepted"], False)
        self.assertIs(report["verified"], False)
        self.assertTrue(report["rejection_reason"])

    def run_target(self, virtual):
        root = (Path(".runtime/w3/agent-targets") / uuid4().hex).resolve()
        root.mkdir(parents=True)
        marker = "TARGET_" + uuid4().hex
        record = {"__type__": "Feature", "data": {"id": 1, "content": marker}}
        environment = (
            MemoryBashEnvironment({1: record}, {}, "read") if virtual
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

    def test_virtual_agent_uses_a_scoped_mcp_service(self):
        root = (Path(".runtime/w3/agent-targets") / uuid4().hex).resolve()
        root.mkdir(parents=True)
        marker = "MCP_" + uuid4().hex
        record = {"__type__": "Feature", "data": {"id": 1, "content": marker}}
        environment = MemoryBashEnvironment({1: record}, {}, "read")
        skill = root / "role.md"
        skill.write_text(
            "Do not read the memory file directly. Call the MCP tool `ghost_get` with "
            "knoxel_id equal to inbox/request.yaml's memory_id to read the memory. "
            "Use only the read tool and the write tool with relative paths for the rest. "
            "Write outbox/report.yaml as a YAML mapping with marker equal to that memory's data.content. "
            "Do not change the memory. Then return JSON with status done and summary copied.",
            encoding="utf-8")
        manifest = root / "probe.workflow.md"
        manifest.write_text("---\n" + yaml.safe_dump({
            "name": "mcp-probe", "driver": {"kind": "pm-coder"},
            "roles": {"work": {"skill": str(skill), "mcp": ["ghost"], "result_contract": {"schema": {
                "status": {"enum": ["done"]}, "summary": "string",
            }}}},
            "phases": [{"name": "work", "kind": "role", "role": "work",
                        "on_status": {"done": "stop"}, "on_invalid": {"action": "stop"}}],
        }) + "---\n", encoding="utf-8")
        connection = self.connection
        with ScopedMcpRegistry() as registry:
            descriptors = registry.serve({"ghost": build_mcp(environment)})
            runtime = WorkflowRuntime(
                environment, Event(), {},
                agent_model=AgentModel(
                    connection.base_url, connection.model, connection.api_key.get_secret_value(),
                    context_window=96000, enable_thinking=False, live_test=True,
                ),
                mcp_servers=descriptors,
            )
            runner = WorkflowRunner(manifest, root / "runtime", base_dir=root)
            with self.owner.use(connection):
                result = runner.invoke(
                    WorkflowRequest("probe", {"memory_id": 1}), runtime,
                )
        self.assertEqual(result.status, WorkflowStatus.Completed)
        self.assertEqual(result.report, {"marker": marker})
        self.assertEqual(yaml.safe_load(environment.read_text("ghost/knoxels/1.yaml")), record)
        config = list(result.artifact_dir.glob("mcp/*.json"))
        self.assertEqual(len(config), 1)
        servers = json.loads(config[0].read_text(encoding="utf-8"))["mcpServers"]
        self.assertIn("ghost", servers)
        sessions = list((result.artifact_dir / "pm-coder").glob("*/messages.json"))
        messages = json.loads(sessions[0].read_text(encoding="utf-8"))
        calls = {part["tool_name"] for message in messages for part in message["parts"] if part["part_kind"] == "tool-call"}
        self.assertTrue(any("ghost_get" in name for name in calls), calls)
        (root / "evidence.json").write_text(json.dumps({
            "target": "virtual", "model": connection.model, "status": result.status,
            "report": result.report, "mcp_calls": sorted(calls),
            "artifact_dir": str(result.artifact_dir),
        }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
