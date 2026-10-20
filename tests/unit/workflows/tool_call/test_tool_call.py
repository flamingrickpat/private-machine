"""The tool_call workflow: configuration, registration, routing, pinned dispatch, report, and memory.

No test calls a model. The probe test opens one real HTTP listener on the
loopback interface; it does not speak the MCP protocol. The dispatch test
runs the real kernel, supervisor, and consolidator; without a model
connection the session stops at its ``blocked`` boundary before any
external effect.
"""
from __future__ import annotations

import hashlib
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
import socket
import struct
import threading
import time
import zlib
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

from pydantic import ValidationError

from pm.bus.bus import BusSystem
from pm.bus.envelopes import Envelope, new_id
from pm.llm.config import load_config
from pm.llm.mcp_config import McpConfigError, ToolCallConfig
from pm.memory.bash_view import MemoryBashTarget
from pm.paths import REPO_ROOT
from pm.records.media import record_media
from pm.runs.run_receipt import ManagedRunReceipt
from pm.runs.supervisor import ManagedRunSupervisor
from pm.runs.workflow_module import PROJECT_ROOT
from pm.utils import yaml_util
from tests.support import ManualClock, Recorder
from tests.unit.workflows.main_cognition.cycle_support import ScriptedModel, last_user_text, scripted_tools
from tests.unit.workflows.main_cognition.test_decisions import ghost_with_entities, say, user
from workflows.companion import _cognition_workflow_options, _managed_workflows
from workflows.main_cognition.host import MainCogLoop
from workflows.main_cognition.schemas import WorkflowOption
from workflows.memory_consolidation.artifact_consolidation import Consolidator
from workflows.self_management.host import SelfManagement
from workflows.tool_call.catalogue import ToolCatalogue
from workflows.tool_call.consolidation import tool_call_outcome
from workflows.tool_call.host import WORKFLOW_DESCRIPTION, build_workflow_spec
from workflows.tool_call.report import dump_report, reply_evidence, retain_interrupted_report
from workflows.tool_call.schemas import ArtifactClaim, ArtifactEvidence, ToolCallReport, ToolImage
from workflows.tool_call.steps import keep_tool_images, probe_mcp_servers, session_activity, split_tool_name

MANIFEST = PROJECT_ROOT / "workflows" / "main_cognition" / "main_cognition.workflow.md"
SELF_MANAGEMENT = PROJECT_ROOT / "workflows" / "self_management" / "self_management.workflow.md"
DECISION = "Make him that skiing cat picture, the silliest one you can, and show it off."
MISSING = "PM_TOOL_CALL_TEST_TOKEN_THAT_IS_NOT_SET"


def tools_config(headers: dict | None = None) -> ToolCallConfig:
    return ToolCallConfig.model_validate({"mcp_servers": [
        {"name": "demo", "url": "http://127.0.0.1:9/mcp", "headers": headers or {}}]})


def red_png(size: int = 4) -> bytes:
    """Return a valid solid red PNG."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    row = b"\x00" + bytes((220, 30, 30)) * size
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(row * size)) + chunk(b"IEND", b""))


def no_env_file():
    """Point the .env fallback at a missing file, so only the process environment counts."""
    return mock.patch.dict(os.environ, {"PM_ENV_FILE": str(Path(__file__).with_name("no-such.env"))})


class ConfigTests(unittest.TestCase):
    def test_missing_header_variable_is_an_explicit_config_error(self):
        config = tools_config({"Authorization": "Bearer ${" + MISSING + "}"})
        with no_env_file():
            os.environ.pop(MISSING, None)
            with self.assertRaisesRegex(McpConfigError, MISSING):
                config.check_environment()
            with self.assertRaisesRegex(McpConfigError, MISSING):
                build_workflow_spec(config)

    def test_header_reference_resolves_from_the_environment(self):
        config = tools_config({"Authorization": "Bearer ${" + MISSING + "}"})
        with no_env_file(), mock.patch.dict(os.environ, {MISSING: "secret"}):
            self.assertEqual(config.mcp_servers[0].resolved_headers(), {"Authorization": "Bearer secret"})
        self.assertNotIn("secret", repr(config))

    def test_model_config_parses_the_tool_call_section(self):
        data = yaml_util.safe_load((REPO_ROOT / "config.example.yaml").read_text(encoding="utf-8"))
        data["tool_call"] = {"timeout_seconds": 60, "mcp_servers": [
            {"name": "image_gen", "url": "http://127.0.0.1:9877/mcp", "request_timeout_seconds": 30}]}
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(yaml_util.safe_dump(data), encoding="utf-8")
            config = load_config(path)
        self.assertEqual(config.tool_call.timeout_seconds, 60)
        self.assertEqual([server.name for server in config.tool_call.mcp_servers], ["image_gen"])
        self.assertEqual(load_config(REPO_ROOT / "config.example.yaml").tool_call.mcp_servers, [])

    def test_server_names_are_unique_and_urls_are_http(self):
        with self.assertRaises(ValidationError):
            ToolCallConfig.model_validate({"mcp_servers": [{"name": "a", "url": "http://x/mcp"},
                                                           {"name": "a", "url": "http://y/mcp"}]})
        with self.assertRaises(ValidationError):
            ToolCallConfig.model_validate({"mcp_servers": [{"name": "a", "url": "file:///mcp"}]})


class RegistrationTests(unittest.TestCase):
    def test_tool_call_is_registered_only_with_mcp_servers(self):
        self.assertNotIn("tool_call", _managed_workflows())
        self.assertNotIn("tool_call", _managed_workflows(tool_call=ToolCallConfig()))
        spec = _managed_workflows(tool_call=tools_config())["tool_call"]
        self.assertFalse(spec.resumable)
        self.assertIsNotNone(spec.dispatch_context)
        self.assertIsNotNone(spec.on_interrupted)

    def test_routing_offers_tool_call_with_its_tools_only_when_registered(self):
        catalogue = SimpleNamespace(describe=lambda: "  - server demo:\n    - paint: Paint one picture.")
        with_tools = SimpleNamespace(workflows=_managed_workflows(tool_call=tools_config()))
        options = {item["name"]: item["description"] for item in _cognition_workflow_options(with_tools, catalogue)}
        self.assertTrue(options["tool_call"].startswith(WORKFLOW_DESCRIPTION))
        self.assertIn("paint: Paint one picture.", options["tool_call"])
        without = SimpleNamespace(workflows=_managed_workflows())
        self.assertNotIn("tool_call", {item["name"] for item in _cognition_workflow_options(without, catalogue)})

    def test_an_unreachable_server_is_unavailable_and_not_asked_again_at_once(self):
        clock = ManualClock(100.0)
        catalogue = ToolCatalogue(tools_config(), clock=clock)
        self.assertIn("server demo: unavailable now", catalogue.describe())
        with mock.patch("workflows.tool_call.catalogue._list_tools") as listing:
            self.assertIn("unavailable now", catalogue.describe())
            listing.assert_not_called()


def paint_route(messages) -> dict:
    """Route the decision to the tool session, with routed instructions that the dispatch must drop."""
    return {"action_type": "ToolCallAndReply", "workflow_name": "tool_call",
            "arguments": {"instructions": "a rewritten goal", "tool_goal": "a cat on skis"}}


class RoutingAndDispatchTests(unittest.TestCase):
    def test_tools_reach_route_only_and_the_run_reads_the_pinned_decision(self):
        ghost = ghost_with_entities()
        jesse = say(ghost, "Jesse", "Can you make me a cool image of a skiing cat?")
        model = ScriptedModel({"decide": {"variants": [DECISION]}, "route": paint_route})
        catalogue = [WorkflowOption(name="dialogue_story", description="speak"),
                     WorkflowOption(name="tool_call", description=WORKFLOW_DESCRIPTION
                                    + "\n  - server demo:\n    - paint_secret_tool: Paint one picture.")]
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            bus = BusSystem()
            errors: list[BaseException] = []
            module = MainCogLoop("main_cog", bus, errors.append, ManualClock(), None,
                                 working_target=MemoryBashTarget(ghost, "read"), workflow=MANIFEST,
                                 tools_factory=scripted_tools(model), runtime_root=root / "runtime", ghost=ghost)
            module.workflow_catalogue_provider = lambda: catalogue
            bus.subscribe(module.module_id, module.accepts, module.deliver)
            requests = Recorder(bus, "action.request")
            module.handle_batch([user(jesse, "c-jesse")])
            self.assertEqual(errors, [])
            prompts = {phase: [last_user_text(messages) for name, messages in zip(model.calls, model.messages)
                               if name == phase] for phase in ("decide", "route")}
            self.assertTrue(prompts["decide"] and prompts["route"])
            self.assertTrue(all("paint_secret_tool" not in text and "tool_call" not in text
                                for text in prompts["decide"]))
            self.assertTrue(all("paint_secret_tool" in text for text in prompts["route"]))
            envelope = requests.events[0]
            self.assertEqual(envelope.target, "tool_call")
            self.assertEqual(envelope.payload["content"], DECISION)
            self.assertEqual(envelope.payload["arguments"], {"tool_goal": "a cat on skis"})

            # Self-management turns the action into a start; the supervisor runs the real workflow.
            control = SelfManagement("self_management", bus, errors.append, time.monotonic, None,
                                     working_target=None, workflow=SELF_MANAGEMENT,
                                     tools_factory=lambda environment, batch: {}, runtime_root=root / "sm",
                                     ghost=ghost)
            control.managed_workflows = frozenset({"tool_call"})
            start = control.prepare_input([envelope])
            consolidator = Consolidator(ghost, clock=time.time)
            supervisor = ManagedRunSupervisor(
                ghost, runtime_root=root / "runs", workflows={"tool_call": build_workflow_spec(tools_config())},
                clock=time.time, task_root=root / "tasks",
                on_terminal=lambda receipt: consolidator.consolidate(receipt["run_id"], receipt["artifact_path"]))
            receipt = supervisor.start(action_id=start["action_id"], workflow_name="tool_call",
                                       instructions=start["instructions"], parent_correlation=start["parent_correlation"],
                                       arguments=start["arguments"])
            supervisor.join(receipt["run_id"])
            receipt = supervisor.status(receipt["run_id"])
            artifacts = Path(receipt["artifact_path"])
            request = yaml_util.safe_load((artifacts / "inbox" / "request.yaml").read_text(encoding="utf-8"))
            report = ToolCallReport.model_validate(
                yaml_util.safe_load((artifacts / "outbox" / "report.yaml").read_text(encoding="utf-8")))

            # The instruction is the decision text, unchanged, and the context is the source cycle.
            self.assertEqual(request["instruction"], DECISION)
            self.assertEqual(request["tool_goal"], "a cat on skis")
            context = request["tool_call_context"]
            source = context["source_action"]
            self.assertEqual(context["context_source"], "dispatch")
            self.assertEqual(source["decision"], DECISION)
            self.assertEqual(source["action_id"], envelope.payload["action_id"])
            self.assertEqual(source["action_schema_id"], envelope.payload["knoxel_id"])
            self.assertEqual(source["tick_id"], envelope.payload["gwt_tick"])
            self.assertEqual(source["correlation_id"], "c-jesse")
            self.assertEqual([item["knoxel_id"] for item in source["stimuli"]], [jesse])
            self.assertIn("skiing cat", source["shard_story"] + "".join(item["text"] for item in source["stimuli"]))
            self.assertEqual(receipt["action_id"], envelope.payload["action_id"])
            self.assertEqual(receipt["parent_correlation"], "c-jesse")

            # No model connection: a blocked report, a failed receipt, and the summary on the event.
            self.assertEqual(report.outcome, "blocked")
            self.assertFalse(report.accepted)
            self.assertEqual((report.action_schema_id, report.correlation_id, report.stimulus_ids),
                             (envelope.payload["knoxel_id"], "c-jesse", [jesse]))
            self.assertEqual(receipt["state"], "failed")
            self.assertEqual(receipt["events"][-1]["message"]["detail"], report.summary)

            # Consolidation links the record to the action and stays idempotent.
            records = [item for item in ghost.all_features if "tool_call_report" in item.metadata]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].metadata["source_action_schema_id"], envelope.payload["knoxel_id"])
            self.assertEqual(records[0].metadata["action_id"], envelope.payload["action_id"])
            count = len(ghost.all_knoxels)
            again = consolidator.consolidate(receipt["run_id"], receipt["artifact_path"])
            self.assertEqual(len(ghost.all_knoxels), count)
            self.assertIn(records[0].id, again["memory_ids"])
            supervisor.recover_interrupted()
            self.assertEqual(len(ghost.all_knoxels), count)


def report_with_image(root: Path, run_id: str = "run-1") -> ToolCallReport:
    """Write a tool image and the same bytes as a claimed local file; return the report."""
    data = red_png()
    digest = hashlib.sha256(data).hexdigest()
    (root / "outbox").mkdir(parents=True)
    (root / "outbox" / "tool-image.png").write_bytes(data)
    (root / "outbox" / "red.png").write_bytes(data)
    return ToolCallReport(
        run_id=run_id, instruction="make a red test image", outcome="succeeded", summary="Made it.",
        accepted=True, tool_images=[ToolImage(path="outbox/tool-image.png", server="demo", tool="make_test_image",
                                              media_type="image/png", sha256=digest, size_bytes=len(data))],
        artifacts=[ArtifactEvidence(claim=ArtifactClaim(path="outbox/red.png", media_type="image/png", server="demo"),
                                    file_verified=True, sha256=digest, size_bytes=len(data))],
        action_id="action-1", action_schema_id=7, correlation_id="c-1")


class McpProbeTests(unittest.TestCase):
    """The reachability probe splits answering servers from unreachable ones."""

    def test_a_listening_server_is_live_and_a_refused_port_is_dead(self):
        server = HTTPServer(("127.0.0.1", 0), _RefuseHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            live_url = f"http://127.0.0.1:{server.server_port}/mcp"
            dead_url = f"http://127.0.0.1:{_closed_port()}/mcp"
            config = ToolCallConfig.model_validate({"mcp_servers": [
                {"name": "up", "url": live_url}, {"name": "down", "url": dead_url}]})
            self.assertEqual(probe_mcp_servers(config.mcp_servers, timeout_seconds=2.0), (["up"], ["down"]))
        finally:
            server.shutdown()
            server.server_close()

    def test_report_keeps_configured_live_and_dead_server_names(self):
        report = ToolCallReport(run_id="r", instruction="i", outcome="blocked", summary="s",
                                mcp_server_names=["image_gen", "pm_minecraft"],
                                mcp_live_servers=["pm_minecraft"], mcp_dead_servers=["image_gen"])
        self.assertEqual(ToolCallReport.model_validate(yaml_util.safe_load(dump_report(report))).mcp_dead_servers,
                         ["image_gen"])


class _RefuseHandler(BaseHTTPRequestHandler):
    """Answer every request with 400. Any HTTP status means the server listens."""

    def do_GET(self) -> None:  # noqa: N802 - the http.server name
        self.send_response(400)
        self.end_headers()

    def log_message(self, *args) -> None:
        """Keep the test output clean."""


def _closed_port() -> int:
    """Return a loopback port that no process listens on."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class ReportAndMemoryTests(unittest.TestCase):
    def test_report_schema(self):
        report = ToolCallReport(run_id="r", instruction="i", outcome="interrupted", summary="s")
        dumped = report.model_dump(mode="json", by_alias=True)
        self.assertEqual(dumped["schema"], "pm.tool-call-report.v1")
        self.assertFalse(dumped["verified"])
        self.assertEqual(ToolCallReport.model_validate(dumped), report)
        with self.assertRaises(ValidationError):
            ToolCallReport.model_validate({**dumped, "verified": True})
        with self.assertRaises(ValidationError):
            ToolCallReport.model_validate({**dumped, "outcome": "done"})
        for path in ("../x.png", "outbox/a/b.png", "C:/x.png", "x.png"):
            with self.assertRaises(ValidationError):
                ArtifactClaim(path=path)
        with self.assertRaises(ValidationError):
            ArtifactClaim(path="outbox/x.png", uri="mcp://demo/x.png")

    def test_tool_image_is_stored_on_the_outcome_record_through_the_media_api(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = report_with_image(root)
            record = tool_call_outcome("run-1", dump_report(report), root)
        media = record_media(record)
        self.assertEqual(len(media.images), 1)
        image = media.images[0]
        self.assertEqual(image.data(), red_png())
        self.assertEqual((image.provenance.source, image.provenance.server, image.provenance.tool,
                          image.provenance.run_id, image.provenance.artifact_path),
                         ("tool", "demo", "make_test_image", "run-1", "outbox/tool-image.png"))
        self.assertEqual(record.content.count("[image:"), 1)
        self.assertTrue(record.content.endswith(image.tag))
        self.assertNotIn(image.data_b64, record.content)
        self.assertEqual(record.metadata["source_action_schema_id"], 7)

    def test_session_record_gives_calls_and_images(self):
        from pydantic_ai.messages import (BinaryImage, ModelMessagesTypeAdapter, ModelRequest, ModelResponse,
                                          ToolCallPart, ToolReturnPart)
        data = red_png()
        messages = [ModelResponse(parts=[ToolCallPart(tool_name="demo_make_test_image", args={"color": "red"})]),
                    ModelRequest(parts=[ToolReturnPart(tool_name="demo_make_test_image",
                                                       content=[BinaryImage(data=data, media_type="image/png")])]),
                    ModelResponse(parts=[ToolCallPart(tool_name="read", args={"path": "x"})])]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "messages.json"
            path.write_bytes(ModelMessagesTypeAdapter.dump_json(messages))
            calls, images = session_activity(path, ["demo"])
            environment = SimpleNamespace(written={}, write_binary=lambda name, value: environment.written.update({name: value}))
            kept = keep_tool_images(environment, images + images)
        self.assertEqual([(call.server, call.tool) for call in calls], [("demo", "make_test_image")])
        self.assertEqual(len(kept), 1)
        self.assertEqual(environment.written[kept[0].path], data)
        self.assertEqual(split_tool_name("image_gen_paint", ["image", "image_gen"]), ("image_gen", "paint"))

    def test_reply_evidence_rechecks_local_bytes_and_stays_unverified(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = report_with_image(root)
            (root / "outbox" / "report.yaml").write_text(dump_report(report), encoding="utf-8")
            receipt = {"artifact_path": str(root)}
            evidence = reply_evidence(receipt)
            self.assertFalse(evidence["task_verified"])
            self.assertTrue(all(check["unchanged"] for check in evidence["file_checks"]))
            (root / "outbox" / "red.png").write_bytes(b"changed")
            checks = {check["path"]: check["unchanged"] for check in reply_evidence(receipt)["file_checks"]}
            self.assertEqual(checks, {"outbox/red.png": False, "outbox/tool-image.png": True})

    def test_interrupted_run_gets_one_report_and_one_memory_across_restarts(self):
        ghost = ghost_with_entities()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            consolidator = Consolidator(ghost, clock=time.time)
            supervisor = ManagedRunSupervisor(
                ghost, runtime_root=root / "runs", workflows={"tool_call": build_workflow_spec(tools_config())},
                clock=time.time, task_root=root / "tasks",
                on_terminal=lambda receipt: consolidator.consolidate(receipt["run_id"], receipt["artifact_path"]))
            artifacts = root / "tasks" / "run-9"
            batch = ghost.begin_batch("interrupted-run")
            batch.set_run_receipt("run-9", ManagedRunReceipt(
                run_id="run-9", action_id="action-9", workflow_name="tool_call", parent_correlation="c-9",
                state="interrupted", latest_event="restart", artifact_path=str(artifacts),
                instructions=DECISION, arguments={"tool_goal": "a cat"}).to_dict())
            ghost.commit(batch)
            supervisor.recover_interrupted()
            report = ToolCallReport.model_validate(
                yaml_util.safe_load((artifacts / "outbox" / "report.yaml").read_text(encoding="utf-8")))
            self.assertEqual((report.outcome, report.instruction, report.action_id), ("interrupted", DECISION, "action-9"))
            count = len(ghost.all_knoxels)
            text = (artifacts / "outbox" / "report.yaml").read_text(encoding="utf-8")
            retain_interrupted_report(supervisor.status("run-9"))
            supervisor.recover_interrupted()
            self.assertEqual((artifacts / "outbox" / "report.yaml").read_text(encoding="utf-8"), text)
            self.assertEqual(len(ghost.all_knoxels), count)
            records = [item for item in ghost.all_features if "tool_call_report" in item.metadata]
            self.assertEqual(len(records), 1)
            self.assertIn("interrupted", records[0].content)


if __name__ == "__main__":
    unittest.main()
