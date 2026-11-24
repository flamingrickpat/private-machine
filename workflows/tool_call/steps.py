"""Execute one tool goal in one native session and retain its terminal report.

The Python role holds one model lease and one pm-coder session with every
configured MCP server for the whole goal. The session can call tools,
inspect results, and revise its work. After the session stops, the host
reads the session record for the tool calls and the images that tools
returned, keeps those images as outbox files, checks every local artifact
that the agent claims, and writes ``outbox/report.yaml``.

The workflow sends no public speech and commits no memory. The supervisor's
terminal hook consolidates the report, and the reply path reads it.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import nullcontext
from workflows.minecraft.runtime import MinecraftWorkflowRuntime
from workflows.minecraft_body.lease import body_key
from workflows.minecraft_body.gateway import SessionBodyGateway
from pm.host.mcp_listeners import ScopedMcpRegistry
from dataclasses import replace, dataclass
import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import TypedDict, cast
from collections.abc import Sequence
from pydantic_ai.messages import BinaryContent, ToolReturnContent
from pydantic import ValidationError
from pm_workflows.python_role import RoleContext
from pm.host.folder_target import FolderEnvironment
from pm.runs.role_runtime import PMWorkflowRuntime
from pm.runs.role_io import read_request
from pm.runs.managed_boundary import BoundaryReadSettings
from pm.context_render.prompt import RenderSettings, RenderedPrompt, RendererCapture, RendererInput
from pm.context_render.turns import Text, Media, MediaKind
from pm.llm.mcp_config import McpServerConfig
from workflows.tool_call.rendering import ToolSessionRenderInput, render_tool_session
from workflows.minecraft_body.world_text import MinecraftToolResult, WorldObservation, coordinates, item_text

import httpx
from pm_workflows.drivers import PmCoderDriver
from pm_workflows.inputs import InputError
from pm_workflows.runtime import WorkflowTerminated

from pm.llm.structured import repair_agent_result
from pm.llm.multimodal import ImageAttachment
from pm.constants import CYCLE_RETRY_DELAY_SECONDS, CYCLE_RETRY_MAX_DELAY_SECONDS
from pm.llm.mcp_config import ToolCallConfig
from pm.records.media import IMAGE_TYPES

from workflows.tool_call.report import dump_report
from workflows.tool_call.schemas import (
    REPORT_PATH, AgentOutcome, ArtifactClaim, ArtifactEvidence, ToolCallReport, ToolCallRequest, ToolCallSeen,
    ToolImage,
)

#: File extension of each image type that the host keeps from a tool result.
IMAGE_SUFFIXES = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp", "image/gif": ".gif"}
SKILL = Path(__file__).with_name("resources") / "execute.md"

#: Seconds to wait for one MCP server to answer the reachability probe.
MCP_PROBE_TIMEOUT_SECONDS = 3.0


@dataclass(frozen=True, slots=True)
class McpProbeSettings:
    timeout_seconds: float = MCP_PROBE_TIMEOUT_SECONDS


@dataclass(frozen=True, slots=True)
class McpProbeResult:
    live: tuple[str, ...]
    dead: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ToolName:
    server: str
    tool: str


@dataclass(frozen=True, slots=True)
class ToolBinaryImage:
    server: str
    tool: str
    data: bytes
    media_type: str


@dataclass(frozen=True, slots=True)
class ToolSessionActivity:
    calls: tuple[ToolCallSeen, ...]
    images: tuple[ToolBinaryImage, ...]


class ToolRoleResult(TypedDict):
    status: str
    summary: str


def probe_mcp_servers(servers: list[McpServerConfig], settings: McpProbeSettings = McpProbeSettings()) -> McpProbeResult:
    """Return the names of the MCP servers that answer, and the names of those that do not.

    The probe sends one HTTP request to each server URL and reads only the
    transport result. Any HTTP status means the server listens. A connect
    error or a connect timeout means it does not. The probe does not run the
    MCP handshake, so it cannot prove that a server offers usable tools.

    The native session opens every server in its MCP file at once, and one
    server that refuses the connection stops the whole session. The caller
    therefore keeps only the servers this probe returns as live.
    """
    live: list[str] = []
    dead: list[str] = []
    for server in servers:
        try:
            httpx.get(server.url, timeout=settings.timeout_seconds)
        except httpx.TransportError:
            dead.append(server.name)
        else:
            live.append(server.name)
    return McpProbeResult(tuple(live), tuple(dead))


def inspect_artifacts(environment: FolderEnvironment, claims: list[ArtifactClaim]) -> list[ArtifactEvidence]:
    """Read local file metadata after the agent stops. Never fetch external URIs.

    Local files must stay inside the run's outbox after symlink resolution.
    Missing and empty files stay explicit evidence errors. A claim of the
    host report path is ignored, because the host owns that file. This read does not
    inspect pixels or prove that the goal succeeded.
    """
    evidence: list[ArtifactEvidence] = []
    for claim in claims:
        item = ArtifactEvidence(claim=claim)
        if claim.path is None:
            item.reason = "The agent supplied an external reference. The host did not inspect its bytes."
        elif claim.path == REPORT_PATH:
            # The host owns this file and overwrites it. It is never an artifact.
            item.reason = "This is the host report path, not an artifact. The host ignored the claim."
        else:
            try:
                environment.check_file_boundary(claim.path, "outbox")
                metadata = environment.file_metadata(claim.path)
                item.size_bytes = metadata.size_bytes
                item.sha256 = metadata.content_sha256
                item.file_verified = metadata.size_bytes > 0
                if item.file_verified:
                    item.local_path = str((environment.root / environment.canonical_relative_path(claim.path)).resolve())
                    item.reason = "The host read nonempty local bytes. Content and quality remain unverified."
                else:
                    item.reason = "The local artifact is empty."
            except (OSError, ValueError, InputError) as error:
                item.reason = f"The host did not inspect the local artifact ({type(error).__name__})."
        evidence.append(item)
    return evidence


def split_tool_name(name: str, servers: list[str]) -> ToolName:
    """Split a native tool name ``<server>_<tool>`` into its server and tool.

    The longest matching server name wins, so ``image`` and ``image_gen``
    stay distinct. A name without a server prefix is a built-in tool of the
    session and gets an empty server.
    """
    for server in sorted(servers, key=len, reverse=True):
        if name.startswith(server + "_"):
            return ToolName(server, name[len(server) + 1:])
    return ToolName("", name)


def _files(content: ToolReturnContent) -> Iterator[BinaryContent]:
    """Yield the binary attachments of one content value, preserving order."""
    """Yield every binary file inside one message content value, in order."""

    if isinstance(content, BinaryContent):
        yield content
    elif isinstance(content, list):
        children = cast(Sequence[ToolReturnContent], content)
        for item in children:
            yield from _files(item)


def session_activity(messages_path: Path, servers: list[str]) -> ToolSessionActivity:
    """Return the MCP calls and the tool images of one native session record.

    The images are ``(server, tool, bytes, media type)`` in session order.
    The session record is the pm-coder ``messages.json``. A compaction can
    replace older messages with a summary; their images are then no longer
    in the record (documented limit). A missing record means no activity.
    """
    from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart

    if not messages_path.is_file():
        return ToolSessionActivity((), ())
    calls: list[ToolCallSeen] = []
    by_id: dict[str, ToolCallSeen] = {}
    images: list[ToolBinaryImage] = []
    for message in ModelMessagesTypeAdapter.validate_json(messages_path.read_bytes()):
        for part in message.parts:
            if isinstance(message, ModelResponse) and isinstance(part, ToolCallPart):
                name = split_tool_name(part.tool_name, servers)
                if name.server:
                    call = ToolCallSeen(server=name.server, tool=name.tool, arguments=part.args_as_json_str(),
                                        call_id=part.tool_call_id)
                    calls.append(call)
                    by_id[part.tool_call_id] = call
            elif isinstance(message, ModelRequest) and isinstance(part, ToolReturnPart):
                name = split_tool_name(part.tool_name, servers)
                if name.server:
                    call = by_id.get(part.tool_call_id)
                    if call is not None:
                        call.returned_facts = minecraft_return_facts(name.tool, part.content)
                    images += [ToolBinaryImage(name.server, name.tool, item.data, item.media_type) for item in _files(part.content)
                               if item.media_type in IMAGE_TYPES]
    return ToolSessionActivity(tuple(calls), tuple(images))


def minecraft_return_facts(tool: str, content: ToolReturnContent) -> str:
    supported = {"minecraft_observe", "minecraft_mine_block", "minecraft_drop_item", "minecraft_build",
                 "minecraft_inspect_block", "minecraft_walk_to_exact", "minecraft_walk_to_visible"}
    if tool not in supported:
        return ""
    parts: Sequence[ToolReturnContent] = cast(Sequence[ToolReturnContent], content) if isinstance(content, list) else [content]
    for value in parts:
        if not isinstance(value, dict) or "ok" not in value:
            continue
        try:
            if tool == "minecraft_observe" and value["ok"]:
                world = WorldObservation.model_validate(value)
                inventory = ", ".join(item_text(item) for item in world.inventory) or "empty"
                return (f"feet {coordinates(world.camera.feet_position)}, inventory {inventory}, "
                        f"health {world.player.health}, deaths {world.death_count}")
            return MinecraftToolResult.model_validate(value).facts()
        except ValidationError as error:
            logging.getLogger(__name__).warning("Cannot summarize %s server return: %s invalid fields.", tool, error.error_count())
            return "Unsupported response shape. The session retains the original server return; the effect is unverified."
    return ""


def keep_tool_images(environment: FolderEnvironment, images: tuple[ToolBinaryImage, ...]) -> list[ToolImage]:
    """Write each tool image once to ``outbox/tool-image-<digest>.<ext>`` and describe it.

    The file name comes from the bytes, so the same image is kept once and a
    repeated write changes nothing.
    """
    kept: dict[str, ToolImage] = {}
    for image in images:
        digest = hashlib.sha256(image.data).hexdigest()
        if digest in kept:
            continue
        path = f"outbox/tool-image-{digest[:16]}{IMAGE_SUFFIXES[image.media_type]}"
        environment.write_binary(path, image.data)
        kept[digest] = ToolImage(path=path, server=image.server, tool=image.tool, media_type=image.media_type,
                                 sha256=digest, size_bytes=len(image.data))
    return list(kept.values())


@dataclass(frozen=True, slots=True)
class NativeToolPrompt:
    prompt: str
    images: tuple[ImageAttachment, ...]


def encode_native_tool_prompt(rendered: RenderedPrompt) -> NativeToolPrompt:
    text: list[str] = []
    images: list[ImageAttachment] = []
    for part in rendered.turns[1].content.parts:
        if isinstance(part, Text):
            text.append(part.text)
        elif isinstance(part, Media) and part.kind is MediaKind.Image:
            if part.data is None:
                raise ValueError("Native tool-session images require captured bytes, not a URI.")
            images.append(ImageAttachment(part.data, part.mime_type))
        else:
            raise TypeError("The native tool-session adapter accepts text and image attachments only.")
    return NativeToolPrompt("".join(text), tuple(images))


def _session_mcp_file(path: Path, config: ToolCallConfig) -> None:
    """Write the native MCP file of the session, with resolved headers.

    The file holds credentials. It stays in the ignored kernel data folder
    and the caller deletes it when the session ends. The native loader
    connects only to the servers in this file: no workspace discovery.
    """
    servers = {server.name: {"url": server.url, "headers": server.resolved_headers(),
                             "requestTimeoutMs": server.request_timeout_seconds * 1000}
               for server in config.mcp_servers}
    path.write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")


def _execute(context: RoleContext, report: ToolCallReport, *, private: bool) -> None:
    """Hold one model lease and one native session through the complete goal.

    The native loader owns the MCP connections and tool prefixes throughout
    the session. The workflow does not retry a session: a timeout can follow
    a completed external effect. An instruction change during the session
    makes the result uncertain; it does not roll back earlier effects.
    """
    runtime = context.runtime
    assert isinstance(runtime, PMWorkflowRuntime)
    environment = runtime.environment
    assert isinstance(environment, FolderEnvironment)
    request = read_request(environment, ToolCallRequest)
    if request.run_id != context.task_id:
        raise ValueError("The request run identifier does not match the workflow invocation.")
    tool_context = request.tool_call_context
    source = tool_context.source_action
    report.instruction = request.instruction
    report.tool_goal = request.tool_goal
    report.related_run_id = tool_context.related_run_id
    report.context_source = tool_context.context_source
    report.context_tick_id = tool_context.captured_tick_id
    report.context_turn_ids = [turn.tick_id for turn in tool_context.gwt_turns]
    report.context_record_revisions = {record.record_id: record.revision for record in tool_context.records}
    if source is not None:
        report.action_id = source.action_id
        report.action_schema_id = source.action_schema_id
        report.source_tick_id = source.tick_id
        report.correlation_id = source.correlation_id
        report.stimulus_ids = [item.knoxel_id for item in source.stimuli]
    config = runtime.tool_call_config
    assert config is not None
    report.mcp_server_names = [server.name for server in config.mcp_servers]
    if not config.mcp_servers:
        report.outcome = "blocked"
        report.summary = "No MCP servers are configured for this tool goal."
        report.rejection_reason = report.summary
        return
    # One unreachable server stops the native session before it makes any
    # tool call. Probe after the cheap checks, keep the live servers, and
    # name the dead ones in the report.
    boundary = runtime.managed_boundary
    assert boundary is not None
    instructions = boundary.read()
    report.instruction_revision = instructions["revision"]
    report.instruction = instructions["instruction"] or request.instruction
    runtime.check_cancelled()
    if runtime.agent_model is None and runtime.classification_session is None:
        report.outcome = "blocked"
        report.summary = "The tool goal needs a configured model connection."
        report.rejection_reason = report.summary
        return
    probe = probe_mcp_servers(config.mcp_servers)
    delay = CYCLE_RETRY_DELAY_SECONDS
    attempt = 1
    while not probe.live:
        logging.getLogger(__name__).warning("No MCP answered for run %s, attempt %d, retrying in %gs: %s",
                                            context.task_id, attempt, delay, ", ".join(probe.dead))
        runtime.stop_event.wait(delay)
        runtime.check_cancelled()
        probe = probe_mcp_servers(config.mcp_servers)
        delay = min(delay * 2, CYCLE_RETRY_MAX_DELAY_SECONDS)
        attempt += 1
    live = list(probe.live)
    dead = list(probe.dead)
    report.mcp_live_servers = live
    report.mcp_dead_servers = dead
    if dead:
        config = ToolCallConfig(model_connection=config.model_connection, timeout_seconds=config.timeout_seconds,
            gwt_turn_limit=config.gwt_turn_limit, mcp_servers=[server for server in config.mcp_servers if server.name in live])
    servers = live

    mcp_file = context.kernel_data / "tool-call-mcp.json"
    log_root = context.kernel_data / "tool-call-agent"
    session_name = context.task_id + "-tool"
    if runtime.agent_model is not None:
        session = nullcontext(runtime.agent_model)
    else:
        assert runtime.classification_session is not None
        session = runtime.classification_session(config.model_connection, False, runtime.stop_event)
    listeners = ScopedMcpRegistry()
    scoped_body = None
    gateway: SessionBodyGateway | None = None
    from workflows.minecraft_body.world_text import StopObservation
    interrupted_at: StopObservation | None = None
    suite_status: list[str] = []
    try:
        configured_servers: list[McpServerConfig] = []
        proxy_url: str | None = None
        for server in config.mcp_servers:
            if isinstance(runtime, MinecraftWorkflowRuntime) and body_key(server.url) == body_key(runtime.body_lease.endpoint):
                scoped_body = runtime.body_lease.ownership
                assert scoped_body is not None
                owner = scoped_body.owner()
                paused = scoped_body.paused_task()
                if paused is not None and paused.run_id == context.task_id:
                    interrupt = environment.root / f"body-stop-{paused.epoch}.json"
                    interrupted_at = StopObservation.model_validate_json(interrupt.read_bytes())
                state = f"busy with {owner}; use self_management.status, wait or terminate it" if owner else "available on first body action"
                suite_status.append(f"{server.name} {state}")
                if proxy_url is None:
                    gateway = SessionBodyGateway(runtime.body_lease.endpoint, context.task_id, environment, runtime, visual=True)
                    schemas = [row for row in gateway.tool_schemas if row["name"] not in
                               {"minecraft_execute_typescript", "minecraft_suicide", "minecraft_connect", "minecraft_disconnect"}]
                    proxy_url = listeners.serve({"body": gateway.server(schemas)})["body"]["url"]
                configured_servers.append(McpServerConfig(name=server.name, url=proxy_url,
                    request_timeout_seconds=server.request_timeout_seconds))
            else:
                suite_status.append(f"{server.name} available")
                configured_servers.append(server)
        config = config.model_copy(update={"mcp_servers": configured_servers})
        _session_mcp_file(mcp_file, config)
        with session as model:
            assert model is not None
            assert runtime.model_session is not None
            connection = runtime.model_session.model.connection
            assert connection.completion_token_limit is not None
            settings = RenderSettings(context_window=model.context_window,
                input_fraction=connection.context_input_fraction,
                reserved_output_tokens=connection.completion_token_limit)
            from workflows.assistant.session import PRIVATE_INSTRUCTIONS, private_session
            session_instructions = PRIVATE_INSTRUCTIONS if private else SKILL.read_text(encoding="utf-8")
            session_instructions += "\nMCP suite status: " + ", ".join(suite_status)
            if interrupted_at is not None:
                session_instructions += ("\nResume the remaining body task from the retained world. "
                    "Read current state before acting. Do not repeat a command with an uncertain outcome. "
                    "Previous command receipts: " + str(environment.root / "scratch" / "actor-journal.jsonl"))
                session_instructions += " Confirmed stop: " + interrupted_at.reading.summary()
                for effect in interrupted_at.uncertain_effects:
                    session_instructions += "\n" + effect
            if dead:
                session_instructions += "\nUnavailable MCP services in this session: " + ", ".join(dead) + "."
            render_input = ToolSessionRenderInput(request, report.instruction, tuple(servers),
                "" if private else str(environment.root.resolve()), session_instructions)
            capture_settings = runtime.model_session.capture_settings("tool.session")
            with RendererCapture(capture_settings, RendererInput("tool.session", render_input, settings)) as capture:
                rendered = render_tool_session(render_input, settings)
                capture.save_output(rendered)
            native_prompt = encode_native_tool_prompt(rendered)
            prompt = native_prompt.prompt
            if private:
                session_result = asyncio.run(private_session(model, mcp_file, log_root, session_name + "_1",
                    native_prompt, runtime, config.timeout_seconds))
            else:
                agent_runtime = replace(runtime, agent_model=replace(model, enable_thinking=True))
                driver = PmCoderDriver(log_root=log_root, timeout_seconds=config.timeout_seconds)
                session_result = driver.run_session(
                    run_id=session_name, attempt=1, skill=str(SKILL), prompt=prompt,
                    images=native_prompt.images,
                    work_dir=context.workspace, runtime=agent_runtime, mcp_config=mcp_file,
                    trace_file=Path(report.trace_path), result_file=context.kernel_data / "tool-call-agent-result.json")
    finally:
        mcp_file.unlink(missing_ok=True)
        if gateway is not None:
            gateway.close_admission()
        listeners.close()
        if scoped_body is not None:
            scoped_body.finish(context.task_id, runtime.stop_event.is_set(), environment.root)
        # Tool calls and images before an error or a cancellation are real
        # effects too. Keep them in the report in every case.
        activity = session_activity(log_root / f"{session_name}_1" / "messages.json", servers)
        report.tool_calls = list(activity.calls)
        report.tool_images = keep_tool_images(environment, activity.images)
    runtime.check_cancelled()
    if isinstance(session_result, str):
        final = boundary.read(BoundaryReadSettings(close=True))
        report.final_instruction_revision = final["revision"]
        report.agent_session = session_name + "_1"
        report.summary = session_result
        report.accepted = final["revision"] == report.instruction_revision
        report.outcome = "succeeded" if report.accepted else "uncertain"
        return
    report.agent_session = session_result.session_ref
    if session_result.exit_code != 0:
        report.outcome = "uncertain"
        report.summary = "The tool session failed. Its external effects remain unverified."
        report.rejection_reason = report.summary
        return

    outcome = repair_agent_result(session_result, AgentOutcome, model, instruction=prompt, cancel=runtime.stop_event)
    report.agent_outcome = outcome
    report.artifacts = inspect_artifacts(environment, outcome.artifacts)
    report.outcome = outcome.outcome
    report.summary = outcome.summary
    if outcome.outcome == "succeeded" and any(
        item.claim.path not in {None, REPORT_PATH} and not item.file_verified for item in report.artifacts
    ):
        report.outcome = "uncertain"
        report.summary = "The agent reported success. At least one claimed local file did not pass host inspection."

    runtime.check_cancelled()
    final = boundary.read(BoundaryReadSettings(close=True))
    report.final_instruction_revision = final["revision"]
    if final["revision"] != report.instruction_revision:
        report.outcome = "uncertain"
        report.summary = "The goal changed during the session. The retained result describes the earlier goal."
    report.accepted = report.outcome == "succeeded"
    report.rejection_reason = "" if report.accepted else report.summary


def _write_report(context: RoleContext, report: ToolCallReport) -> None:
    """Persist host evidence without a memory commit or public message.

    The supervisor owns the terminal state and consolidation after this
    write. A failed write propagates: a missing report cannot establish
    completion.
    """
    assert context.runtime is not None
    context.runtime.environment.write_text(REPORT_PATH, dump_report(report))


def run(context: RoleContext) -> ToolRoleResult:
    return execute_goal(context, private=False)


def execute_goal(context: RoleContext, *, private: bool) -> ToolRoleResult:
    """Return a terminal role result for expected task failures and successes.

    Cancellation retains a cancelled report, then raises ``WorkflowTerminated``.
    Unexpected exceptions retain a report, then propagate with their traceback.
    External effects can survive either case. No retry or rollback occurs here.
    """
    assert context.runtime is not None
    report = ToolCallReport(
        run_id=context.task_id, instruction="", outcome="uncertain",
        summary="The tool goal did not produce a usable result.",
        trace_path=str(context.kernel_data / "tool-call-agent-trace.jsonl"),
    )
    try:
        context.runtime.check_cancelled()
        _execute(context, report, private=private)
        context.runtime.check_cancelled()
    except WorkflowTerminated:
        report.outcome = "cancelled"
        report.accepted = False
        report.summary = "The tool goal was cancelled. External effects can remain."
        report.rejection_reason = report.summary
        _write_report(context, report)
        raise
    except Exception as error:
        report.outcome = "uncertain"
        report.accepted = False
        report.error_type = type(error).__name__
        report.summary = "The tool goal stopped after an unexpected engineering error. Its effects remain unverified."
        report.rejection_reason = report.summary
        _write_report(context, report)
        raise
    _write_report(context, report)
    return {"status": "done", "summary": report.summary}
