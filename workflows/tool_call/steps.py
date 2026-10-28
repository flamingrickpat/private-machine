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
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import traceback
from typing import Any

import httpx
from pm_workflows.drivers import PmCoderDriver
from pm_workflows.inputs import InputError
from pm_workflows.runtime import WorkflowTerminated

from pm.llm.http_client import ModelError
from pm.llm.structured import repair_agent_result
from pm.llm.multimodal import agent_images
from pm.llm.mcp_config import ToolCallConfig
from pm.records.media import IMAGE_TYPES
from pm.runs.errors import TaskServiceError
from pm.utils import yaml_util

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


def probe_mcp_servers(servers: list[Any], timeout_seconds: float = MCP_PROBE_TIMEOUT_SECONDS) -> tuple[list[str], list[str]]:
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
            httpx.get(server.url, timeout=timeout_seconds)
        except httpx.TransportError:
            dead.append(server.name)
        else:
            live.append(server.name)
    return live, dead


def inspect_artifacts(environment: Any, claims: list[ArtifactClaim]) -> list[ArtifactEvidence]:
    """Read local file metadata after the agent stops. Never fetch external URIs.

    Local files must stay inside the run's outbox after symlink resolution.
    Missing and empty files stay explicit evidence errors. A claim of the
    host report path is ignored, because the host owns that file. This read does not
    inspect pixels or prove that the goal succeeded.
    """
    evidence = []
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
                    item.local_path = str(environment._path(claim.path).resolve())
                    item.reason = "The host read nonempty local bytes. Content and quality remain unverified."
                else:
                    item.reason = "The local artifact is empty."
            except (OSError, ValueError, InputError) as error:
                item.reason = f"The host did not inspect the local artifact ({type(error).__name__})."
        evidence.append(item)
    return evidence


def split_tool_name(name: str, servers: list[str]) -> tuple[str, str]:
    """Split a native tool name ``<server>_<tool>`` into its server and tool.

    The longest matching server name wins, so ``image`` and ``image_gen``
    stay distinct. A name without a server prefix is a built-in tool of the
    session and gets an empty server.
    """
    for server in sorted(servers, key=len, reverse=True):
        if name.startswith(server + "_"):
            return server, name[len(server) + 1:]
    return "", name


def _files(content: Any) -> Iterator[Any]:
    """Yield the binary attachments of one content value, preserving order."""
    """Yield every binary file inside one message content value, in order."""
    from pydantic_ai.messages import BinaryContent

    if isinstance(content, BinaryContent):
        yield content
    elif isinstance(content, list):
        for item in content:
            yield from _files(item)


def session_activity(messages_path: Path, servers: list[str]) -> tuple[list[ToolCallSeen], list[tuple[str, str, bytes, str]]]:
    """Return the MCP calls and the tool images of one native session record.

    The images are ``(server, tool, bytes, media type)`` in session order.
    The session record is the pm-coder ``messages.json``. A compaction can
    replace older messages with a summary; their images are then no longer
    in the record (documented limit). A missing record means no activity.
    """
    from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart

    if not messages_path.is_file():
        return [], []
    calls: list[ToolCallSeen] = []
    images: list[tuple[str, str, bytes, str]] = []
    for message in ModelMessagesTypeAdapter.validate_json(messages_path.read_bytes()):
        for part in message.parts:
            if isinstance(message, ModelResponse) and isinstance(part, ToolCallPart):
                server, tool = split_tool_name(part.tool_name, servers)
                if server:
                    calls.append(ToolCallSeen(server=server, tool=tool, arguments=part.args_as_json_str()))
            elif isinstance(message, ModelRequest) and isinstance(part, ToolReturnPart):
                server, tool = split_tool_name(part.tool_name, servers)
                if server:
                    images += [(server, tool, item.data, item.media_type) for item in _files(part.content)
                               if item.media_type in IMAGE_TYPES]
    return calls, images


def keep_tool_images(environment: Any, images: list[tuple[str, str, bytes, str]]) -> list[ToolImage]:
    """Write each tool image once to ``outbox/tool-image-<digest>.<ext>`` and describe it.

    The file name comes from the bytes, so the same image is kept once and a
    repeated write changes nothing.
    """
    kept: dict[str, ToolImage] = {}
    for server, tool, data, media_type in images:
        digest = hashlib.sha256(data).hexdigest()
        if digest in kept:
            continue
        path = f"outbox/tool-image-{digest[:16]}{IMAGE_SUFFIXES[media_type]}"
        environment.write_binary(path, data)
        kept[digest] = ToolImage(path=path, server=server, tool=tool, media_type=media_type,
                                 sha256=digest, size_bytes=len(data))
    return list(kept.values())


def session_prompt(request: ToolCallRequest, instruction: str, servers: list[str], work_root: Path) -> str:
    """Return the first prompt of the session: the companion, its decision, and its context.

    The companion's own parts are story-framed: the session is the
    companion's own work on its own decision. The tool rules stay in the
    skill text (``resources/execute.md``), which can be plain.
    """
    context = request.tool_call_context
    lines = [f"You are {context.companion_name}. You decided to do something, and this session is where you do "
             "it with your own tools.", "", "Who you are:", context.character_card.strip(), "",
             "Your decision, in your own words:", f'"{instruction}"']
    if request.tool_goal:
        lines += ["", f"The concrete goal for your tools: {request.tool_goal}"]
    source = context.source_action
    if source is not None:
        events = [f"- ({item.kind}, from {item.source}) {item.text}" for item in source.stimuli]
        lines += ["", "What led to this decision:", *(events or ["- (no new event)"])]
        if source.affect:
            lines += ["", f"How you felt: {source.affect}"]
        if source.workspace:
            lines += ["", "What you lived through in that moment:", source.workspace]
        if source.shard_story:
            lines += ["", "What was in your mind then:", source.shard_story]
    else:
        lines += ["", "No moment of your mind started this session directly. Below is your newest context at "
                      "the start of the session."]
    if context.gwt_turns:
        lines += ["", "Your recent moments, oldest first:",
                  *(f"- {turn.summary}" for turn in context.gwt_turns if turn.summary.strip())]
    lines += ["", f"Your MCP tool servers: {', '.join(servers)}. Each tool name starts with its server name and "
                  "an underscore.",
              f"Your work folder on this host: {work_root}. Put files for the host directly into outbox/ there.",
              "Return only the terminal JSON object that the instructions specify."]
    return "\n".join(lines)


def session_images(request: ToolCallRequest) -> list[tuple[bytes, str]]:
    """Return the ``(bytes, media_type)`` pictures of the moment that started this session.

    The session prompt carries the alt text of each stimulus. The session is
    the companion acting on what it saw, so the pictures themselves ride on
    its first turn. A moment with no image tags gets no attachments.
    """
    source = request.tool_call_context.source_action
    if source is None:
        return []
    return agent_images("\n".join(item.text for item in source.stimuli))


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


def _execute(context: Any, report: ToolCallReport) -> None:
    """Hold one model lease and one native session through the complete goal.

    The native loader owns the MCP connections and tool prefixes throughout
    the session. The workflow does not retry a session: a timeout can follow
    a completed external effect. An instruction change during the session
    makes the result uncertain; it does not roll back earlier effects.
    """
    runtime = context.runtime
    environment = runtime.environment
    request = ToolCallRequest.model_validate(yaml_util.safe_load(environment.read_text("inbox/request.yaml")))
    if request.run_id != context.task_id:
        raise ValueError("The request run identifier does not match the workflow invocation.")
    tool_context = request.tool_call_context
    source = tool_context.source_action
    report.instruction = request.instruction
    report.tool_goal = request.tool_goal
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
    config: ToolCallConfig = runtime.tools["tool_call_settings"]()
    report.mcp_server_names = [server.name for server in config.mcp_servers]
    if not config.mcp_servers:
        report.outcome = "blocked"
        report.summary = "No MCP servers are configured for this tool goal."
        report.rejection_reason = report.summary
        return
    # One unreachable server stops the native session before it makes any
    # tool call. Probe after the cheap checks, keep the live servers, and
    # name the dead ones in the report.
    instructions = runtime.tools["managed_instructions"]()
    report.instruction_revision = instructions["revision"]
    report.instruction = instructions["instruction"] or request.instruction
    runtime.check_cancelled()
    if runtime.agent_model is None and runtime.classification_session is None:
        report.outcome = "blocked"
        report.summary = "The tool goal needs a configured model connection."
        report.rejection_reason = report.summary
        return
    live, dead = probe_mcp_servers(config.mcp_servers)
    report.mcp_live_servers = live
    report.mcp_dead_servers = dead
    if not live:
        report.outcome = "blocked"
        report.summary = ("No configured MCP server answers: " + ", ".join(report.mcp_server_names)
                          + (f". Unreachable: {', '.join(dead)}." if dead else "."))
        report.rejection_reason = report.summary
        return
    if dead:
        config = config.model_copy(update={"mcp_servers": [s for s in config.mcp_servers if s.name in live]})
    servers = live

    mcp_file = context.kernel_data / "tool-call-mcp.json"
    log_root = context.kernel_data / "tool-call-agent"
    session_name = context.task_id + "-tool"
    session = (nullcontext(runtime.agent_model) if runtime.agent_model is not None
               else runtime.classification_session(config.model_connection, False, runtime.stop_event))
    prompt = session_prompt(request, report.instruction, servers, environment.root.resolve())
    result = None
    try:
        _session_mcp_file(mcp_file, config)
        with session as model:
            agent_runtime = replace(runtime, agent_model=replace(model, enable_thinking=True))
            driver = PmCoderDriver(log_root=log_root, timeout_seconds=config.timeout_seconds)
            result = driver.run_session(
                run_id=session_name, attempt=1, skill=str(SKILL), prompt=prompt,
                images=session_images(request),
                work_dir=context.workspace, runtime=agent_runtime, mcp_config=mcp_file,
                trace_file=Path(report.trace_path), result_file=context.kernel_data / "tool-call-agent-result.json")
    except WorkflowTerminated:
        raise
    except Exception as error:
        # Provider and MCP errors use several exception families. Keep the
        # diagnostic private and give control back to the companion.
        (context.kernel_data / "tool-call-agent-error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        report.error_type = type(error).__name__
        report.outcome = "uncertain"
        report.summary = "The tool session stopped after an error. Its external effects remain unverified."
        report.rejection_reason = report.summary
    finally:
        mcp_file.unlink(missing_ok=True)
        # Tool calls and images before an error or a cancellation are real
        # effects too. Keep them in the report in every case.
        calls, images = session_activity(log_root / f"{session_name}_1" / "messages.json", servers)
        report.tool_calls = calls
        report.tool_images = keep_tool_images(environment, images)
    if result is None:
        return
    runtime.check_cancelled()
    report.agent_session = result.session_ref
    if result.exit_code != 0:
        report.outcome = "uncertain"
        report.summary = "The tool session failed. Its external effects remain unverified."
        report.rejection_reason = report.summary
        return

    outcome = repair_agent_result(result, AgentOutcome, model, instruction=prompt, cancel=runtime.stop_event)
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
    final = runtime.tools["managed_instructions"](close=True)
    report.final_instruction_revision = final["revision"]
    if final["revision"] != report.instruction_revision:
        report.outcome = "uncertain"
        report.summary = "The goal changed during the session. The retained result describes the earlier goal."
    report.accepted = report.outcome == "succeeded"
    report.rejection_reason = "" if report.accepted else report.summary


def _write_report(context: Any, report: ToolCallReport) -> None:
    """Persist host evidence without a memory commit or public message.

    The supervisor owns the terminal state and consolidation after this
    write. A failed write propagates: a missing report cannot establish
    completion.
    """
    context.runtime.environment.write_text(REPORT_PATH, dump_report(report))


def run(context: Any) -> dict[str, str]:
    """Return a terminal role result for expected task failures and successes.

    Cancellation retains a cancelled report, then raises ``WorkflowTerminated``.
    Unexpected exceptions retain a report, then propagate with their traceback.
    External effects can survive either case. No retry or rollback occurs here.
    """
    report = ToolCallReport(
        run_id=context.task_id, instruction="", outcome="uncertain",
        summary="The tool goal did not produce a usable result.",
        trace_path=str(context.kernel_data / "tool-call-agent-trace.jsonl"),
    )
    try:
        context.runtime.check_cancelled()
        _execute(context, report)
        context.runtime.check_cancelled()
    except WorkflowTerminated:
        report.outcome = "cancelled"
        report.accepted = False
        report.summary = "The tool goal was cancelled. External effects can remain."
        report.rejection_reason = report.summary
        _write_report(context, report)
        raise
    except (InputError, ModelError, TaskServiceError, httpx.HTTPError,
            ValueError, OSError, TimeoutError, yaml_util.YAMLError) as error:
        report.outcome = "uncertain"
        report.accepted = False
        report.error_type = type(error).__name__
        report.summary = "The tool goal stopped after an input, service, or artifact error. Its effects remain unverified."
        report.rejection_reason = report.summary
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
