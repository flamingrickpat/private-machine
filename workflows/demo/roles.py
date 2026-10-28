"""Fixed cognitive behavior for the six prototype roles.

Each function reads ``inbox/request.yaml`` through ``context.runtime.environment``,
writes a semantic report to ``outbox/report.yaml``, and returns the kernel result
object. The fixed output is the prototype behavior, not evidence of LLM wiring.

The demo workflow manifests name these modules through ``role_skills``. Nothing
here calls a model, a memory store, or a body: the modules exist so the real
kernel, bus, and managed-run paths can be exercised with a known answer.

Reading map: ``main_cognition`` routes user input by keyword, ``dialogue``
echoes the accepted action content, ``schedule`` and ``status`` drive the
in-memory task table, ``tool_dialogue`` answers a memory search, and
``inner_thought`` emits one fixed thought.
"""
from __future__ import annotations

from typing import Any

from pm_workflows.python_role import RoleContext

from pm.utils import yaml_util


def _environment(context: RoleContext) -> Any:
    """Return the invocation environment the role may read and write."""
    return context.runtime.environment


def _read_request(context: RoleContext) -> dict[str, Any]:
    """Return the request mapping that the kernel wrote for this attempt."""
    return yaml_util.safe_load(_environment(context).read_text("inbox/request.yaml"))


def _write_report(context: RoleContext, report: dict[str, Any]) -> None:
    env = _environment(context)
    env.write_text("scratch/report.yaml", yaml_util.safe_dump(report, sort_keys=False))
    env.move("scratch/report.yaml", "outbox/report.yaml")


def _message_content(message: dict[str, Any]) -> str:
    """Return the content text of one request message."""
    return str(message.get("knoxel", {}).get("data", {}).get("content", ""))


def _speaker(message: dict[str, Any]) -> str:
    """Return the speaker name of a dialogue message. The persona supplies the name."""
    return str(message.get("knoxel", {}).get("data", {}).get("override_source") or "")


def main_cognition(context: RoleContext) -> dict[str, Any]:
    """Route demonstration user input by keyword and publish its decisions."""
    request = _read_request(context)
    messages = request.get("messages", [])
    decisions: list[dict[str, str]] = []
    for message in messages:
        if message.get("topic") != "user.input":
            continue
        correlation_id = message.get("correlation_id", "")
        text = _message_content(message)
        speaker = _speaker(message)
        lowered = text.strip().lower()

        if lowered.startswith("/status "):
            task_id = text.strip().split(" ", 1)[1]
            decisions.append({
                "correlation_id": correlation_id,
                "route": "status",
                "content": "",
                "task_id": task_id,
                "task_kind": "",
            })
        elif lowered == "/memory":
            # The tool dialogue searches memories about the speaker.
            decisions.append({
                "correlation_id": correlation_id,
                "route": "tool_dialogue",
                "content": speaker,
                "task_id": "",
                "task_kind": "",
            })
        elif "minecraft" in lowered:
            decisions.append({
                "correlation_id": correlation_id,
                "route": "schedule",
                "content": "",
                "task_id": "task-" + correlation_id,
                "task_kind": "minecraft",
            })
            decisions.append({
                "correlation_id": correlation_id,
                "route": "dialogue",
                "content": "I will schedule your Minecraft request.",
                "task_id": "",
                "task_kind": "",
            })
        elif lowered == "/demo-task":
            decisions.append({
                "correlation_id": correlation_id,
                "route": "schedule",
                "content": "",
                "task_id": "task-" + correlation_id,
                "task_kind": "dummy",
            })
            decisions.append({
                "correlation_id": correlation_id,
                "route": "dialogue",
                "content": "I will schedule a dummy task.",
                "task_id": "",
                "task_kind": "",
            })
        elif "dumb robot" in lowered:
            decisions.append({
                "correlation_id": correlation_id,
                "route": "dialogue",
                "content": f"You are a dumb human, {speaker}! Dumb and nasty.",
                "task_id": "",
                "task_kind": "",
            })
        else:
            decisions.append({
                "correlation_id": correlation_id,
                "route": "dialogue",
                "content": f"I heard you, {speaker}.",
                "task_id": "",
                "task_kind": "",
            })

    report = {
        "summary": f"Dummy cognition processed {len(messages)} inputs.",
        "decisions": decisions,
    }
    _write_report(context, report)
    return {"status": "done", "summary": "dummy main cognition completed"}


def dialogue(context: RoleContext) -> dict[str, Any]:
    """Echo each accepted action content as the demonstration reply."""
    request = _read_request(context)
    replies = [
        {"correlation_id": message.get("correlation_id", ""), "content": _message_content(message)}
        for message in request.get("messages", [])
    ]
    _write_report(context, {"replies": replies})
    return {"status": "done", "summary": "dummy dialogue completed"}


def inner_thought(context: RoleContext) -> dict[str, Any]:
    """Write the one fixed demonstration thought."""
    _write_report(context, {"thought": "I wonder what they will ask next."})
    return {"status": "done", "summary": "dummy inner thought completed"}


def schedule(context: RoleContext) -> dict[str, Any]:
    """Report the tasks that the demonstration decisions asked to schedule."""
    request = _read_request(context)
    tasks: list[dict[str, str]] = []
    for message in request.get("messages", []):
        data = message.get("knoxel", {}).get("data", {})
        metadata = data.get("metadata", {})
        tasks.append({
            "correlation_id": message.get("correlation_id", ""),
            "task_id": metadata.get("task_id", ""),
            "task_kind": metadata.get("task_kind", ""),
        })
    _write_report(context, {"tasks": tasks})
    return {"status": "done", "summary": "dummy scheduling completed"}


def status(context: RoleContext) -> dict[str, Any]:
    """Report the demonstration task that the request names."""
    request = _read_request(context)
    queries: list[dict[str, str]] = []
    for message in request.get("messages", []):
        data = message.get("knoxel", {}).get("data", {})
        metadata = data.get("metadata", {})
        queries.append({
            "correlation_id": message.get("correlation_id", ""),
            "task_id": metadata.get("task_id", ""),
        })
    _write_report(context, {"queries": queries})
    return {"status": "done", "summary": "dummy status completed"}


def tool_dialogue(context: RoleContext) -> dict[str, Any]:
    """Answer a memory search with the fixed demonstration text."""
    request = _read_request(context)
    tools = context.runtime.tools
    replies: list[dict[str, str]] = []
    for message in request.get("messages", []):
        # The demo main cognition put the speaker name into the action content.
        name = _message_content(message)
        count = len(tools["ghost_search"](name))
        replies.append({"correlation_id": message.get("correlation_id", ""),
                        "content": f"I found {count} memories about {name}."})
    _write_report(context, {"replies": replies})
    return {"status": "done", "summary": "dummy tool dialogue completed"}


__all__ = ["main_cognition", "dialogue", "inner_thought", "schedule", "status", "tool_dialogue"]