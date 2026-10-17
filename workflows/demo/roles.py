"""Fixed cognitive behavior for the six prototype roles.

Each function reads ``inbox/request.yaml`` through ``context.runtime.environment``,
writes a semantic report to ``outbox/report.yaml``, and returns the kernel result
object. The fixed output is the prototype behavior, not evidence of LLM wiring.
"""
from __future__ import annotations

from pm.utils import yaml_util


def _environment(context):
    return context.runtime.environment


def _read_request(context) -> dict:
    return yaml_util.safe_load(_environment(context).read_text("inbox/request.yaml"))


def _write_report(context, report: dict) -> None:
    env = _environment(context)
    env.write_text("scratch/report.yaml", yaml_util.safe_dump(report, sort_keys=False))
    env.move("scratch/report.yaml", "outbox/report.yaml")


def _message_content(message: dict) -> str:
    return str(message.get("knoxel", {}).get("data", {}).get("content", ""))


def main_cognition(context) -> dict:
    request = _read_request(context)
    messages = request.get("messages", [])
    decisions: list[dict] = []
    for message in messages:
        if message.get("topic") != "user.input":
            continue
        correlation_id = message.get("correlation_id", "")
        text = _message_content(message)
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
            decisions.append({
                "correlation_id": correlation_id,
                "route": "tool_dialogue",
                "content": "",
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
                "content": "You are a dumb human, Jesse! Dumb and nasty.",
                "task_id": "",
                "task_kind": "",
            })
        else:
            decisions.append({
                "correlation_id": correlation_id,
                "route": "dialogue",
                "content": "I heard you, Jesse.",
                "task_id": "",
                "task_kind": "",
            })

    report = {
        "summary": f"Dummy cognition processed {len(messages)} inputs.",
        "decisions": decisions,
    }
    _write_report(context, report)
    return {"status": "done", "summary": "dummy main cognition completed"}


def dialogue(context) -> dict:
    request = _read_request(context)
    replies = [
        {"correlation_id": m.get("correlation_id", ""), "content": _message_content(m)}
        for m in request.get("messages", [])
    ]
    _write_report(context, {"replies": replies})
    return {"status": "done", "summary": "dummy dialogue completed"}


def inner_thought(context) -> dict:
    _write_report(context, {"thought": "I wonder what Jesse will ask next."})
    return {"status": "done", "summary": "dummy inner thought completed"}


def schedule(context) -> dict:
    request = _read_request(context)
    tasks = []
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


def status(context) -> dict:
    request = _read_request(context)
    queries = []
    for message in request.get("messages", []):
        data = message.get("knoxel", {}).get("data", {})
        metadata = data.get("metadata", {})
        queries.append({
            "correlation_id": message.get("correlation_id", ""),
            "task_id": metadata.get("task_id", ""),
        })
    _write_report(context, {"queries": queries})
    return {"status": "done", "summary": "dummy status completed"}


def tool_dialogue(context) -> dict:
    request = _read_request(context)
    tools = context.runtime.tools
    count = len(tools["ghost_search"]("Jesse"))
    replies = [
        {
            "correlation_id": m.get("correlation_id", ""),
            "content": f"I found {count} memories about Jesse.",
        }
        for m in request.get("messages", [])
    ]
    _write_report(context, {"replies": replies})
    return {"status": "done", "summary": "dummy tool dialogue completed"}


__all__ = ["main_cognition", "dialogue", "inner_thought", "schedule", "status", "tool_dialogue"]