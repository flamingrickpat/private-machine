"""Assistant dialogue role.

This is a separate message contract from Story dialogue. A Python role keeps
the normal user, assistant, and tool-call structure, executes the granted
tools, appends their real results, and returns one final assistant utterance.
A denied tool never runs. A tool receipt is the only support for a claim that
a management operation happened.

``build_messages`` and the tool-call parsing are pure and tested without a
kernel, model, or MemoryStore.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import yaml
from workflows.public_speech.public_context import memory_sections, task_lines
from workflows.public_speech.speech_review import is_blocked, load_policy, review_speech

from workflows.dialogue_assistant.schemas import (
    AssistantMessage,
    AssistantReport,
    AssistantRequest,
    ToolReceipt,
)

WORKFLOW_DIR = Path(__file__).resolve().parent


def load_prompts(path: Path | None = None) -> dict:
    return yaml.safe_load((path or WORKFLOW_DIR / "prompts.yaml").read_text(encoding="utf-8"))


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item.strip()}" for item in items if str(item).strip()) or "(none)"


def render_memories(request: AssistantRequest, prompts: dict) -> str:
    """Render the planned memory lanes, or the plain memory list without a plan.

    Lines already shown as prior chat turns or as the current message are left out.
    """
    if request.public_context is None:
        return "## What comes back to you right now\n" + _bullets(request.memories)
    shown = [message.content for message in request.prior_dialogue if message.role in {"user", "assistant"}]
    blocks = memory_sections(request.public_context, prompts["memory_lanes"],
                             exclude_texts=[*shown, request.current_user_message])
    return "\n\n".join(blocks) or "## What comes back to you right now\n(nothing in particular)"


def build_messages(request: AssistantRequest, prompts: dict) -> list[AssistantMessage]:
    """Return the system message, the preserved prior transcript, then input.

    The user turn holds the derived context in plain sections (no JSON), the
    task results, and the current message. Tool calls and their results then
    follow as normal chat messages.
    """
    system = prompts["system_prompt"].format(
        companion_name=request.companion_name,
        user_name=request.user_name,
        abilities=request.abilities.strip() or "- Text conversation only.",
    ).strip()
    tasks = task_lines(request.public_context) if request.public_context is not None else []
    user = prompts["user_template"].format(
        companion_name=request.companion_name,
        user_name=request.user_name,
        character_card=request.character_card.strip() or "(none)",
        time=request.time.strip() or "(unknown)",
        memories=render_memories(request, prompts),
        feelings=request.feelings.strip() or "calm, nothing strong",
        workspace=request.workspace.strip() or "(nothing in particular)",
        intent=request.selected_action.strip() or "answer in your own voice",
        tasks="\n".join(tasks) or "- No task result yet.",
        current_user_message=request.current_user_message,
    ).strip()
    messages = [AssistantMessage(role="system", content=system)]
    messages.extend(request.prior_dialogue)
    messages.append(AssistantMessage(role="user", content=user))
    return messages


def tool_schemas(request: AssistantRequest) -> list[dict]:
    """Return the OpenAI function schemas for the request's declared tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters or {"type": "object", "properties": {}},
            },
        }
        for tool in request.tools
    ]


def wire_message(message: AssistantMessage) -> dict:
    """Return one OpenAI-compatible message for a request body.

    Dropping empty optional fields keeps strict providers happy. Alibaba rejects
    an empty ``tool_calls`` array with HTTP 400, but the local server tolerated
    it. Always keep ``role`` and ``content`` because a tool-call turn may carry
    empty text.
    """
    data = message.model_dump(mode="json")
    if not data.get("tool_calls"):
        data.pop("tool_calls", None)
    for field in ("tool_call_id", "name"):
        if not data.get(field):
            data.pop(field, None)
    return data


def wire_messages(messages: list[AssistantMessage]) -> list[dict]:
    """Return the whole transcript in the request wire form."""
    return [wire_message(message) for message in messages]


def parse_completion(raw: Any) -> dict:
    if isinstance(raw, str):
        raw = json.loads(raw)
    try:
        message = raw["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        raise ValueError("The assistant completion has no choices[0].message.") from None
    if not isinstance(message, dict):
        raise ValueError("The assistant message is not a mapping.")
    return message


# Management claims and the tool receipt that must support each one.
_MANAGEMENT_CLAIMS = {
    "started the task": "managed_start",
    "started a task": "managed_start",
    "started the run": "managed_start",
    "cancelled the run": "managed_terminate",
    "terminated the run": "managed_terminate",
    "stopped the task": "managed_terminate",
    "updated the instructions": "managed_update",
    "changed the instructions": "managed_update",
    "answered the question": "managed_answer",
}


def unsupported_management_claims(final_text: str, receipts: list[ToolReceipt]) -> list[str]:
    """Return management claims with no executed tool receipt behind them."""
    lowered = final_text.casefold()
    executed = {receipt.name for receipt in receipts if receipt.executed}
    unsupported: list[str] = []
    for phrase, tool in _MANAGEMENT_CLAIMS.items():
        if phrase in lowered and tool not in executed:
            unsupported.append(phrase)
    return unsupported


def _tool_call(call: dict) -> tuple[str, str, dict]:
    function = call.get("function") or {}
    name = str(function.get("name", ""))
    call_id = str(call.get("id", ""))
    raw_arguments = function.get("arguments") or {}
    if isinstance(raw_arguments, str):
        try:
            raw_arguments = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError:
            raw_arguments = {"_raw": raw_arguments}
    if not isinstance(raw_arguments, dict):
        raw_arguments = {"value": raw_arguments}
    return name, call_id, raw_arguments


async def _call_mcp(servers: dict, name: str, arguments: dict) -> Any:
    from fastmcp import Client

    for descriptor in servers.values():
        url = descriptor.get("url") if isinstance(descriptor, dict) else None
        if not url:
            continue
        async with Client(url) as client:
            tools = await client.list_tools()
            if not any(tool.name == name for tool in tools):
                continue
            result = await client.call_tool(name, arguments)
            if result.is_error:
                raise RuntimeError(f"Tool {name!r} raised: {result}")
            return result.data if result.data is not None else result.structured_content
    raise KeyError(name)


async def execute_tool_calls(
    context,
    message: dict,
    request: AssistantRequest,
    servers: dict,
) -> tuple[list[ToolReceipt], list[AssistantMessage]]:
    """Execute granted calls through MCP, or directly when no MCP is bound."""
    calls = message.get("tool_calls") or []
    receipts: list[ToolReceipt] = []
    tool_messages: list[AssistantMessage] = []
    grant = set(request.tool_grant)
    direct = {
        name: fn for name, fn in (context.runtime.tools or {}).items()
        if name != "assistant_complete"
    }
    for call in calls:
        name, call_id, arguments = _tool_call(call)
        receipt = ToolReceipt(
            tool_call_id=call_id, name=name, arguments=arguments, executed=False,
        )
        try:
            if name not in grant:
                raise PermissionError(f"Tool {name!r} is not granted for this run.")
            if name == "minecraft_observe" and name in direct:
                result = await asyncio.to_thread(direct[name], **arguments)
                receipt.transport = "python"
            elif servers:
                result = await _call_mcp(servers, name, arguments)
                receipt.transport = "mcp"
            elif name in direct:
                result = direct[name](**arguments)
                receipt.transport = "python"
            else:
                raise KeyError(name)
            receipt.executed = True
            receipt.result = result
            content = json.dumps(result, ensure_ascii=False, default=str)
        except BaseException as error:  # a denied or failed tool is data, not a crash
            receipt.error = f"{type(error).__name__}: {error}"
            content = json.dumps({"error": receipt.error})
        receipts.append(receipt)
        tool_messages.append(AssistantMessage(
            role="tool", content=content, tool_call_id=call_id, name=name,
        ))
    return receipts, tool_messages


def validate_public_speech(text: str) -> None:
    """Reject captured raw tool protocol before claim review or publication."""
    if text.lstrip().startswith(("<tool_call>", "<function=")):
        raise ValueError("The model returned tool protocol instead of public speech.")


def run(context) -> dict:
    environment = context.runtime.environment
    request = AssistantRequest.model_validate(
        yaml.safe_load(environment.read_text("inbox/request.yaml"))
    )
    prompts = load_prompts()
    messages = build_messages(request, prompts)
    servers = dict(context.runtime.mcp_servers or {})
    schemas = tool_schemas(request)
    completion = context.runtime.tools["assistant_complete"]

    receipts: list[ToolReceipt] = []
    final_text = ""
    if request.minecraft_read == "inventory":
        # A direct inventory question needs one fresh observation, then speech.
        call = {"tool_calls": [{"id": "inventory-observation", "type": "function", "function": {
            "name": "minecraft_observe", "arguments": json.dumps({"include_image": False})}}]}
        messages.append(AssistantMessage(role="assistant", tool_calls=call["tool_calls"]))
        observed, tool_messages = asyncio.run(execute_tool_calls(context, call, request, servers))
        receipts.extend(observed)
        messages.extend(tool_messages)
        schemas = []
    turns = request.max_tool_turns
    for turn in range(turns):
        payload = wire_messages(messages)
        # On the last allowed turn, ban tools so the model must answer in text.
        allow_tools = schemas if turn < turns - 1 else []
        raw = asyncio.run(completion(payload, tools=allow_tools, **prompts.get("generation", {})))
        message = parse_completion(raw)
        if message.get("tool_calls") and allow_tools:
            messages.append(AssistantMessage(
                role="assistant",
                content=str(message.get("content") or ""),
                tool_calls=[dict(call) for call in message["tool_calls"]],
            ))
            new_receipts, additions = asyncio.run(
                execute_tool_calls(context, message, request, servers)
            )
            messages.extend(additions)
            receipts.extend(new_receipts)
            continue
        final_text = str(message.get("content") or "").strip()
        if final_text:
            messages.append(AssistantMessage(role="assistant", content=final_text))
            break
    if not final_text:
        # The tool budget is done and the model returned no text. Force one
        # final, tool-free completion so the run always produces a result.
        payload = wire_messages(messages)
        raw = asyncio.run(completion(payload, tools=[], **prompts.get("generation", {})))
        message = parse_completion(raw)
        final_text = str(message.get("content") or "").strip()
        messages.append(AssistantMessage(role="assistant", content=final_text))

    validate_public_speech(final_text)
    accepted = bool(final_text)
    review = None
    if request.public_context is not None and final_text:
        # Tool replies get one hallucination review and no regeneration loop:
        # a second tool run could change the world. Only a blocking kind,
        # such as a claimed tool result that no receipt shows, rejects.
        policy = load_policy()
        review = asyncio.run(review_speech(context.runtime.tools["public_complete"], final_text,
                                           request.public_context, policy=policy,
                                           receipts=[item.model_dump(mode="json") for item in receipts]))
        accepted = accepted and not is_blocked(review, policy)
    report = AssistantReport(
        companion_name=request.companion_name,
        message_roles=[message.role for message in messages],
        messages=messages,
        final_text=final_text,
        tool_receipts=receipts,
        accepted=accepted,
        rejection_reason="" if accepted else ("; ".join(f"{item.kind}: {item.quote}" for item in review.violations)
                                             if review is not None else "The model returned an empty reply."),
        speech_review=review,
    )
    environment.write_text(
        "scratch/report.yaml",
        yaml.safe_dump(report.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
    )
    environment.move("scratch/report.yaml", "outbox/report.yaml")
    return {
        "status": "done",
        "summary": f"assistant reply with {len(receipts)} tool receipts",
    }


__all__ = [
    "build_messages",
    "execute_tool_calls",
    "load_prompts",
    "parse_completion",
    "render_memories",
    "run",
    "tool_schemas",
    "unsupported_management_claims",
    "wire_message",
    "wire_messages",
]
