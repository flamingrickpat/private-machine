"""Assistant dialogue role.

This is a separate message contract from Story dialogue. A Python role keeps
the normal user, assistant, and tool-call structure, executes the granted
tools, appends their real results, and returns one final assistant utterance.
A denied tool never runs. A tool receipt is the only support for a claim that
a management operation happened.

One final utterance then passes an evidence review. A blocked draft returns
for correction until one draft passes. See ``review_and_revise``.

``build_messages`` and the tool-call parsing are pure and tested without a
kernel, model, or MemoryStore.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from pm.llm.activation import ChatResult
from pm.llm.completion import AsyncTextCompletion, Completion
from pm.llm.prompt_encoding import OpenAIChatEncoder, normalize_turns
from pm.context_render.turns import Turn

from pm.utils import yaml_util
import httpx
from workflows.public_speech.public_context import DialogueTurn, memory_sections, task_lines
from workflows.public_speech.speech_text import SpeechOutputError, strip_narrator_setup
from workflows.public_speech.speech_review import (
    SpeechReview, block_reason, is_blocked, load_prompts as load_review_prompts, review_speech,
    select_spoken_text,
)

from workflows.dialogue_assistant.schemas import (
    AssistantMessage,
    AssistantReport,
    AssistantRequest,
    ToolReceipt,
)

WORKFLOW_DIR = Path(__file__).resolve().parent

#: Corrections asked for on one conversation before generation restarts cold.
REPAIR_FOLLOWUPS = 10


class ToolExecutionError(RuntimeError):
    """A remote tool reports failure after the request reaches its transport."""


def load_prompts(path: Path | None = None) -> dict:
    return yaml_util.safe_load((path or WORKFLOW_DIR / "prompts.yaml").read_text(encoding="utf-8"))


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item.strip()}" for item in items if str(item).strip()) or "(none)"


def render_memories(request: AssistantRequest, prompts: dict) -> str:
    """Render the shard story and the memory lanes, or the plain memory list without a context.

    A reply that main cognition chose shows the workspace shards of its
    cycle in the words of the cycle's story prefix. Lines already shown as
    prior chat turns or as the current message are left out of the lanes.
    """
    if request.public_context is None:
        return "## What comes back to you right now\n" + _bullets(request.memories)
    context = request.public_context
    shown = [message.content for message in request.prior_dialogue if message.role in {"user", "assistant"}]
    blocks = memory_sections(context, prompts["memory_lanes"],
                             exclude_texts=[*shown, request.current_user_message])
    if context.shard_story.strip():
        title = prompts["shards_title"].format(companion_name=request.companion_name,
                                               partner_name=request.partner_name)
        blocks.insert(0, title + "\n" + context.shard_story.strip())
    return "\n\n".join(blocks) or "## What comes back to you right now\n(nothing in particular)"


def build_messages(request: AssistantRequest, prompts: dict) -> list[AssistantMessage]:
    """Return the system message, the preserved prior transcript, then input.

    A context turn holds the derived sections and task receipts. The final
    user turn holds the current message verbatim. Tool calls and their results
    follow as normal chat messages. An older quoted request cannot replace
    the final user turn.
    """
    system = prompts["system_prompt"].format(
        companion_name=request.companion_name,
        partner_name=request.partner_name,
        abilities=request.abilities.strip() or "- Text conversation only.",
    ).strip()
    tasks = task_lines(request.public_context) if request.public_context is not None else []
    user = prompts["user_template"].format(
        companion_name=request.companion_name,
        partner_name=request.partner_name,
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
    if request.current_user_message:
        messages.append(AssistantMessage(role="user", content=request.current_user_message))
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

    Omit empty optional fields from the message contract. Always keep ``role``
    and ``content`` because a tool-call turn can carry empty text.
    """
    data = message.model_dump(mode="json")
    if not data.get("tool_calls"):
        data.pop("tool_calls", None)
    for field in ("tool_call_id", "name"):
        if not data.get(field):
            data.pop(field, None)
    for field in ("reasoning", "reasoning_content", "reasoning_details"):
        if data.get(field) is None:
            data.pop(field, None)
    return data


def wire_messages(messages: list[AssistantMessage]) -> list[dict]:
    """Return the whole transcript in the request wire form."""
    return [wire_message(message) for message in messages]


def completion_turns(messages: list[AssistantMessage]) -> list[Turn]:
    """Read the preserved legacy transcript into neutral completion turns."""
    return normalize_turns(wire_messages(messages))


def parse_completion(raw: Any) -> dict:
    if isinstance(raw, Completion):
        return OpenAIChatEncoder().message(raw.turn)
    if isinstance(raw, ChatResult):
        return raw.choices[0].message.model_dump(mode="json")
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
                raise ToolExecutionError(f"Tool {name!r} raised: {result}")
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
        except (PermissionError, OSError, httpx.HTTPError, ToolExecutionError) as error:
            # Denials and transport failures become receipts. Programming errors
            # and cancellation retain their traceback and reach the caller.
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
        raise SpeechOutputError("The model returned tool protocol instead of public speech.")


def final_public_speech(text: str, request: AssistantRequest) -> str:
    """Return one candidate utterance after the public-speech guards.

    Raw tool protocol never becomes speech. Narrator framing is removed. An
    empty candidate raises ``SpeechOutputError``, so the host keeps the
    exchange pending instead of publishing an invented line.
    """
    validate_public_speech(text)
    speech = strip_narrator_setup(text, request.partner_name)
    if not speech:
        raise SpeechOutputError("The Assistant completion returned no public speech after its final text attempt.")
    return speech


def review_conversation(request: AssistantRequest) -> list[DialogueTurn]:
    """Give the reviewer the same spoken history and current request as generation.

    Tool results travel separately as receipts. Earlier task instructions stay
    attributed to their own turn instead of becoming current instructions.
    """
    turns = [DialogueTurn(speaker=request.partner_name if message.role == "user" else request.companion_name,
                          text=message.content)
             for message in request.prior_dialogue if message.role in {"user", "assistant"} and message.content.strip()]
    if request.current_user_message.strip():
        turns.append(DialogueTurn(speaker=request.partner_name, text=request.current_user_message))
    return turns


def review_and_revise(request: AssistantRequest, candidate: str, messages: list[AssistantMessage],
                      receipts: list[ToolReceipt], prompts: dict,
                      complete: AsyncTextCompletion,
                      review_complete: AsyncTextCompletion) -> tuple[str, SpeechReview, list[AssistantMessage]]:
    """Revise a blocked reply until one draft passes the evidence review.

    Two checks block a draft. The agentic review blocks a false task result.
    ``unsupported_management_claims`` blocks a management claim with no
    executed tool receipt behind it; a supplied task receipt already carries
    that support, so the claim check stands aside there. Every other review
    finding stays advisory.

    Each block appends the rejected words and the block reason to the
    conversation and asks for one new completion. That follow-up call carries
    no tools: the tool budget ran once in the caller, so a repair can change
    the words and can never change the world. After ``REPAIR_FOLLOWUPS``
    corrections on one conversation, the next call reads the retained
    transcript alone. A long repair chain repeats its own wrong claims, so the
    chain is dropped rather than extended. There is no total limit; the loop
    stops only on an accepted draft, on cancellation, or on empty output.

    Return the accepted utterance, its review, and every rejected draft with
    its correction request, oldest first, for the report transcript.
    """
    context = request.public_context
    review_prompts = load_review_prompts()
    conversation = review_conversation(request)
    receipt_payloads = [item.model_dump(mode="json") for item in receipts]
    retained = list(messages)
    trail: list[AssistantMessage] = []
    repairs: list[AssistantMessage] = []
    text = candidate
    followups = 0
    attempts = 0
    while True:
        attempts += 1
        review = asyncio.run(review_speech(review_complete, text, context, prompts=review_prompts,
                                           conversation=conversation, receipts=receipt_payloads))
        text = select_spoken_text(text, review)
        claims = [] if context.task_receipt else unsupported_management_claims(text, receipts)
        reason = (block_reason(review) if is_blocked(review)
                  else "unsupported management claim: " + ", ".join(claims) if claims else "")
        if not reason:
            if attempts > 1:
                print(f"[dialogue_assistant] Reply accepted after {attempts} attempts.", flush=True)
            return text, review, trail
        print(f"[dialogue_assistant] Reply rejected (attempt {attempts}): {reason}", flush=True)
        correction = [AssistantMessage(role="assistant", content=text),
                      AssistantMessage(role="user",
                                       content=prompts["rejected_reply_recovery"].format(reason=reason))]
        trail = [*trail, *correction]
        if followups >= REPAIR_FOLLOWUPS:
            print("[dialogue_assistant] Ten follow-ups failed. Restarting from the retained transcript.",
                  flush=True)
            repairs = []
            followups = 0
        else:
            print(f"[dialogue_assistant] Requesting correction {followups + 1}/{REPAIR_FOLLOWUPS}.", flush=True)
            repairs = [*repairs, *correction]
            followups += 1
        message = parse_completion(asyncio.run(
            complete(completion_turns([*retained, *repairs]), tools=[], **prompts.get("generation", {}))))
        text = final_public_speech(str(message.get("content") or "").strip(), request)


def run(context) -> dict:
    environment = context.runtime.environment
    request = AssistantRequest.model_validate(
        yaml_util.safe_load(environment.read_text("inbox/request.yaml"))
    )
    prompts = load_prompts()
    messages = build_messages(request, prompts)
    servers = dict(context.runtime.mcp_servers or {})
    schemas = tool_schemas(request)
    completion = context.runtime.tools["assistant_complete"]

    receipts: list[ToolReceipt] = []
    # Retain completed tool effects before another completion can fail.
    environment.write_text("scratch/tool-receipts.json", "[]")
    final_text = ""
    if request.minecraft_read in {"inventory", "state"}:
        # A body-state question needs one fresh observation, then speech.
        call = {"tool_calls": [{"id": "body-observation", "type": "function", "function": {
            "name": "minecraft_observe", "arguments": json.dumps({"include_image": False})}}]}
        messages.append(AssistantMessage(role="assistant", tool_calls=call["tool_calls"]))
        observed, tool_messages = asyncio.run(execute_tool_calls(context, call, request, servers))
        receipts.extend(observed)
        environment.write_text("scratch/tool-receipts.json", json.dumps([item.model_dump(mode="json") for item in receipts]))
        messages.extend(tool_messages)
        schemas = []
    turns = request.max_tool_turns
    for turn in range(turns):
        payload = completion_turns(messages)
        # On the last allowed turn, ban tools so the model must answer in text.
        allow_tools = schemas if turn < turns - 1 else []
        raw = asyncio.run(completion(payload, tools=allow_tools, **prompts.get("generation", {})))
        message = parse_completion(raw)
        if message.get("tool_calls") and allow_tools:
            messages.append(AssistantMessage(
                role="assistant",
                content=str(message.get("content") or ""),
                tool_calls=[dict(call) for call in message["tool_calls"]],
                reasoning=message.get("reasoning"), reasoning_content=message.get("reasoning_content"),
                reasoning_details=message.get("reasoning_details"),
            ))
            new_receipts, additions = asyncio.run(
                execute_tool_calls(context, message, request, servers)
            )
            messages.extend(additions)
            receipts.extend(new_receipts)
            environment.write_text("scratch/tool-receipts.json", json.dumps([item.model_dump(mode="json") for item in receipts]))
            continue
        final_text = str(message.get("content") or "").strip()
        if final_text:
            break
    if not final_text:
        # The tool budget is done and the model returned no text. Force one
        # final, tool-free completion so the run always produces a result.
        payload = completion_turns(messages)
        raw = asyncio.run(completion(payload, tools=[], **prompts.get("generation", {})))
        message = parse_completion(raw)
        final_text = str(message.get("content") or "").strip()

    final_text = final_public_speech(final_text, request)
    review = None
    if request.public_context is not None:
        # A blocked draft returns for correction until one draft passes, so
        # only an accepted reply ever reaches the report and the host guard.
        final_text, review, trail = review_and_revise(
            request, final_text, messages, receipts, prompts, completion,
            context.runtime.tools["public_complete"])
        messages.extend(trail)
    messages.append(AssistantMessage(role="assistant", content=final_text))
    report = AssistantReport(
        companion_name=request.companion_name,
        message_roles=[message.role for message in messages],
        messages=messages,
        final_text=final_text,
        tool_receipts=receipts,
        accepted=True,
        speech_review=review,
        review_reason=review.reason if review is not None else "",
    )
    environment.write_text(
        "scratch/report.yaml",
        yaml_util.safe_dump(report.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
    )
    environment.move("scratch/report.yaml", "outbox/report.yaml")
    return {
        "status": "done",
        "summary": f"assistant reply with {len(receipts)} tool receipts",
    }


__all__ = [
    "build_messages",
    "execute_tool_calls",
    "final_public_speech",
    "load_prompts",
    "parse_completion",
    "render_memories",
    "review_and_revise",
    "review_conversation",
    "run",
    "tool_schemas",
    "unsupported_management_claims",
    "wire_message",
    "wire_messages",
]
