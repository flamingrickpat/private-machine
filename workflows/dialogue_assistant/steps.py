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
from dataclasses import replace, dataclass
from collections.abc import Sequence
from typing import Literal, TypedDict
from pm_workflows.python_role import RoleContext
from pydantic import TypeAdapter, JsonValue
from pm.llm.completion import Completion
from pm.llm.session import ModelSession
from pm.llm.generation import GenerationSettings, ToolSchema, ToolFunctionSchema
from pm.llm.role_model import role_model
from pm.context_render.turns import Turn, ToolCall, Role
from pm.context_render.prompt import RendererCapture, RendererInput, RenderedPrompt

from workflows.dialogue_assistant.rendering import (AssistantPrompts, AssistantRenderInput, AssistantContinuationInput,
    render_assistant_prompt, render_assistant_continuation, assistant_transcript)
from pm.utils import yaml_util
import httpx
from workflows.public_speech.public_context import DialogueTurn, memory_sections, task_lines
from workflows.public_speech.speech_text import (
    SpeechOutputError, strip_classifier_report, strip_narrator_setup, validate_public_speech,
)
from workflows.public_speech.speech_review import (
    SpeechReview, block_reason, is_blocked, load_prompts as load_review_prompts, review_speech,
    select_spoken_text,
)

from workflows.dialogue_assistant.schemas import (
    AssistantMessage, AssistantToolCall, AssistantFunctionCall,
    AssistantReport,
    AssistantRequest,
    ToolReceipt,
    McpEndpoint as McpEndpoint,
)

WORKFLOW_DIR = Path(__file__).resolve().parent

class ToolExecutionError(RuntimeError):
    """A remote tool reports failure after the request reaches its transport."""


def load_prompts(path: Path | None = None) -> AssistantPrompts:
    return TypeAdapter(AssistantPrompts).validate_python(yaml_util.safe_load((path or WORKFLOW_DIR / "prompts.yaml").read_text(encoding="utf-8")))


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item.strip()}" for item in items if str(item).strip()) or "(none)"


def render_memories(request: AssistantRequest, prompts: AssistantPrompts) -> str:
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


def build_messages(request: AssistantRequest, prompts: AssistantPrompts) -> list[AssistantMessage]:
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
    tool_guidance = "Answer in ordinary speech from the supplied observations and task results."
    if request.tool_grant and request.max_tool_turns > 1 and not request.minecraft_read:
        tool_guidance = "Use a granted tool if the reply needs a fresh observation."
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
        tool_guidance=tool_guidance,
        current_user_message=request.current_user_message,
    ).strip()
    messages = [AssistantMessage(role=Role.System, content=system)]
    messages.extend(request.prior_dialogue)
    messages.append(AssistantMessage(role=Role.User, content=user))
    if request.current_user_message:
        messages.append(AssistantMessage(role=Role.User, content=request.current_user_message))
    return messages


def tool_schemas(request: AssistantRequest) -> tuple[ToolSchema, ...]:
    return tuple(ToolSchema(function=ToolFunctionSchema(name=tool.name, description=tool.description,
        parameters=tool.parameters)) for tool in request.tools)


def completion_turns(messages: Sequence[AssistantMessage]) -> list[Turn]:
    turns: list[Turn] = []
    for message in messages:
        calls = tuple(ToolCall(call.id, call.function.name, call.function.arguments) for call in message.tool_calls)
        turns.append(Turn(message.role, message.content, calls, message.tool_call_id, message.name, message.provider_state))
    return turns


def parse_completion(raw: Completion) -> AssistantMessage:
    turn = raw.turn
    calls = [AssistantToolCall(id=call.id, function=AssistantFunctionCall(name=call.name, arguments=call.arguments))
             for call in turn.tool_calls]
    return AssistantMessage(role=turn.role, content=turn.content.text, tool_calls=calls,
        tool_call_id=turn.tool_call_id, name=turn.name, provider_state=turn.provider_state)


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


@dataclass(frozen=True, slots=True)
class ToolInvocation:
    name: str
    call_id: str
    arguments_json: str


@dataclass(frozen=True, slots=True)
class ToolExecutionBatch:
    receipts: tuple[ToolReceipt, ...]
    messages: tuple[AssistantMessage, ...]


@dataclass(frozen=True, slots=True)
class McpToolResult:
    text: str
    succeeded: bool


def render_mcp_result(value: JsonValue, indentation: int = 0) -> str:
    padding = "  " * indentation
    if isinstance(value, dict):
        fields: list[str] = []
        for name, content in value.items():
            label = name.replace("_", " ")
            if isinstance(content, (dict, list)):
                fields.append(f"{padding}{label}:\n{render_mcp_result(content, indentation + 1)}")
            else:
                fields.append(f"{padding}{label}: {render_mcp_result(content)}")
        return "\n".join(fields)
    if isinstance(value, list):
        return "\n".join(f"{padding}- {render_mcp_result(item, indentation + 1).lstrip()}" for item in value)
    if value is None:
        return "(none)"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


async def _call_mcp(servers: Sequence[McpEndpoint], invocation: ToolInvocation) -> McpToolResult:
    from fastmcp import Client
    arguments = TypeAdapter[dict[str, JsonValue]](dict[str, JsonValue]).validate_json(invocation.arguments_json)
    for descriptor in servers:
        async with Client(descriptor.url) as client:
            tools = await client.list_tools()
            if not any(tool.name == invocation.name for tool in tools):
                continue
            result = await client.call_tool(invocation.name, arguments)
            if result.is_error:
                raise ToolExecutionError(f"Tool {invocation.name!r} raised: {result}")
            data = result.structured_content if result.structured_content is not None else result.data
            payload = TypeAdapter[JsonValue](JsonValue).validate_python(data)
            succeeded = not (isinstance(payload, dict) and payload.get("ok") is False)
            return McpToolResult(render_mcp_result(payload), succeeded)
    raise KeyError(invocation.name)


async def execute_tool_calls(context: RoleContext, message: AssistantMessage, request: AssistantRequest,
                             servers: Sequence[McpEndpoint]) -> ToolExecutionBatch:
    receipts: list[ToolReceipt] = []
    tool_messages: list[AssistantMessage] = []
    grant = set(request.tool_grant)
    for call in message.tool_calls:
        invocation = ToolInvocation(call.function.name, call.id, call.function.arguments)
        receipt = ToolReceipt(tool_call_id=invocation.call_id, name=invocation.name,
                              arguments_json=invocation.arguments_json, executed=False)
        try:
            if invocation.name not in grant:
                raise PermissionError(f"Tool {invocation.name!r} is not granted for this run.")
            if not servers:
                raise RuntimeError("Assistant tool execution requires its scoped MCP services.")
            result = await _call_mcp(servers, invocation)
            content = result.text
            receipt.succeeded = result.succeeded
            receipt.transport = "mcp"
            receipt.executed = True
            receipt.result_text = content
        except (PermissionError, OSError, httpx.HTTPError, ToolExecutionError) as error:
            receipt.error = f"{type(error).__name__}: {error}"
            content = "Tool failed: " + receipt.error
        receipts.append(receipt)
        tool_messages.append(AssistantMessage(role=Role.Tool, content=content,
            tool_call_id=invocation.call_id, name=invocation.name))
    return ToolExecutionBatch(tuple(receipts), tuple(tool_messages))


def final_public_speech(text: str, request: AssistantRequest) -> str:
    """Return one candidate utterance after the public-speech guards.

    Strip classifier headers and narrator framing before checking for tool
    protocol. An empty candidate raises ``SpeechOutputError``, so the host keeps the
    exchange pending instead of publishing an invented line.
    """
    speech = strip_classifier_report(strip_narrator_setup(text, request.partner_name))
    validate_public_speech(speech)
    if not speech:
        raise SpeechOutputError("The Assistant completion returned no public speech after its final text attempt.")
    if request.public_context and request.public_context.task_receipt:
        from workflows.minecraft_body.spatial import coordinates_in_answer, mentions_coordinates
        receipt = request.public_context.task_receipt
        if receipt.awaiting_stop_observation and mentions_coordinates(speech):
            raise SpeechOutputError("Acknowledge receipt of the stop request. Coordinates must wait for the final observation.")
        stopped = receipt.stop_observation
        if stopped is not None:
            point = stopped.reading.observation.camera.feet_position
            if not coordinates_in_answer(speech, point):
                raise SpeechOutputError(f"Answer from the confirmed post-stop position ({point.x}, {point.y}, {point.z}).")
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


def capture_assistant_continuation(session: ModelSession, input: AssistantContinuationInput) -> RenderedPrompt:
    settings = session.settings
    with RendererCapture(session.capture_settings("assistant-continuation"),
                         RendererInput("assistant-continuation", input, settings)) as capture:
        rendered = render_assistant_continuation(input, settings)
        capture.save_output(rendered)
    return rendered


def sample_public_speech(session: ModelSession, request: AssistantRequest, input: AssistantContinuationInput) -> str:
    """Resample text from the retained transcript without repeating a tool call."""
    while True:
        session.check_cancelled()
        rendered = capture_assistant_continuation(session, input)
        raw = asyncio.run(session.complete_tools_async(rendered.turns, tools=(), settings=input.generation))
        message = parse_completion(raw)
        try:
            return final_public_speech(message.content.strip(), request)
        except SpeechOutputError as error:
            print(f"[dialogue_assistant] Invalid spoken response: {error}. Resampling.", flush=True)
            input = replace(input, problem=str(error))


def review_and_revise(request: AssistantRequest, candidate: str, messages: list[AssistantMessage],
                      receipts: list[ToolReceipt], prompts: AssistantPrompts,
                      complete: ModelSession,
                      review_complete: ModelSession) -> tuple[str, SpeechReview, list[AssistantMessage]]:
    """Revise a blocked reply until one draft passes the evidence review.

    Two checks block a draft. The agentic review blocks a false task result.
    ``unsupported_management_claims`` blocks a management claim with no
    executed tool receipt behind it; a supplied task receipt already carries
    that support, so the claim check stands aside there. Every other review
    finding stays advisory.

    Each block appends the rejected words and the block reason to the
    conversation and asks for one new completion. That follow-up call carries
    no tools: the tool budget ran once in the caller, so a repair can change
    the words and can never change the world. Each attempt uses the retained
    transcript and the latest review diagnostic. The capture retains the rejected answer.

    Return the accepted utterance, its review, and every rejected draft with
    its correction request, oldest first, for the report transcript.
    """
    context = request.public_context
    if context is None:
        raise ValueError("Speech review requires the accepted public context.")
    review_prompts = load_review_prompts()
    conversation = review_conversation(request)
    receipt_payloads = tuple(item.evidence() for item in receipts)
    retained = tuple(messages)
    trail: list[AssistantMessage] = []
    text = candidate
    attempts = 0
    while True:
        attempts += 1
        review = asyncio.run(review_speech(review_complete, text, context, prompts=review_prompts,
                                           conversation=conversation, receipts=receipt_payloads, images=request.images))
        text = select_spoken_text(text, review)
        claims = [] if request.task_result_supplied else unsupported_management_claims(text, receipts)
        reason = (block_reason(review) if is_blocked(review)
                  else "unsupported management claim: " + ", ".join(claims) if claims else "")
        if not reason:
            if attempts > 1:
                print(f"[dialogue_assistant] Reply accepted after {attempts} attempts.", flush=True)
            return text, review, trail
        print(f"[dialogue_assistant] Reply rejected (attempt {attempts}): {reason}", flush=True)
        problem = prompts["rejected_reply_recovery"].format(reason=reason)
        trail.append(AssistantMessage(role=Role.Assistant, content=text))
        trail.append(AssistantMessage(role=Role.User, content=problem))
        generation = prompts.get("generation", GenerationSettings())
        repair_input = AssistantContinuationInput(
            messages=retained, tools=(), images=request.images, generation=generation,
            problem=problem, rejected_answer=text,
        )
        text = sample_public_speech(complete, request, repair_input)


class AssistantRoleResult(TypedDict):
    status: Literal["done"]
    summary: str


def run(context: RoleContext) -> AssistantRoleResult:
    """Kernel entry point for one Assistant reply."""
    from workflows.dialogue_assistant.runtime import assistant_runtime
    runtime = assistant_runtime(context.runtime)
    environment = runtime.environment
    request = AssistantRequest.model_validate(
        yaml_util.safe_load(environment.read_text("inbox/request.yaml"))
    )
    prompts = load_prompts()
    completion = role_model(context.runtime)
    generation = prompts.get("generation", GenerationSettings())
    settings = completion.render_settings(max_tokens=generation.output_allowance(completion.model.connection.completion_token_limit))
    request.current_user_message = completion.summarize_long_source(request.current_user_message,
        "current partner message", settings.input_ceiling // 3)
    schemas = tool_schemas(request)
    render_input = AssistantRenderInput(request, prompts, schemas)
    with RendererCapture(completion.capture_settings("assistant"), RendererInput("assistant", render_input, settings)) as capture:
        initial = render_assistant_prompt(render_input, settings)
        capture.save_output(initial)
    messages = assistant_transcript(initial)
    servers = runtime.mcp_endpoints

    receipts: list[ToolReceipt] = []
    # Retain completed tool effects before another completion can fail.
    environment.write_text("scratch/tool-receipts.json", "[]")
    final_text = ""
    if request.minecraft_read in {"inventory", "state"}:
        # A body-state question needs one fresh observation, then speech.
        call = AssistantMessage(role=Role.Assistant, tool_calls=[AssistantToolCall(id="body-observation",
            function=AssistantFunctionCall(name="minecraft_observe", arguments='{"include_image":false}'))])
        messages.append(call)
        observed = asyncio.run(execute_tool_calls(context, call, request, servers))
        receipts.extend(observed.receipts)
        environment.write_text("scratch/tool-receipts.json", json.dumps([item.model_dump(mode="json") for item in receipts]))
        messages.extend(observed.messages)
        schemas = ()
    turns = request.max_tool_turns
    for turn in range(turns):
        # On the last allowed turn, ban tools so the model must answer in text.
        allow_tools = schemas if turn < turns - 1 else ()
        payload = capture_assistant_continuation(completion, AssistantContinuationInput(tuple(messages), allow_tools, request.images, prompts.get("generation", GenerationSettings()))).turns
        raw = asyncio.run(completion.complete_tools_async(payload, tools=allow_tools, settings=prompts.get("generation", GenerationSettings())))
        message = parse_completion(raw)
        if message.tool_calls and allow_tools:
            messages.append(message)
            additions = asyncio.run(execute_tool_calls(context, message, request, servers))
            messages.extend(additions.messages)
            receipts.extend(additions.receipts)
            environment.write_text("scratch/tool-receipts.json", json.dumps([item.model_dump(mode="json") for item in receipts]))
            continue
        final_text = message.content.strip()
        if final_text:
            break
    try:
        final_text = final_public_speech(final_text, request)
    except SpeechOutputError as error:
        repair = AssistantContinuationInput(tuple(messages), (), request.images, generation, problem=str(error))
        final_text = sample_public_speech(completion, request, repair)
    review = None
    if request.public_context is not None:
        # A blocked draft returns for correction until one draft passes, so
        # only an accepted reply ever reaches the report and the host guard.
        final_text, review, trail = review_and_revise(
            request, final_text, messages, receipts, prompts, completion,
            completion)
        messages.extend(trail)
    messages.append(AssistantMessage(role=Role.Assistant, content=final_text))
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
]
