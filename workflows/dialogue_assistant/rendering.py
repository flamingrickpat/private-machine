from __future__ import annotations

from workflows.public_speech.public_context import rank_public_evidence, rank_memory_texts

from dataclasses import dataclass
from typing import TypedDict, NotRequired

from pm.context_render.budget import TokenEstimator
from pm.context_render.correction import render_regeneration_feedback
from pm.llm.generation import GenerationSettings, ToolSchema
from pm.llm.protocol_context import reserve_protocol
from pm.context_render.images import ImageCatalog, resolve_media
from pm.context_render.prompt import RenderSettings, RenderedPrompt, EvidenceDecision, require_rendered_prompt
from pm.context_render.turns import Text, Media, MediaKind, Role, Turn
from pm.records.media import RecordImage
from workflows.dialogue_assistant.schemas import AssistantRequest, AssistantMessage, AssistantToolCall, AssistantFunctionCall
from workflows.public_speech.public_context import MemoryLaneTemplate, EvidenceItem


class AssistantPrompts(TypedDict):
    system_prompt: str
    user_template: str
    shards_title: str
    memory_lanes: list[MemoryLaneTemplate]
    rejected_reply_recovery: str
    time_now: str
    time_since: str
    time_first: str
    generation: NotRequired[GenerationSettings]


@dataclass(frozen=True, slots=True)
class AssistantRenderInput:
    request: AssistantRequest
    prompts: AssistantPrompts
    tools: tuple[ToolSchema, ...] = ()


@dataclass(frozen=True, slots=True)
class AssistantContinuationInput:
    messages: tuple[AssistantMessage, ...]
    tools: tuple[ToolSchema, ...]
    images: tuple[RecordImage, ...] = ()
    generation: GenerationSettings = GenerationSettings()
    problem: str = ""
    rejected_answer: str = ""


def render_assistant_prompt(input: AssistantRenderInput, settings: RenderSettings) -> RenderedPrompt:
    request, prompts = input.request, input.prompts
    settings = assistant_protocol_settings(settings, input.tools, prompts.get("generation", GenerationSettings()))
    from workflows.dialogue_assistant.steps import build_messages, completion_turns
    estimator = TokenEstimator(estimate_factor=settings.estimate_factor)
    context = request.public_context
    images = ImageCatalog(request.images)
    selected_context = context.model_copy(deep=True) if context is not None else None
    if selected_context is not None:
        selected_context.evidence = []
        selected_context.memories = []
    required_dialogue = [message for message in request.prior_dialogue
                         if message.tool_calls or message.role == "tool"]
    for message in request.prior_dialogue[-2:]:
        if message not in required_dialogue:
            required_dialogue.append(message)
    required_dialogue = [message for message in request.prior_dialogue if message in required_dialogue]
    selected = request.model_copy(deep=True)
    selected.prior_dialogue = required_dialogue
    selected.public_context = selected_context
    selected.memories = []
    turns = resolve_media(tuple(completion_turns(build_messages(selected, prompts))), images)
    require_rendered_prompt("assistant dialogue", turns, settings)
    evidence: list[EvidenceItem] = []
    decisions: list[EvidenceDecision] = []
    seen: set[str] = set()
    if context is not None:
        for item in rank_public_evidence(context.evidence):
            text = " ".join(item.text.casefold().split())
            if text in seen:
                decisions.append(EvidenceDecision(item.knoxel_id, item.score, False, "duplicate"))
                continue
            seen.add(text)
            trial_context = context.model_copy(deep=True)
            trial_context.evidence = [*evidence, item]
            trial_context.memories = []
            trial = selected.model_copy(deep=True)
            trial.public_context = trial_context
            trial_turns = resolve_media(tuple(completion_turns(build_messages(trial, prompts))), images)
            included = estimator.estimate(trial_turns) <= settings.input_ceiling
            decisions.append(EvidenceDecision(item.knoxel_id, item.score, included, "relevance" if included else "budget"))
            if included:
                evidence.append(item)
                selected, turns = trial, trial_turns
    from pm.memory.retrieval import distinctive_terms
    focus_terms = set(distinctive_terms(request.current_user_message))
    represented_terms: set[str] = set()
    for message in required_dialogue:
        represented_terms.update(focus_terms.intersection(distinctive_terms(message.content)))
    for message in reversed(request.prior_dialogue):
        if message in required_dialogue:
            continue
        shared_terms = focus_terms.intersection(distinctive_terms(message.content))
        new_terms = shared_terms - represented_terms
        if not new_terms:
            continue
        trial = selected.model_copy(deep=True)
        trial.prior_dialogue = [item for item in request.prior_dialogue if item == message or item in selected.prior_dialogue]
        trial_turns = resolve_media(tuple(completion_turns(build_messages(trial, prompts))), images)
        if estimator.estimate(trial_turns) <= settings.input_ceiling:
            selected, turns = trial, trial_turns
            represented_terms.update(new_terms)
    groups: list[str] = []
    if context is None:
        for memory in rank_memory_texts(request.memories, request.current_user_message):
            trial = selected.model_copy(deep=True)
            trial.memories = [*selected.memories, memory.text]
            trial_turns = resolve_media(tuple(completion_turns(build_messages(trial, prompts))), images)
            if estimator.estimate(trial_turns) <= settings.input_ceiling:
                selected, turns = trial, trial_turns
                groups.append(f"plain-memory:{memory.position}")
    return require_rendered_prompt("assistant dialogue", turns, settings, tuple(decisions), selected_groups=tuple(groups))


def assistant_protocol_settings(settings: RenderSettings, tools: tuple[ToolSchema, ...], generation: GenerationSettings) -> RenderSettings:
    actual_generation = generation.with_tools(tools)
    output_tokens = actual_generation.output_allowance(settings.reserved_output_tokens)
    resolved = RenderSettings(context_window=settings.context_window, input_fraction=settings.input_fraction,
        reserved_output_tokens=output_tokens, protocol_tokens=settings.protocol_tokens,
        correction_tokens=settings.correction_tokens, estimate_factor=settings.estimate_factor)
    return reserve_protocol(resolved, actual_generation)


def render_assistant_continuation(input: AssistantContinuationInput, settings: RenderSettings) -> RenderedPrompt:
    from workflows.dialogue_assistant.steps import completion_turns
    messages = input.messages
    effective_settings = assistant_protocol_settings(settings, input.tools, input.generation)
    turns = resolve_media(tuple(completion_turns(list(messages))), ImageCatalog(input.images))
    if input.problem:
        feedback = render_regeneration_feedback("assistant dialogue", input.problem, effective_settings)
        turns = (*turns, Turn(Role.User, feedback.text))
        effective_settings = effective_settings.without_correction_reserve()
    return require_rendered_prompt("assistant tool continuation", turns, effective_settings)


def assistant_transcript(prompt: RenderedPrompt) -> list[AssistantMessage]:
    messages: list[AssistantMessage] = []
    for turn in prompt.turns:
        segments: list[str] = []
        for part in turn.content.parts:
            if isinstance(part, Text):
                segments.append(part.text)
            elif isinstance(part, Media) and part.kind is MediaKind.Image:
                segments.append(f"[image:{part.name}]")
            else:
                raise TypeError("Assistant transcript requires text or captured images.")
        calls = [AssistantToolCall(id=call.id, function=AssistantFunctionCall(name=call.name, arguments=call.arguments))
                 for call in turn.tool_calls]
        messages.append(AssistantMessage(role=turn.role, content="".join(segments), tool_calls=calls,
            tool_call_id=turn.tool_call_id, name=turn.name, provider_state=turn.provider_state))
    return messages
