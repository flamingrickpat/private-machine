from __future__ import annotations

from workflows.public_speech.public_context import rank_public_evidence, rank_memory_texts

from dataclasses import dataclass
from typing import TypedDict, NotRequired

from pm.context_render.budget import TokenEstimator, evidence_excerpt
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
    groups: list[str] = []
    relax_pins = estimator.estimate(turns) > settings.input_ceiling
    if relax_pins:
        # Old call/result pairs leave the prompt together. The current tool
        # continuation has a separate renderer and keeps its protocol intact.
        required_dialogue = []
        selected.prior_dialogue = []
        groups.append("assistant dialogue pins relaxed")
        turns = resolve_media(tuple(completion_turns(build_messages(selected, prompts))), images)
        if estimator.estimate(turns) > settings.input_ceiling:
            selected.character_card = ""
            selected.workspace = ""
            selected.feelings = ""
            if selected_context is not None:
                selected_context.shard_story = ""
            turns = resolve_media(tuple(completion_turns(build_messages(selected, prompts))), images)
    retained = settings.input_ceiling
    while estimator.estimate(turns) > settings.input_ceiling and retained:
        retained //= 2
        selected.current_user_message = evidence_excerpt(request.current_user_message, retained)
        selected.selected_action = evidence_excerpt(request.selected_action, retained)
        selected.abilities = evidence_excerpt(request.abilities, retained)
        if selected_context is not None and context is not None and context.task_receipt is not None:
            selected_context.task_receipt = context.task_receipt.prompt_excerpt(retained)
        turns = resolve_media(tuple(completion_turns(build_messages(selected, prompts))), images)
    require_rendered_prompt("assistant dialogue", turns, settings)
    evidence: list[EvidenceItem] = []
    decisions: list[EvidenceDecision] = []
    seen: set[str] = set()
    if context is not None:
        for item in rank_public_evidence(context.evidence, prefer_recent=relax_pins):
            text = " ".join(item.text.casefold().split())
            if text in seen:
                decisions.append(EvidenceDecision(item.knoxel_id, item.score, False, "duplicate"))
                continue
            seen.add(text)
            assert selected.public_context is not None
            trial_context = selected.public_context.model_copy(deep=True)
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
    prior_dialogue = request.prior_dialogue
    if relax_pins:
        prior_dialogue = [message for message in prior_dialogue if not message.tool_calls and message.role != Role.Tool]
    for message in reversed(prior_dialogue):
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
    actual_generation = generation.for_request(tools=tools)
    output_tokens = actual_generation.output_allowance(settings.reserved_output_tokens)
    resolved = RenderSettings(context_window=settings.context_window, input_fraction=settings.input_fraction,
        reserved_output_tokens=output_tokens, protocol_tokens=settings.protocol_tokens,
        correction_tokens=settings.correction_tokens, estimate_factor=settings.estimate_factor)
    return reserve_protocol(resolved, actual_generation)


def render_assistant_continuation(input: AssistantContinuationInput, settings: RenderSettings) -> RenderedPrompt:
    from workflows.dialogue_assistant.steps import completion_turns
    effective_settings = assistant_protocol_settings(settings, input.tools, input.generation)
    images = ImageCatalog(input.images)
    turns = resolve_media(tuple(completion_turns(input.messages)), images)
    correction: tuple[Turn, ...] = ()
    if input.problem:
        feedback = render_regeneration_feedback("assistant dialogue", input.problem, effective_settings)
        correction = (Turn(Role.User, feedback.text),)
        effective_settings = effective_settings.without_correction_reserve()
    estimator = TokenEstimator(estimate_factor=settings.estimate_factor)
    if estimator.estimate((*turns, *correction)) <= effective_settings.input_ceiling:
        return require_rendered_prompt("assistant tool continuation", (*turns, *correction), effective_settings)
    # Fold completed call/result pairs into attributed observations. Signed provider
    # reasoning and tool arguments remain intact in the durable transcript.
    retained = effective_settings.input_ceiling
    while True:
        compact: list[Turn] = []
        for message in input.messages:
            if message.tool_calls:
                for call in message.tool_calls:
                    compact.append(Turn(Role.User, f"Previous tool request {call.id}: {call.function.name} "
                        + evidence_excerpt(call.function.arguments, retained)))
            elif message.role == Role.Tool:
                compact.append(Turn(Role.User, f"Recorded result of {message.name} ({message.tool_call_id}):\n"
                    + evidence_excerpt(message.content, retained)))
            else:
                content = message.content if message.role == Role.System else evidence_excerpt(message.content, retained)
                compact.append(Turn(message.role, content))
        turns = (*resolve_media(tuple(compact), images), *correction)
        if estimator.estimate(turns) <= effective_settings.input_ceiling or retained == 0:
            return require_rendered_prompt("assistant tool continuation", turns, effective_settings,
                selected_groups=("completed tool transcript with bounded excerpts",))
        retained //= 2


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
