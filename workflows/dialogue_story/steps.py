"""Story dialogue role.

Entry point: ``run(context)``. The role builds interleaved Story messages,
generates and reviews companion lines until one passes the agentic review.
It writes a full report to ``outbox/report.yaml``.
Runtime ownership is described in [the Story host overview][workflows.dialogue_story.host].

The message layout is a contract:

  1. system    - the story-writer instruction + architecture description
  2. user      - opening directive + character card + time
  3. assistant - dialogue block (both speakers, verbatim)
  4. user      - artifact block (memories, facts, narratives)
  ...          - alternating dialogue and artifact blocks in world-time order
  last user    - GWT state + feelings + action + continue directive

Dialogue lines (both speakers) always land in assistant turns. Derived
artifacts (memories, summaries, feelings) always land in user turns. The
voice is anchored only by real dialogue in assistant turns; summary wording
in user turns cannot leak into it. There is no assistant prefill.

The review (``speech_review.yaml``):

- Empty output permits one recovery with the full original context.
- Each valid companion line receives one review.
- A false task result receives correction feedback until a reply passes.
- After ten follow-ups, generation starts again from the original context.
- Each review returns a judgment, a reason, and advisory findings.
- Only ``false_task_result`` blocks the reply. A finding such as an invented
  ability or an invented event stays in the receipt for review.
- The reason of the accepted review is recorded in the report and in the
  published reply.

The workflow never stores a model-invented next user utterance. No tool call
is requested from the generation model. ``build_story_messages`` is pure and
is tested directly without a kernel, model, or MemoryStore.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Literal, TypedDict, Unpack, cast

from pydantic import TypeAdapter

from pm_workflows.python_role import RoleContext
from pm.context_render.turns import Role, Turn
from pm.llm.completion import AsyncTextCompletion
from pm.llm.http_client import EmptyCompletionError
from pm.host.durable_files import atomic_json

from pm.utils import yaml_util
from pm.memory.retrieval import utc_time
from workflows.public_speech.public_context import (
    ability_lines, memory_lines, memory_sections, plain_memory_text, task_lines, time_lines,
)
from workflows.public_speech.public_context import Presentation, PublicContext
from workflows.public_speech.story_rendering import (
    ArtifactLine, DialogueLine, dialogue_lines, render_interleaved_story,
)
from workflows.public_speech.speech_text import SpeechOutputError, strip_narrator_setup
from workflows.public_speech.speech_review import (
    block_reason, build_review_messages, finding_summary, is_blocked, load_prompts as load_review_prompts, review_speech,
    select_spoken_text, ReviewPrompts,
)

from workflows.dialogue_story.schemas import (
    DialogueTurn,
    OntologyVerdict,
    StoryAttempt,
    StoryReport,
    StoryRequest,
    StoryPrompts,
)

WORKFLOW_DIR = Path(__file__).resolve().parent



def load_prompts(path: Path | None = None) -> StoryPrompts:
    """Validate the existing YAML fields before any rendering or model call."""
    data = yaml_util.safe_load((path or WORKFLOW_DIR / "prompts.yaml").read_text(encoding="utf-8"))
    return TypeAdapter(StoryPrompts).validate_python(data)


def render_prior_dialogue(turns: list[DialogueTurn], prompts: dict | None = None) -> str:
    """Render prior turns verbatim as one story transcript, both speakers included.

    Speech renders as ``says: "..."`` and the companion's earlier private
    thoughts as ``thinks: *...*``, the established story-element forms. A day
    marker such as ``*(30 May 2026)*`` precedes the first turn of each day, so
    a window that spans several days does not read as one evening.
    """
    prompts = prompts or {}
    speech = prompts.get("speaker_line", '{speaker} says: "{text}"')
    thought = prompts.get("thought_line", "{speaker} thinks: *{text}*")
    marker = prompts.get("day_marker", "*({date})*")
    lines, day = [], None
    for turn in turns:
        if turn.timestamp:
            moment = utc_time(turn.timestamp)
            if moment.date() != day:
                day = moment.date()
                lines.append(marker.format(date=f"{moment.day} {moment.strftime('%B %Y')}"))
        lines.append((thought if turn.kind == "thought" else speech).format(speaker=turn.speaker, text=turn.text))
    return "\n".join(lines)


def _section(title: str, lines: list[str]) -> str:
    body = "\n".join(line for line in lines if line.strip())
    return f"{title}\n{body}" if body else ""


def _inner_lines(request: StoryRequest, prompts: StoryPrompts, names: dict[str, str]) -> list[str]:
    """Render the required private state and disclosure choice for generation."""
    lines: list[str] = []
    if request.feelings:
        lines.append(prompts["inner_feelings"].format(feelings=request.feelings, **names))
    if request.regulation in prompts["inner_regulation"]:
        lines.append(prompts["inner_regulation"][request.regulation].format(**names))
    if request.workspace.strip():
        lines.append(prompts["inner_mind"].format(workspace=request.workspace.strip(), **names))
    if request.selected_action.strip():
        lines.append(prompts["inner_intent"].format(intent=request.selected_action.strip(), **names))
    presentation = request.public_context.presentation if request.public_context else Presentation()
    lines.append(prompts["presentation"][presentation.mode].format(
        summary=presentation.shareable_summary or "nothing",
        purpose=presentation.purpose or "unstated", **names))
    return lines


def build_context_message(request: StoryRequest, prompts: StoryPrompts, *, avoid: str = "") -> str:
    """Render the user message: every derived or summarized piece of context.

    Verbatim dialogue never appears here. Memories that duplicate a verbatim
    transcript line, or the current message, are left out.

    A reply that main cognition chose shows the workspace shards of its
    cycle in the words of the cycle's story prefix (``shard_story`` of the
    public context). A standalone situation has no shards and shows its
    planned memory lanes instead.
    """
    names = {"companion_name": request.companion_name, "partner_name": request.partner_name}
    context = request.public_context
    shown = {turn.knoxel_id for turn in request.prior_dialogue if turn.knoxel_id is not None}
    if request.current_message_id is not None:
        shown.add(request.current_message_id)
    fallback_title = prompts["memory_lanes"][-1]["title"].format(**names)
    shard_story = context.shard_story.strip() if context is not None else ""
    if context is not None:
        abilities = ability_lines(context)
        from workflows.public_speech.ontology import term_lines
        ontology_terms = term_lines(context)
        transcript_texts = [turn.text for turn in request.prior_dialogue] + [request.current_user_message]
        memory_blocks = memory_sections(context, prompts["memory_lanes"], exclude_ids=shown,
                                        exclude_texts=transcript_texts)
        if not memory_blocks and not shard_story:
            # Plain memory strings are the fallback when the action carried no
            # evidence records, for example in a standalone situation.
            shown_texts = {" ".join(text.split()) for text in transcript_texts}
            memory_blocks = [_section(fallback_title, [f"- {plain_memory_text(item)}" for item in request.memories
                                                       if plain_memory_text(item)
                                                       and plain_memory_text(item) not in shown_texts])]
        tasks = task_lines(context) or [
            "- No current task receipt or direct tool observation is supplied. "
            "Current position, health and inventory are unknown without an observation."
        ]
    else:
        abilities = [f"- {item}" for item in request.supported_capabilities]
        abilities += [f"- Cannot: {item}" for item in request.unsupported_capabilities]
        abilities += [f"- Note: {item}" for item in request.capability_notes]
        ontology_terms = []
        memory_blocks = [_section(fallback_title, [f"- {item}" for item in request.memories])]
        tasks = []
    memory_blocks = [part for part in memory_blocks if part]
    if memory_blocks:
        memory_blocks.insert(0, prompts["memories_note"].strip())
    parts = [
        prompts["context_start"].format(**names).strip(),
        _section(prompts["character_title"], [request.character_card_story.strip()]),
        _section(prompts["architecture_title"].format(**names), [prompts["architecture_text"].format(**names).strip()]),
        _section(prompts["abilities_title"].format(**names), abilities),
        _section(prompts["ontology_title"].format(**names), ontology_terms),
        _section(prompts["time_title"], time_lines(request.now, request.prior_dialogue, prompts, names)),
        _section(prompts["shards_title"].format(**names),
                 [prompts["shards_note"].format(**names).strip(), shard_story]) if shard_story else "",
        *memory_blocks,
        _section(prompts["goals_title"].format(**names), [f"- {goal}" for goal in request.goals]),
        _section(prompts["tasks_title"], tasks),
        _section(prompts["inner_title"].format(**names), _inner_lines(request, prompts, names)),
        _section(prompts["regeneration_title"], [avoid]) if avoid else "",
        prompts["context_end"].format(**names).strip(),
    ]
    return "\n\n".join(part for part in parts if part)


def build_story_messages(request: StoryRequest, prompts: StoryPrompts, *, avoid: str = "") -> list[Turn]:
    """Return the interleaved Story turns: system, opening, alternating blocks, directive."""
    dialogue = _split_dialogue(request, prompts)
    artifacts = _artifact_lines(request, prompts) + _thought_lines(dialogue.private_thoughts, prompts)
    return render_interleaved_story(
        system_text=build_system_text(request, prompts),
        opening_text=build_opening_text(request, prompts),
        dialogue=dialogue.speech,
        artifacts=artifacts,
        directive_text=build_directive_text(request, prompts, avoid=avoid),
    )


def build_system_text(request: StoryRequest, prompts: StoryPrompts) -> str:
    """System turn: story-writer instruction + architecture + abilities + ontology."""
    names = {"companion_name": request.companion_name, "partner_name": request.partner_name}
    parts = [prompts["system_prompt"].strip()]
    parts.append(_section(prompts["architecture_title"].format(**names),
                          [prompts["architecture_text"].format(**names).strip()]))
    context = request.public_context
    if context is not None:
        abilities = ability_lines(context)
        from workflows.public_speech.ontology import term_lines
        ontology_terms = term_lines(context)
    else:
        abilities = [f"- {item}" for item in request.supported_capabilities]
        abilities += [f"- Cannot: {item}" for item in request.unsupported_capabilities]
        abilities += [f"- Note: {item}" for item in request.capability_notes]
        ontology_terms = []
    parts.append(_section(prompts["abilities_title"].format(**names), abilities))
    parts.append(_section(prompts["ontology_title"].format(**names), ontology_terms))
    return "\n\n".join(part for part in parts if part)


def build_opening_text(request: StoryRequest, prompts: StoryPrompts) -> str:
    """Opening user turn: story instruction + character card + time."""
    names = {"companion_name": request.companion_name, "partner_name": request.partner_name}
    parts = [
        prompts["story_instruction"].format(**names).strip(),
        _section(prompts["character_title"], [request.character_card_story.strip()]),
        _section(prompts["time_title"], time_lines(request.now, request.prior_dialogue, prompts, names)),
    ]
    return "\n\n".join(part for part in parts if part)


def build_directive_text(request: StoryRequest, prompts: StoryPrompts, *, avoid: str = "") -> str:
    """Final user turn: shards + goals + tasks + inner state + continue directive."""
    names = {"companion_name": request.companion_name, "partner_name": request.partner_name}
    context = request.public_context
    shard_story = context.shard_story.strip() if context is not None else ""
    parts = []
    if shard_story:
        parts.append(_section(prompts["shards_title"].format(**names),
                              [prompts["shards_note"].format(**names).strip(), shard_story]))
    if context is not None:
        tasks = task_lines(context)
    else:
        tasks = []
    parts.append(_section(prompts["goals_title"].format(**names), [f"- {goal}" for goal in request.goals]))
    parts.append(_section(prompts["tasks_title"], tasks))
    parts.append(_section(prompts["inner_title"].format(**names), _inner_lines(request, prompts, names)))
    if avoid:
        parts.append(_section(prompts["regeneration_title"], [avoid]))
    parts.append(prompts["continue_directive"].format(**names).strip())
    return "\n\n".join(part for part in parts if part)


@dataclass(frozen=True, slots=True)
class StoryDialogueParts:
    """Spoken transcript lines and the private turns rendered as artifacts."""

    speech: list[DialogueLine]
    private_thoughts: list[DialogueTurn]


def _split_dialogue(request: StoryRequest, prompts: StoryPrompts) -> StoryDialogueParts:
    """Separate spoken lines and thoughts from prior dialogue and current speech.

    Only real spoken utterances become assistant dialogue lines. The companion's
    own earlier thoughts stay out of the spoken transcript; the caller renders
    them as user artifacts. The current user message is always speech.
    """
    turns = list(request.prior_dialogue)
    if request.current_user_message:
        turns.append(DialogueTurn(speaker=request.partner_name, text=request.current_user_message,
                                  knoxel_id=request.current_message_id))
    speech_turns = [t for t in turns if t.kind != "thought"]
    thought_turns = [t for t in turns if t.kind == "thought"]
    return StoryDialogueParts(dialogue_lines(speech_turns, prompts), thought_turns)


def _thought_lines(thought_turns: list[DialogueTurn], prompts: StoryPrompts) -> list[ArtifactLine]:
    """Render the companion's earlier thoughts as user artifact lines."""
    template = prompts.get("thought_line", "{speaker} thinks: *{text}*")
    lines: list[ArtifactLine] = []
    for turn in thought_turns:
        ts = None
        if turn.timestamp:
            try:
                ts = utc_time(turn.timestamp)
            except (ValueError, TypeError):
                pass
        lines.append(ArtifactLine(timestamp=ts, record_id=turn.knoxel_id or 0, lane="thought",
                                  text=f"- {template.format(speaker=turn.speaker, text=turn.text)}"))
    return lines


def _artifact_lines(request: StoryRequest, prompts: StoryPrompts) -> list[ArtifactLine]:
    """Evidence records as ArtifactLine items, excluding verbatim transcript lines."""
    context = request.public_context
    if context is None:
        return []
    shown = {turn.knoxel_id for turn in request.prior_dialogue if turn.knoxel_id is not None}
    if request.current_message_id is not None:
        shown.add(request.current_message_id)
    transcript_texts = [turn.text for turn in request.prior_dialogue] + [request.current_user_message]
    shown_texts = {" ".join(text.split()) for text in transcript_texts}
    lines: list[ArtifactLine] = []
    for item in context.evidence:
        if item.knoxel_id in shown:
            continue
        body = plain_memory_text(item.text)
        if not body or body in shown_texts:
            continue
        ts = None
        if item.timestamp:
            try:
                ts = utc_time(item.timestamp)
            except (ValueError, TypeError):
                pass
        lines.append(ArtifactLine(timestamp=ts, record_id=item.knoxel_id,
                                  lane=item.lane or "focus", text=f"- {body}"))
    return lines


def _line_start(text: str, marker: str) -> int | None:
    """Return the index where marker opens a line, or None. An inline match is not a turn boundary."""
    index = text.find(marker)
    while index != -1:
        if index == 0 or text[index - 1] == "\n":
            return index
        index = text.find(marker, index + 1)
    return None


def clean_reply(raw: str, companion_name: str, partner_name: str, *, prefill: str = "") -> str:
    """Keep only the companion's continuation.

    ``prefill`` is the exact transcript sent in the final assistant turn.
    Remove its full echo before speaker and day-marker checks.
    Then cut the text at the first marker of any other
    speaker or the start of a non-public state line. This prevents a
    model-invented next user turn from becoming stored dialogue.

    A marker only ends the reply when it opens a line, so no punctuation and no
    inline mention of another speaker can truncate a spoken utterance. Every
    companion bubble before that boundary is kept, in order, one per line.
    """
    reply = (raw or "").lstrip()
    if prefill and reply.startswith(prefill):
        reply = reply[len(prefill):]
    reply = reply.strip()
    leading_marker = False
    for prefix in (f"{companion_name} says:", f"{companion_name}:"):
        if reply.lower().startswith(prefix.lower()):
            reply = reply[len(prefix):].lstrip()
            if reply.startswith('"'):
                reply = reply[1:]
            leading_marker = True
            break

    if reply.startswith((f"{companion_name} thinks:", "StateUpdate:", "Narrative:", "Fact:",
                         "*(", "<tool_call>", "<function=")):
        return ""

    # Cut at the first other-speaker or non-public marker before trimming any
    # preamble. A model-invented user turn is never kept, even when the model
    # later reopens the companion line. Each marker counts only at a line start:
    # the companion quoting someone inside its own sentence is speech, and
    # cutting there would deliver half an utterance.
    other_markers = [
        f"{partner_name} says:", f"{partner_name}:",
        "StateUpdate:", "Narrative:", "Fact:",
        # The transcript shows earlier private thoughts. A continuation that
        # starts a new thought line is not speech.
        f"{companion_name} thinks:",
        # A new day marker is narration, not speech.
        "*(",
    ]
    cut = len(reply)
    for marker in other_markers:
        index = _line_start(reply, marker)
        if index is not None:
            cut = min(cut, index)
    reply = reply[:cut].strip()

    # Keep every own bubble up to that boundary, in order. Keeping only the text
    # after the last own marker delivered the fourth line of a multi-line
    # completion and dropped the first one, which was the line that answered
    # the user (run bz_p_e2e_live_7530, call 502).
    own_bubbles = re.split(rf"^{re.escape(companion_name)}(?: says)?[ \t]*:", reply, flags=re.MULTILINE)
    lead, segments = own_bubbles[0], own_bubbles[1:]
    bubbles = []
    # The marker strip above already removed a leading own marker, so that lead
    # is spoken text. A lead before an unstripped marker is model preamble.
    if lead.strip() and (prefill or leading_marker or not segments):
        bubbles.append(lead)
    bubbles.extend(segments)
    cleaned = []
    for bubble in bubbles:
        bubble = bubble.strip()
        if bubble.startswith('"') and bubble.endswith('"') and len(bubble) > 1:
            bubble = bubble[1:-1].strip()
        elif bubble.endswith('"') and bubble.count('"') % 2 == 1:
            bubble = bubble[:-1].rstrip()
        elif bubble.startswith('"') and bubble.count('"') % 2 == 1:
            bubble = bubble[1:].lstrip()
        if bubble:
            cleaned.append(bubble)
    reply = "\n".join(cleaned)
    return strip_narrator_setup(reply, partner_name)


def review_context(request: StoryRequest) -> PublicContext:
    """Build a minimal public context when the caller supplied none.

    The standalone path has abilities and memories but no task receipt or
    durable ontology. The review still returns one judgment and one reason.
    """
    abilities = [*request.supported_capabilities, *[f"Cannot: {item}" for item in request.unsupported_capabilities],
                 *[f"Note: {item}" for item in request.capability_notes]]
    return PublicContext(
        companion_name=request.companion_name, user_name=request.partner_name,
        partner_name=request.partner_name, character=request.character_card_story, feelings=request.feelings,
        capabilities={"configured": abilities, "unavailable": [], "notes": [], "human_body": False},
        memories=list(request.memories),
    )


def review_conversation(request: StoryRequest) -> list[DialogueTurn]:
    """Return the recent speech plus the current user line for the review."""
    recent = [turn for turn in request.prior_dialogue if turn.kind == "speech"][-6:]
    return [*recent, *([DialogueTurn(speaker=request.partner_name, text=request.current_user_message)]
                      if request.current_user_message else [])]


class StoryReviewDependencies(TypedDict, total=False):
    """Explicit review inputs supplied by standalone generation and runtime capture."""

    context: PublicContext
    review_prompts: ReviewPrompts
    conversation: list[DialogueTurn]
    on_attempt: Callable[[int, StoryAttempt], None]


def generate_reviewed(request: StoryRequest, prompts: StoryPrompts, generate: AsyncTextCompletion,
                      review_complete: AsyncTextCompletion,
                      **review_deps: Unpack[StoryReviewDependencies]) -> tuple[list[StoryAttempt], StoryAttempt]:
    """Revise rejected speech until the review accepts a draft.

    Each rejection adds the draft and review reason to the generation context.
    After ten follow-ups, discard repair turns and start from the original context.
    Retries print their progress to stdout. No tool actions or memory commits occur.
    Each draft permits one empty-output recovery. Exhaustion raises SpeechOutputError.
    Transport, cancellation, recording, and programming errors propagate.
    ``on_attempt`` records each attempt before review and after its result.
    Runtime prompts select optional evidence against their full rendered layout.
    Generation includes required repair turns. Review includes the complete
    candidate. Their separate selection receipts remain on ``StoryAttempt``.
    """
    context = review_deps.get("context") or request.public_context or review_context(request)
    review_prompts = review_deps.get("review_prompts") or load_review_prompts()
    conversation = review_deps.get("conversation") or review_conversation(request)
    original_messages = build_story_messages(request, prompts)
    messages = original_messages
    repair_turns: list[Turn] = []
    prefill = ""  # The interleaved layout has no assistant prefill.
    on_attempt = review_deps.get("on_attempt")
    attempts: list[StoryAttempt] = []
    followups = 0
    while True:
        for index in range(2):
            selection = None
            generation_context = context
            generation_request = request
            if request.input_budget_tokens is not None:
                from workflows.dialogue_story.context import select_story_prompt
                budgeted = select_story_prompt(request, prompts, repair_turns)
                messages = list(budgeted.selection.turns)
                selection = budgeted.selection.receipt()
                generation_request = budgeted.request
                generation_context = budgeted.request.public_context or context
            else:
                messages = [*original_messages, *repair_turns]
            error = ""
            try:
                raw = asyncio.run(generate(messages, **prompts.get("generation", {})))
            except EmptyCompletionError as failure:
                raw, error = "", str(failure)
            candidate = clean_reply(raw, request.companion_name, request.partner_name, prefill=prefill)
            attempt = StoryAttempt(raw_completion=raw, candidate=candidate,
                message_roles=[message.role.value for message in messages],
                context_selection=selection,
                generation_error=error or ("" if candidate else "The completion contains no new public speech."))
            attempts.append(attempt)
            if on_attempt is not None:
                on_attempt(len(attempts), attempt)
            if candidate:
                break
            if index == 0:
                # Keep the full context and request only new speech after empty output.
                repair_turns.append(Turn(Role.User, prompts["empty_completion_recovery"].format(
                    companion_name=request.companion_name, partner_name=request.partner_name)))
        else:
            raise SpeechOutputError("The Story completion returned no new speech after two generation attempts.")
        review_messages = None
        if request.input_budget_tokens is not None:
            from workflows.dialogue_story.context import select_story_prompt

            def render_review(selected: StoryRequest) -> list[Turn]:
                return build_review_messages(candidate, selected.public_context or context,
                                             review_prompts, conversation=conversation)

            reviewed = select_story_prompt(generation_request, prompts, render=render_review)
            generation_context = reviewed.request.public_context or context
            attempt.review_context_selection = reviewed.selection.receipt()
            review_messages = reviewed.selection.turns
            if on_attempt is not None:
                on_attempt(len(attempts), attempt)
        review = asyncio.run(review_speech(review_complete, candidate, generation_context, conversation=conversation,
                                           prompts=review_prompts, prepared_messages=review_messages))
        candidate = select_spoken_text(candidate, review)
        attempt.candidate = candidate
        attempt.review = review
        attempt.blocked = is_blocked(review)
        if on_attempt is not None:
            on_attempt(len(attempts), attempt)
        if not attempt.blocked:
            if len(attempts) > 1:
                print(f"[dialogue_story] Reply accepted after {len(attempts)} attempts.", flush=True)
            return attempts, attempt
        reason = block_reason(review)
        print(f"[dialogue_story] Reply rejected (attempt {len(attempts)}): {reason}", flush=True)
        if followups >= 10:
            print("[dialogue_story] Ten follow-ups failed. Restarting from the original context.", flush=True)
            repair_turns = []
            followups = 0
        else:
            print(f"[dialogue_story] Requesting correction {followups + 1}/10.", flush=True)
            repair_turns.extend([Turn(Role.Assistant, raw),
                                 Turn(Role.User, prompts["rejected_reply_recovery"].format(reason=reason))])
            followups += 1


class StoryRoleResult(TypedDict):
    """The kernel role result; the reviewed speech stays in its separate report."""

    status: Literal["done"]
    summary: str


def run(context: RoleContext) -> StoryRoleResult:
    """Kernel entry point. Runs synchronously on the calling thread."""
    runtime = context.runtime
    assert runtime is not None, "The Story role requires its workflow runtime."
    environment = runtime.environment
    request = StoryRequest.model_validate(
        yaml_util.safe_load(environment.read_text("inbox/request.yaml"))
    )
    prompts = load_prompts()
    atomic_json(context.kernel_data / "story-request.json", request.model_dump(mode="json"))
    tools = runtime.tools
    # The foreign kernel registry erases callable signatures. The host tool
    # factory supplies these concrete asynchronous completion contracts.
    generate = cast(AsyncTextCompletion, tools["story_complete"])
    review_complete = cast(AsyncTextCompletion, tools.get("public_complete") or tools["story_complete"])

    def record_attempt(number: int, attempt: StoryAttempt) -> None:
        environment.write_text(f"scratch/story-attempt-{number:02}.json", attempt.model_dump_json(indent=2))
        # The working target is virtual. Retain the selection and draft in the
        # kernel's physical evidence folder before the next model operation.
        path = context.kernel_data / "story-attempts" / f"{context.phase}-{context.attempt}-{number:02}.json"
        atomic_json(path, attempt.model_dump(mode="json"))

    attempts, chosen = generate_reviewed(request, prompts, generate, review_complete,
                                        on_attempt=record_attempt)
    accepted = bool(chosen.candidate) and not chosen.blocked
    review = chosen.review
    verdict = OntologyVerdict(
        version="speech-review-v2", accepted=accepted,
        facts_checked=len(request.public_context.evidence) if request.public_context else len(request.memories),
        fact_conflicts=[finding_summary(review)] if review is not None and review.findings else [],
    )
    report = StoryReport(
        companion_name=request.companion_name,
        message_roles=chosen.message_roles,
        reply_prefix="",
        raw_completion=chosen.raw_completion,
        candidate=chosen.candidate,
        accepted=accepted,
        ontology=verdict,
        attempts=attempts,
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
        "summary": f"story candidate {'accepted' if accepted else 'blocked'} "
                   f"after {len(attempts)} attempts, review {review.judgment if review else 'none'}",
    }


__all__ = [
    "build_directive_text",
    "build_opening_text",
    "build_story_messages",
    "build_system_text",
    "clean_reply",
    "generate_reviewed",
    "load_prompts",
    "render_prior_dialogue",
    "review_context",
    "review_conversation",
    "run",
]
