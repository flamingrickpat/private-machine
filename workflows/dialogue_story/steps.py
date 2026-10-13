"""Story dialogue role.

Entry point: ``run(context)``. The role builds the three Story messages,
generates one companion line, reviews it for hallucinations, generates again
when the review weight is too high, and writes a full report to
``outbox/report.yaml``.

The message roles are a contract:

  1. system    - the story-writer instruction
  2. user      - all derived material: character card, inner architecture,
                 abilities, time, memories, goals, current feelings, intent
  3. assistant - only verbatim prose: prior dialogue exactly as spoken, the
                 current user line, and the open prefill ``<name> says: "``

Verbatim speech and derived prose never share a message. The voice is
anchored only by real dialogue, and summary wording cannot leak into it.

Generation attempts (``speech_review.yaml``):

- Each attempt is reviewed once. Lies, teasing, and exaggeration never count.
- An attempt below ``regenerate_at`` is published at once.
- Otherwise the next attempt receives a short "avoid this" note in the user
  message. After ``max_attempts``, the lowest-weight attempt is published,
  unless it still has a blocking kind (a false task result).

The workflow never stores a model-invented next user utterance. No tool call
is requested from the generation model. ``build_story_messages`` is pure and
is tested directly without a kernel, model, or MemoryStore.
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path

import yaml
from pm.memory.retrieval import utc_time
from workflows.public_speech.public_context import (
    ability_lines, memory_sections, plain_memory_text, task_lines, time_lines,
)
from workflows.public_speech.speech_review import (
    is_blocked, load_policy, regeneration_note, review_speech, should_regenerate,
)

from workflows.dialogue_story.schemas import (
    DialogueTurn,
    OntologyVerdict,
    OntologyViolation,
    StoryAttempt,
    StoryReport,
    StoryRequest,
)

WORKFLOW_DIR = Path(__file__).resolve().parent


def load_prompts(path: Path | None = None) -> dict:
    return yaml.safe_load((path or WORKFLOW_DIR / "prompts.yaml").read_text(encoding="utf-8"))


def load_ontology(path: Path | None = None) -> dict:
    return yaml.safe_load((path or WORKFLOW_DIR / "ontology.yaml").read_text(encoding="utf-8"))


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


def _inner_lines(request: StoryRequest, prompts: dict, names: dict) -> list[str]:
    lines = []
    if request.feelings:
        lines.append(prompts["inner_feelings"].format(feelings=request.feelings, **names))
    if request.regulation in prompts["inner_regulation"]:
        lines.append(prompts["inner_regulation"][request.regulation].format(**names))
    if request.workspace.strip():
        lines.append(prompts["inner_mind"].format(workspace=request.workspace.strip(), **names))
    if request.selected_action.strip():
        lines.append(prompts["inner_intent"].format(intent=request.selected_action.strip(), **names))
    presentation = request.public_context.presentation if request.public_context else None
    mode = presentation.mode if presentation else "candid"
    lines.append(prompts["presentation"][mode].format(
        summary=getattr(presentation, "shareable_summary", "") or "nothing",
        purpose=getattr(presentation, "purpose", "") or "unstated", **names))
    return lines


def build_context_message(request: StoryRequest, prompts: dict, *, avoid: str = "") -> str:
    """Render the user message: every derived or summarized piece of context.

    Verbatim dialogue never appears here. Memories that duplicate a verbatim
    transcript line, or the current message, are left out.
    """
    names = {"companion_name": request.companion_name, "user_name": request.user_name}
    context = request.public_context
    shown = {turn.knoxel_id for turn in request.prior_dialogue if turn.knoxel_id is not None}
    if request.current_message_id is not None:
        shown.add(request.current_message_id)
    fallback_title = prompts["memory_lanes"][-1]["title"].format(**names)
    if context is not None:
        abilities = ability_lines(context)
        transcript_texts = [turn.text for turn in request.prior_dialogue] + [request.current_user_message]
        memory_blocks = memory_sections(context, prompts["memory_lanes"], exclude_ids=shown,
                                        exclude_texts=transcript_texts)
        if not memory_blocks:
            # Plain memory strings are the fallback when the action carried no
            # evidence records, for example in a standalone situation.
            shown_texts = {" ".join(text.split()) for text in transcript_texts}
            memory_blocks = [_section(fallback_title, [f"- {plain_memory_text(item)}" for item in request.memories
                                                       if plain_memory_text(item)
                                                       and plain_memory_text(item) not in shown_texts])]
        tasks = task_lines(context)
    else:
        abilities = [f"- {item}" for item in request.supported_capabilities]
        abilities += [f"- Cannot: {item}" for item in request.unsupported_capabilities]
        abilities += [f"- Note: {item}" for item in request.capability_notes]
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
        _section(prompts["time_title"], time_lines(request.now, request.prior_dialogue, prompts, names)),
        *memory_blocks,
        _section(prompts["goals_title"].format(**names), [f"- {goal}" for goal in request.goals]),
        _section(prompts["tasks_title"], tasks),
        _section(prompts["inner_title"].format(**names), _inner_lines(request, prompts, names)),
        _section(prompts["regeneration_title"], [avoid]) if avoid else "",
        prompts["context_end"].format(**names).strip(),
    ]
    return "\n\n".join(part for part in parts if part)


def build_story_messages(request: StoryRequest, prompts: dict, *, avoid: str = "") -> list[dict]:
    """Return the three contract messages: system, derived context, verbatim story."""
    names = {"companion_name": request.companion_name, "user_name": request.user_name}
    transcript = render_prior_dialogue(request.prior_dialogue, prompts)
    current = prompts["speaker_line"].format(speaker=request.user_name, text=request.current_user_message)
    story = "\n".join(part for part in [transcript, current, prompts["reply_prefix"].format(**names)] if part)
    return [
        {"role": "system", "content": prompts["system_prompt"].strip()},
        {"role": "user", "content": build_context_message(request, prompts, avoid=avoid)},
        {"role": "assistant", "content": story},
    ]


def clean_reply(raw: str, companion_name: str, user_name: str) -> str:
    """Keep only the companion's continuation.

    The backend may echo the assistant prefill or may continue after it. Remove
    an echoed prefix. Then cut the text at the first marker of any other
    speaker or the start of a non-public state line. This prevents a
    model-invented next user turn from becoming stored dialogue.
    """
    reply = (raw or "").strip()
    for prefix in (f"{companion_name} says:", f"{companion_name}:"):
        if reply.lower().startswith(prefix.lower()):
            reply = reply[len(prefix):].lstrip()
            if reply.startswith('"'):
                reply = reply[1:]
            break

    # Cut at the first other-speaker or non-public marker before trimming any
    # preamble. A model-invented user turn is never kept, even when the model
    # later reopens the companion line.
    other_markers = [
        f"\n{user_name} says:", f"{user_name} says:",
        f"\n{user_name}:", f"{user_name}:",
        "\nStateUpdate:", "\nNarrative:", "\nFact:",
        # The transcript shows earlier private thoughts. A continuation that
        # starts a new thought line is not speech.
        f"\n{companion_name} thinks:",
        # A new day marker is narration, not speech.
        "\n*(",
    ]
    cut = len(reply)
    for marker in other_markers:
        index = reply.find(marker)
        if index != -1:
            cut = min(cut, index)
    reply = reply[:cut].strip()

    own_markers = [
        f"\n{companion_name} says:", f"{companion_name} says:",
        f"\n{companion_name}:", f"{companion_name}:",
    ]
    last_end = 0
    for marker in own_markers:
        index = reply.rfind(marker)
        if index != -1:
            last_end = max(last_end, index + len(marker))
    candidate = reply[last_end:].strip()
    if candidate:
        reply = candidate

    if reply.startswith('"') and reply.endswith('"') and len(reply) > 1:
        reply = reply[1:-1].strip()
    elif reply.endswith('"') and reply.count('"') % 2 == 1:
        reply = reply[:-1].rstrip()
    return reply


def check_ontology(candidate: str, ontology: dict, request: StoryRequest) -> OntologyVerdict:
    """Reject explicit unsupported capability claims.

    Rudeness, sarcasm, insults, and swearing are inside the established voice
    and are not checked. Only named capability or perception claims are.
    """
    violations: list[OntologyViolation] = []
    for rule in ontology.get("disallowed_claims", []):
        match = re.search(rule["pattern"], candidate)
        if match:
            violations.append(OntologyViolation(
                rule_id=str(rule["id"]), reason=str(rule["reason"]),
                match=match.group(0)[:120],
            ))

    fact_conflicts = _fact_conflicts(candidate, request)
    return OntologyVerdict(
        version=str(ontology.get("version", "unknown")),
        accepted=not violations and not fact_conflicts,
        violations=violations,
        facts_checked=len(request.memories),
        fact_conflicts=fact_conflicts,
    )


def _fact_conflicts(candidate: str, request: StoryRequest) -> list[str]:
    """Minimal established-fact guard.

    A referenced fact is only used to reject a literal denial of that same
    fact. The boundary does not try to validate arbitrary new claims, which
    would need a real contradiction model. No fact means no fact check.
    """
    conflicts: list[str] = []
    lowered = candidate.casefold()
    for fact in request.memories:
        text = fact.strip()
        if not text:
            continue
        core = text.replace("*(Fact:", "").replace(")*", "").strip()
        words = [word for word in re.findall(r"[a-z]+", core.casefold()) if len(word) > 4]
        if len(words) < 2:
            continue
        if all(word in lowered for word in words[:2]) and re.search(
            r"(?i)\b(never|not true|didn't happen|did not happen|no such)\b", lowered
        ):
            conflicts.append(text[:160])
    return conflicts


def generate_reviewed(request: StoryRequest, prompts: dict, generate, review_complete) -> tuple[list[StoryAttempt], StoryAttempt]:
    """Generate, review, and regenerate until an attempt passes or attempts run out.

    ``generate`` and ``review_complete`` are async completion callables. The
    result is every attempt plus the chosen one: the first attempt below the
    regeneration weight, otherwise the attempt with the lowest weight (the
    earliest one on a tie). Model transport errors propagate.
    """
    policy = load_policy()
    attempts: list[StoryAttempt] = []
    avoid = ""
    for _ in range(int(policy["policy"]["max_attempts"])):
        messages = build_story_messages(request, prompts, avoid=avoid)
        raw = asyncio.run(generate(messages, **prompts.get("generation", {})))
        if not isinstance(raw, str):
            raise TypeError("story_complete must return the completion text as a string.")
        candidate = clean_reply(raw, request.companion_name, request.user_name)
        review = asyncio.run(review_speech(review_complete, candidate, request.public_context,
                                           conversation=[*[turn for turn in request.prior_dialogue if turn.kind == "speech"][-6:], DialogueTurn(
                                               speaker=request.user_name, text=request.current_user_message)],
                                           policy=policy))
        attempts.append(StoryAttempt(raw_completion=raw, candidate=candidate, review=review, avoid=avoid,
                                     blocked=is_blocked(review, policy)))
        if candidate and not should_regenerate(review, policy):
            break
        avoid = regeneration_note(review, policy) if review.violations else ""
    usable = [item for item in attempts if item.candidate] or attempts
    chosen = min(usable, key=lambda item: item.review.score)
    return attempts, chosen


def run(context) -> dict:
    """Kernel entry point. Runs synchronously on the calling thread."""
    environment = context.runtime.environment
    request = StoryRequest.model_validate(
        yaml.safe_load(environment.read_text("inbox/request.yaml"))
    )
    prompts = load_prompts()
    tools = context.runtime.tools
    if request.public_context is not None:
        attempts, chosen = generate_reviewed(request, prompts, tools["story_complete"], tools["public_complete"])
        accepted = bool(chosen.candidate) and not chosen.blocked
        verdict = OntologyVerdict(version="speech-review-v1", accepted=accepted,
                                  facts_checked=len(request.public_context.evidence),
                                  fact_conflicts=[f"{item.kind}: {item.quote}" for item in chosen.review.violations])
    else:
        messages = build_story_messages(request, prompts)
        raw = asyncio.run(tools["story_complete"](messages, **prompts.get("generation", {})))
        candidate = clean_reply(raw, request.companion_name, request.user_name)
        verdict = check_ontology(candidate, load_ontology(), request)
        chosen = StoryAttempt(raw_completion=raw, candidate=candidate)
        attempts = [chosen]

    messages = build_story_messages(request, prompts, avoid=chosen.avoid)
    report = StoryReport(
        companion_name=request.companion_name,
        message_roles=[message["role"] for message in messages],
        reply_prefix=prompts["reply_prefix"].format(companion_name=request.companion_name),
        raw_completion=chosen.raw_completion,
        candidate=chosen.candidate,
        accepted=verdict.accepted,
        ontology=verdict,
        attempts=attempts,
    )
    environment.write_text(
        "scratch/report.yaml",
        yaml.safe_dump(report.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
    )
    environment.move("scratch/report.yaml", "outbox/report.yaml")
    return {
        "status": "done",
        "summary": f"story candidate {'accepted' if verdict.accepted else 'rejected'} "
                   f"after {len(attempts)} attempt(s), review weight {chosen.review.score if chosen.review else 0}",
    }


__all__ = [
    "build_context_message",
    "build_story_messages",
    "clean_reply",
    "check_ontology",
    "generate_reviewed",
    "load_ontology",
    "load_prompts",
    "render_prior_dialogue",
    "run",
]
