"""One evidence and identity contract for both public dialogue paths.

The host freezes accepted state before generation. Private state guides speech
but does not grant disclosure or establish an external event.
"""
from __future__ import annotations

import re
import json
import hashlib
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from pm.records.base import Provenance, RecordScope
from pm.records.cognitive_records import OntologyBindingRecord
from pm.context_render import ContextKind, render_context
from pm.context_render.turns import Content
from pathlib import Path
from workflows.main_cognition.cycle_commits import active_goal_texts

if TYPE_CHECKING:
    # main_cognition.schemas imports Presentation from this module.
    from workflows.main_cognition.schemas import ActionRequest


class DialogueTurn(BaseModel):
    """One spoken line of public dialogue history."""

    model_config = ConfigDict(extra="forbid")

    speaker: str = Field(min_length=1)
    text: str = Field(min_length=1)
    # Optional source identity and time. Story uses them to avoid repeating a
    # verbatim line as a memory and to mark day changes in the transcript.
    knoxel_id: int | None = None
    timestamp: str | None = None
    # "speech" is public dialogue; "thought" is the companion's own earlier
    # private thought. Only the Story transcript renders thoughts.
    kind: Literal["speech", "thought"] = "speech"


class Presentation(BaseModel):
    """A prior public choice. Deception retains its purpose and known evidence."""

    model_config = ConfigDict(extra="forbid")
    mode: Literal["candid", "withhold", "deceive"] = "candid"
    purpose: str = ""
    known_fact_ids: list[int] = Field(default_factory=list)
    shareable_summary: str = ""


class EvidenceItem(BaseModel):
    """One record of the public context, as ``pm.memory.retrieval.candidate`` renders it.

    ``text`` starts with a machine label (kind, source, evidence id, time).
    ``memory_lines`` turns the label into words. ``lane`` names where the
    record came from: a context-plan lane, ``shard:<name>`` for a workspace
    shard of the cycle, or ``private_state`` for a disclosed private state.
    ``lifecycle_event`` is ``startup`` for a startup event of the runtime.
    """

    model_config = ConfigDict(extra="forbid")

    knoxel_id: int = 0
    text: str = ""
    revision: int = 1
    memory_kind: str = ""
    scope: RecordScope = Field(default_factory=RecordScope)
    provenance: Provenance | None = None
    source_ids: list[int] = Field(default_factory=list)
    timestamp: str | None = None
    historical: bool = False
    score: float = 0.0
    reasons: list[str] = Field(default_factory=list)
    deception_intent_id: int | None = None
    evidence_role: str = ""
    lane: str = ""
    lifecycle_event: str = ""


class PublicContext(BaseModel):
    """Accepted context, source guards, and domain limits for one utterance.

    ``partner_name`` is the entity that the utterance addresses: the user, or
    another entity such as a second companion. ``user_name`` is the user of
    the runtime. ``shard_story`` holds the workspace shards of the cycle that
    chose the utterance, rendered as story without the verbatim dialogue.
    """

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["pm.public-context.v1"] = "pm.public-context.v1"
    companion_name: str
    user_name: str
    partner_name: str = ""
    character: str = ""
    affect: str = ""
    # The same accepted state in words, without numbers, for public prompts.
    feelings: str = ""
    regulation: str = ""
    goals: list[str] = Field(default_factory=list)
    memories: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    shard_story: str = ""
    workspace: str = ""
    selected_action: str = ""
    temporal_context: str = ""
    capabilities: dict = Field(default_factory=dict)
    ontology: list[dict] = Field(default_factory=list)
    presentation: Presentation = Field(default_factory=Presentation)
    action_schema_id: int | None = None
    revisions: dict[int, int] = Field(default_factory=dict)
    source_fingerprints: dict[int, str] = Field(default_factory=dict)
    user_frontier: int | None = None
    stimulus_user_ids: list[int] = Field(default_factory=list)
    task_receipt: dict = Field(default_factory=dict)
    state_tick: int | None = None
    # The context plan trace (lanes, budgets, semantic status) when planned.
    plan: dict = Field(default_factory=dict)


def source_fingerprint(record) -> str:
    """Hash accepted content without delivery and exchange bookkeeping.

    Acknowledgment changes metadata while another utterance reads the same
    source. Those changes do not invalidate its meaning. All other fields,
    including attribution and scope, remain part of the comparison.
    """
    data = record.model_dump(mode="json", exclude={"embedding"})
    metadata = data.get("metadata", {}).copy()
    for key in ("exchange_status", "public_reply_id", "publication_status", "recovery_outcome", "dispatch_status"):
        metadata.pop(key, None)
    data["metadata"] = metadata
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@render_context.register(PublicContext)
def render_public_context(context: PublicContext, kinds: ContextKind = ContextKind.All,
                          *, schema_path: str | Path | None = None, dedupe: bool = True) -> Content:
    """Select public context sections without exposing storage ownership fields.

    Evidence retains original IDs, scope, attribution, and chronology. The
    original context still owns revisions and publication guards. When a
    cognition phase supplies newer affect, it excludes this emotion section.
    """
    selected: dict = {"companion_name": context.companion_name, "user_name": context.user_name}
    if context.partner_name and context.partner_name != context.user_name:
        selected["partner_name"] = context.partner_name
    if kinds & ContextKind.Dialogue:
        selected.update(character=context.character, workspace=context.workspace,
                        selected_action=context.selected_action,
                        presentation=context.presentation.model_dump(mode="python"))
    if kinds & ContextKind.Emotion:
        selected.update(affect=context.affect or context.feelings, regulation=context.regulation)
    if kinds & ContextKind.Facts:
        capabilities = {key: value for key, value in context.capabilities.items() if key != "emotion_state"}
        selected.update(capabilities=capabilities, goals=context.goals, temporal_context=context.temporal_context,
                        task_receipt=context.task_receipt)
    if kinds & ContextKind.Memories:
        selected["evidence"] = [item.model_dump(mode="json") for item in context.evidence]
        if not context.evidence:
            selected["memories"] = context.memories
    if kinds & ContextKind.Ontology:
        selected["ontology"] = context.ontology
    return render_context(selected, kinds, schema_path=schema_path, dedupe=dedupe)


def build_public_context(ghost, request: ActionRequest | None = None, *, capabilities: dict | None = None,
                         planned=None, late_ids: list[int] = ()) -> PublicContext:
    """Freeze one accepted snapshot without importing raw private thoughts.

    ``request`` is the typed action request of the utterance, or None for a
    context without an action (the private grounding turn of main cognition,
    a body workflow). Only an existing action can grant presentation intent:
    the presentation comes from the stored action record, never from the
    request. Invalid deception intent raises ValueError before generation.
    Source revisions guard the commit.

    ``planned`` is a ``PlannedContext`` from ``context_planning``: the shard
    context of the cycle, or the context plan of a standalone situation. Its
    records are the evidence. Without it, the evidence holds the stimuli,
    the startup events, and the disclosed private state only.

    ``late_ids`` are partner dialogue records that arrived after the cycle
    chose the action. They join the stimulus set, so the reply resolves them
    and the commit guard covers them.
    """
    from workflows.main_cognition.schemas import ActionMetadata, ActionRequest
    request = request or ActionRequest()
    view = ghost.snapshot()
    action = view.get(request.knoxel_id) if request.knoxel_id is not None else None
    metadata = ActionMetadata.model_validate(action.metadata) if action is not None else ActionMetadata()
    action_id = action.id if action is not None else None
    records = view.serialize_knoxels()
    presentation = metadata.presentation
    from pm.memory.retrieval import candidate, evidence_role
    if presentation.mode == "deceive" and (
        not presentation.purpose or not presentation.known_fact_ids
        or any(key not in view.revisions for key in presentation.known_fact_ids)
        or any(evidence_role(view, records[key]) in {"companion_utterance", "intent", "imagined", "interpretation"}
               for key in presentation.known_fact_ids if key in view.revisions)
    ):
        raise ValueError("Deception requires a prior purpose and retained facts.")
    evidence: list[EvidenceItem] = list(planned.evidence) if planned is not None else []
    lanes = {item.knoxel_id: item.lane for item in evidence if item.lane}
    ids = {key for item in evidence for key in [item.knoxel_id, *item.source_ids]}
    ids.update(presentation.known_fact_ids)
    stimulus_ids = [item.knoxel_id for item in request.stimuli] + list(late_ids)
    ids.update(stimulus_ids)
    # Original startup events establish restart chronology beside retained facts.
    startup_ids = sorted(key for key, record in records.items()
                         if record["data"].get("metadata", {}).get("origin") == "lifecycle"
                         and record["data"]["metadata"].get("event") == "startup")[-3:]
    ids.update(startup_ids)
    if action_id is not None:
        ids.add(action_id)
    originals: list[EvidenceItem] = []
    selected_ids = [item.knoxel_id for item in evidence]
    # The planned context already budgeted its records. The stimuli, the
    # known facts of a deception, and the startup events come first.
    for key in dict.fromkeys([*stimulus_ids, *presentation.known_fact_ids, *startup_ids, *selected_ids]):
        if key in records and key != action_id:
            original = EvidenceItem.model_validate(candidate(view, key, records[key]))
            if key in startup_ids:
                original.lifecycle_event = "startup"
            if key in lanes:
                original.lane = lanes[key]
            originals.append(original)
    # A17 disclosure: private state enters public context only through the
    # private loop's explicit choice. A reportable felt state names accepted
    # drivers, and a candid thought carries a disclosure recorded before any
    # dialogue generation. Withheld material never reaches this evidence.
    private_state: list[EvidenceItem] = []
    for key, raw in sorted(records.items(),
                          key=lambda item: item[1]["data"].get("tick_id", -1), reverse=True):
        if len(private_state) >= 5:
            break
        private_metadata = raw["data"].get("metadata", {}) or {}
        is_reportable_quale = raw["__type__"] == "QualeKnoxel" and bool(raw["data"].get("reportable"))
        is_candid_thought = (raw["__type__"] == "Feature"
                             and raw["data"].get("feature_type") == "InnerMonologue"
                             and private_metadata.get("loop") == "inner"
                             and (private_metadata.get("disclosure") or {}).get("mode") == "candid")
        if not (is_reportable_quale or is_candid_thought):
            continue
        item = EvidenceItem.model_validate(candidate(view, key, records[key]))
        item.lane = "private_state"
        private_state.append(item)
        ids.add(key)
    originals.extend(private_state)
    states = ghost.states
    state = states[-1] if states else None
    affect = metadata.affect_summary or request.affect
    feelings = ""
    regulation = metadata.regulation or request.regulation
    if state is not None:
        from workflows.main_cognition.affect import describe_mental_state
        # The accepted cycle state is newer than the affect text of the
        # action request. The relationship names its partner by entity id.
        accepted = state.mental_state
        relationship = accepted.state_relationship
        partner = (ghost.get_entity_name(relationship.entity_id) or ""
                   if relationship is not None and relationship.entity_id else "")
        affect = describe_mental_state(accepted, name=ghost.get_companion_name(), partner_name=partner)
        feelings = feelings_in_words(accepted)
        current_state = next((raw["data"] for raw in reversed(list(records.values()))
                              if raw["data"].get("tick_id") == state.tick_id
                              and raw["data"].get("metadata", {}).get("memory_kind") == "internal_state"), None)
        if current_state is not None:
            regulation = str(current_state.get("metadata", {}).get("regulation", regulation))
    ontology = []
    for key, raw in records.items():
        if raw["__type__"] not in {"OntologyTermKnoxel", "OntologyBindingKnoxel"}:
            continue
        item = view.get(key)
        if isinstance(item, OntologyBindingRecord) and not item.active:
            continue
        ontology.append({"id": key, **item.model_dump(mode="json", exclude={"embedding"})})
        ids.add(key)
        if len(ontology) == 24:
            break
    users = [key for key, raw in records.items() if raw["data"].get("feature_type") == "Dialogue"
             and raw["data"].get("override_source") == ghost.get_user_name()]
    config = ghost.ghost_config
    domain_capabilities = capabilities or {
        "text": True, "persistent_memory": True, "human_senses": [], "human_body": False,
        "configured": list(config.supported_capabilities),
        "unavailable": list(map(str, config.unsupported_capabilities)),
        "notes": list(map(str, config.capability_notes)),
        "minecraft": {"status": "unknown", "visual_interpretation": False},
    }
    if state is not None:
        domain_capabilities["emotion_state"] = state.mental_state.state_emotions.model_dump()
    rendered_ids = {item.knoxel_id for item in originals}
    domain_capabilities["evidence_limit"] = {"records": 48, "record_characters": 1200,
        "omitted_ids": sorted(ids - rendered_ids - {action_id})}
    return PublicContext(
        companion_name=ghost.get_companion_name(), user_name=ghost.get_user_name(),
        partner_name=request.partner_name or ghost.get_user_name(),
        character=ghost.get_simple_character_story_block(), affect=str(affect), feelings=feelings,
        regulation=regulation,
        goals=active_goal_texts(ghost)[:12],
        memories=[item.text for item in originals],
        evidence=originals, shard_story=planned.story if planned is not None else "",
        workspace=request.workspace,
        selected_action=str(action.content) if action is not None else request.content,
        temporal_context=ghost.build_body_and_situation_addendum(),
        capabilities=domain_capabilities, ontology=ontology, presentation=presentation,
        action_schema_id=action_id,
        revisions={key: view.revisions[key] for key in ids if key in view.revisions},
        source_fingerprints={key: source_fingerprint(view.get(key)) for key in ids if key in view.revisions},
        user_frontier=max(users, default=None), task_receipt=dict(request.task_receipt),
        stimulus_user_ids=[key for key in stimulus_ids if key in users],
        state_tick=state.tick_id if state is not None else None,
        plan=dict(planned.trace) if planned is not None else {},
    )


def feelings_in_words(state) -> str:
    """Describe an accepted mental state in words, for example "strong fear and mild guilt".

    Core affect and emotions only. Need and appraisal axes read like
    technical labels in prose, so public prompts leave them out.
    """
    from workflows.main_cognition.verbalization import verbalize_core_state, verbalize_emotional_state
    parts = [verbalize_emotional_state(state.state_emotions), verbalize_core_state(state.state_core)]
    return "; ".join(part for part in parts if part)


_LABEL = re.compile(r"^\[(?P<label>[^\]]*)\]\s*")
_SOURCE = re.compile(r"source=([^;\]]+)")
# Labels for memory kinds whose reality differs from a plain memory. Other
# kinds render without a label: the section title says what the lane holds.
_KIND_WORDS = {
    "reported_belief": "Something I believe from what {user_name} told me",
    "observation": "Earlier observation",
    "interpretation": "My earlier interpretation",
    "speculation": "An earlier private thought of mine",
}
# Kinds that never render as prose memories. Internal-state records hold
# numeric affect dumps; feelings reach the prompt in words instead.
_HIDDEN_KINDS = {"internal_state"}


def format_moment(value) -> str:
    """Render a record timestamp as plain words, for example "30 May 2026, 21:03"."""
    if not value:
        return ""
    from pm.memory.retrieval import utc_time
    moment = utc_time(value)
    return f"{moment.day} {moment.strftime('%B %Y, %H:%M')}"


def describe_elapsed(seconds: float | None) -> str:
    """Render a duration as words a person would use, for example "about 4 months"."""
    if seconds is None:
        return ""
    minutes = seconds / 60
    hours = minutes / 60
    days = hours / 24
    if minutes < 2:
        return "a moment"
    if minutes < 90:
        return f"about {round(minutes)} minutes"
    if hours < 36:
        return f"about {round(hours)} hours"
    if days < 14:
        return f"about {round(days)} days"
    if days < 60:
        return f"about {round(days / 7)} weeks"
    if days < 700:
        return f"about {round(days / 30.4)} months"
    return f"about {round(days / 365)} years"


def time_lines(now: str, turns: list[DialogueTurn], templates: dict, names: dict) -> list[str]:
    """Describe now and the gap since the last spoken exchange, in words.

    ``templates`` holds ``time_now``, ``time_since``, and ``time_first`` from a
    workflow prompt file. Private thoughts do not count as an exchange.
    """
    from pm.memory.retrieval import utc_time
    lines = [templates["time_now"].format(now=format_moment(now), **names)] if now else []
    stamps = [turn.timestamp for turn in turns if turn.timestamp and turn.kind == "speech"]
    if stamps and now:
        gap = (utc_time(now) - utc_time(max(stamps))).total_seconds()
        lines.append(templates["time_since"].format(elapsed=describe_elapsed(gap), last=format_moment(max(stamps)), **names))
    elif not stamps:
        lines.append(templates["time_first"].format(**names))
    return lines


def memory_lines(context: PublicContext, *, exclude_ids=(), exclude_texts=(), lanes=None) -> list[str]:
    """Render evidence records as dated plain-language lines, oldest first.

    The machine label of each record (kind, source, evidence id) becomes a
    short phrase and a date. Ids never appear: a model that sees ids tends to
    quote them in speech. Dialogue by the companion or the user reads
    "<name> said", also when it is imported history. Beliefs, observations,
    interpretations, and speculations keep a label; other records need none.

    ``exclude_ids`` and ``exclude_texts`` remove records that the prompt
    already shows verbatim, such as the current message or transcript lines.
    Internal-state records hold numbers, and bare entity records hold only a
    name, so neither renders. ``lanes`` limits the lines to records of those
    context-plan lanes; a record without a lane counts as ``focus``.
    """
    excluded = {int(key) for key in exclude_ids}
    shown = {" ".join(str(text).split()) for text in exclude_texts}
    people = {context.companion_name, context.user_name, context.partner_name} - {""}
    rows = []
    for item in context.evidence:
        if item.knoxel_id in excluded or item.memory_kind in _HIDDEN_KINDS:
            continue
        if lanes is not None and (item.lane or "focus") not in lanes:
            continue
        match = _LABEL.match(item.text)
        label = match.group("label") if match else ""
        body = " ".join((item.text[match.end():] if match else item.text).split())
        if not body or body in people or body in shown:
            continue
        source = _SOURCE.search(label)
        speaker = source.group(1).strip() if source else ""
        if item.lifecycle_event == "startup":
            phrase = "What I know about my own time away"
        elif speaker in people and item.evidence_role in {"user_report", "companion_utterance"}:
            phrase = f"{speaker} said"
        else:
            # The section title already says what a lane holds. Only kinds that
            # change how real the content is keep a label.
            phrase = _KIND_WORDS.get(item.memory_kind, "").format(user_name=context.user_name)
        when = format_moment(item.timestamp)
        label = ", ".join(part for part in (when, phrase) if part)
        rows.append((item.timestamp or "", f"- {label + ': ' if label else ''}{body}"))
    return [line for _, line in sorted(rows, key=lambda row: row[0])]


def memory_sections(context: PublicContext, sections: list[dict], *, exclude_ids=(), exclude_texts=()) -> list[str]:
    """Render one titled block per context-plan lane, skipping empty lanes.

    ``sections`` is the ordered list of ``{"lane", "title"}`` mappings from a
    workflow prompt file. Titles may use ``{companion_name}`` and ``{partner_name}``.
    """
    names = {"companion_name": context.companion_name, "partner_name": context.partner_name}
    blocks = []
    for section in sections:
        lines = memory_lines(context, exclude_ids=exclude_ids, exclude_texts=exclude_texts, lanes={section["lane"]})
        if lines:
            blocks.append(section["title"].format(**names) + "\n" + "\n".join(lines))
    return blocks


def plain_memory_text(text: str) -> str:
    """Remove the machine label from one rendered memory text."""
    match = _LABEL.match(str(text))
    return " ".join((str(text)[match.end():] if match else str(text)).split())


def ability_lines(context: PublicContext) -> list[str]:
    """Render the domain capabilities as short plain sentences."""
    capabilities = context.capabilities or {}
    lines = [f"- {item}" for item in capabilities.get("configured", [])]
    lines += [f"- Cannot: {item}" for item in capabilities.get("unavailable", [])]
    lines += [f"- Note: {item}" for item in capabilities.get("notes", [])]
    if not capabilities.get("human_body", False):
        lines.append(f"- {context.companion_name} has no human body and no human senses.")
    minecraft = capabilities.get("minecraft") or {}
    lines.append(f"- Minecraft connection right now: {minecraft.get('status', 'unknown')}.")
    if minecraft.get("status") == "connected":
        lines.append(f"- {context.companion_name} can act inside Minecraft through connected game tools.")
        lines.append("- The tools are reachable now. Connection alone does not establish an observation or completed action.")
    if minecraft and not minecraft.get("visual_interpretation", False):
        lines.append("- Minecraft observations are structured text. Screenshot pixels have no interpretation support in this session.")
    return lines


def task_lines(context: PublicContext, receipts: list[dict] | None = None) -> list[str]:
    """Render task and tool results without removing their verification facts.

    The review receives the same result as the generator. A clipped result
    cannot establish whether a claimed file or world fact was returned.
    """
    lines = []
    task = context.task_receipt or {}
    if task:
        verified = "verified" if task.get("task_verified") else "not verified"
        identity = f" (run {task['run_id']})" if task.get("run_id") else ""
        lines.append(f"- Task {task.get('workflow_name', 'task')}{identity}: worker state {task.get('state', 'unknown')}, requested effect {verified}.")
        # Worker completion and verified external effects are separate facts.
        # Keep the receipt's original request and artifacts for both readers.
        lines.append("  Receipt: " + json.dumps(task, ensure_ascii=False, default=str))
    for receipt in receipts or []:
        result = receipt.get("result")
        failed = (not receipt.get("executed", True) or receipt.get("error")
                  or (isinstance(result, dict) and result.get("ok") is False))
        outcome = "failed" if failed else "succeeded"
        detail = receipt.get("error") or json.dumps(result, ensure_ascii=False, default=str)
        lines.append(f"- Tool {receipt.get('name', 'tool')} {outcome}: {detail}")
        provenance = {key: receipt[key] for key in ("observed_at", "executed_at", "scope", "source") if receipt.get(key) is not None}
        if provenance:
            lines.append("  Observation time and scope: " + json.dumps(provenance, ensure_ascii=False, default=str))
    return lines


__all__ = ["DialogueTurn", "EvidenceItem", "Presentation", "PublicContext", "ability_lines", "build_public_context", "describe_elapsed", "feelings_in_words",
           "format_moment", "memory_lines", "memory_sections", "plain_memory_text", "time_lines", "task_lines"]
