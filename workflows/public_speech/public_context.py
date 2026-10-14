"""One evidence and identity contract for both public dialogue paths.

The host freezes accepted state before generation. Private state guides speech
but does not grant disclosure or establish an external event.
"""
from __future__ import annotations

import re
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pm.records.memory_records import ActionSchemaRecord
from pm.records.cognitive_records import OntologyBindingRecord
from pm.context_render import ContextKind, render_context
from pm.context_render.turns import Content
from pathlib import Path
from workflows.main_cognition.stage_commits import active_goal_texts


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


class PublicContext(BaseModel):
    """Accepted context, source guards, and domain limits for one utterance."""

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["pm.public-context.v1"] = "pm.public-context.v1"
    companion_name: str
    user_name: str
    character: str = ""
    affect: str = ""
    # The same accepted state in words, without numbers, for public prompts.
    feelings: str = ""
    regulation: str = ""
    goals: list[str] = Field(default_factory=list)
    memories: list[str] = Field(default_factory=list)
    evidence: list[dict] = Field(default_factory=list)
    workspace: str = ""
    selected_action: str = ""
    temporal_context: str = ""
    capabilities: dict = Field(default_factory=dict)
    ontology: list[dict] = Field(default_factory=list)
    presentation: Presentation = Field(default_factory=Presentation)
    action_schema_id: int | None = None
    revisions: dict[int, int] = Field(default_factory=dict)
    user_frontier: int | None = None
    task_receipt: dict = Field(default_factory=dict)
    state_tick: int | None = None
    # The context plan trace (lanes, budgets, semantic status) when planned.
    plan: dict = Field(default_factory=dict)


@render_context.register(PublicContext)
def render_public_context(context: PublicContext, kinds: ContextKind = ContextKind.All,
                          *, schema_path: str | Path | None = None, dedupe: bool = True) -> Content:
    """Select public context sections without exposing storage ownership fields.

    Evidence retains original IDs, scope, attribution, and chronology. The
    original context still owns revisions and publication guards. When a
    cognitive stage supplies newer affect, it excludes this emotion section.
    """
    selected: dict = {"companion_name": context.companion_name, "user_name": context.user_name}
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
        selected["evidence"] = context.evidence
        if not context.evidence:
            selected["memories"] = context.memories
    if kinds & ContextKind.Ontology:
        selected["ontology"] = context.ontology
    return render_context(selected, kinds, schema_path=schema_path, dedupe=dedupe)


def build_public_context(ghost, payload: dict, *, capabilities: dict | None = None, planned=None) -> PublicContext:
    """Freeze one accepted snapshot without importing raw private thoughts.

    Only an existing action can grant presentation intent. Invalid deception
    intent raises ValueError before generation. Source revisions guard commit.

    ``planned`` is a ``PlannedContext`` from ``context_planning``. When it is
    given, its lane-tagged records replace the cognition-selected evidence and
    the 48-record source-chain expansion: the plan already budgeted them. The
    revisions of their source records still guard the commit.
    """
    view = ghost.snapshot()
    action = view.get_knoxel_by_id(payload.get("knoxel_id"))
    metadata = action.metadata if isinstance(action, ActionSchemaRecord) else {}
    records = view.serialize_knoxels()
    presentation = Presentation.model_validate(metadata.get("presentation", {}))
    from pm.memory.retrieval import candidate, evidence_role
    if presentation.mode == "deceive" and (
        not presentation.purpose or not presentation.known_fact_ids
        or any(key not in view.revisions for key in presentation.known_fact_ids)
        or any(evidence_role(view, records[key]) in {"companion_utterance", "intent", "imagined", "interpretation"}
               for key in presentation.known_fact_ids if key in view.revisions)
    ):
        raise ValueError("Deception requires a prior purpose and retained facts.")
    evidence = (list(planned.evidence) if planned is not None
                else list(metadata.get("memory_evidence", payload.get("memory_evidence", []))))
    lanes = {int(item["knoxel_id"]): item["lane"] for item in evidence if item.get("lane")}
    ids = {key for item in evidence for key in [item["knoxel_id"], *item.get("source_ids", [])]}
    ids.update(presentation.known_fact_ids)
    ids.update(int(item["knoxel_id"]) for item in payload.get("stimuli", []) if "knoxel_id" in item)
    # Original startup events establish restart chronology beside retained facts.
    startup_ids = sorted(key for key, record in records.items()
                         if record["data"].get("metadata", {}).get("origin") == "lifecycle"
                         and record["data"]["metadata"].get("event") == "startup")[-3:]
    ids.update(startup_ids)
    if action is not None:
        ids.add(action.id)
    originals = []
    selected_ids = [int(item["knoxel_id"]) for item in evidence]
    evidence_use = metadata.get("memory_trace", {}).get("query_rewrite", {}).get("evidence_use", "general")
    stimulus_ids = [int(item["knoxel_id"]) for item in payload.get("stimuli", []) if "knoxel_id" in item]
    # Reserve original leaves beside selected summaries. A long source chain
    # must not turn one direct chat request into an unbounded history dump.
    if planned is not None:
        order = list(dict.fromkeys([*stimulus_ids, *presentation.known_fact_ids, *startup_ids, *selected_ids]))
    else:
        order = list(dict.fromkeys([*stimulus_ids, *presentation.known_fact_ids, *selected_ids, *startup_ids,
                                   *sorted(ids - set(selected_ids))]))[:48]
    for key in order:
        if key in records and key != getattr(action, "id", None):
            if (evidence_use == "event" and key not in selected_ids
                    and evidence_role(view, records[key]) == "companion_utterance"):
                continue
            original = candidate(view, key, records[key])
            if key in startup_ids:
                original["lifecycle_event"] = "startup"
            if key in lanes:
                original["lane"] = lanes[key]
            originals.append(original)
    states = ghost.states
    state = states[-1] if states else None
    affect = metadata.get("affect_summary") or payload.get("affect", "")
    feelings = ""
    regulation = str(metadata.get("regulation") or payload.get("regulation", ""))
    if state is not None:
        from workflows.main_cognition.affect_dynamics import summarize_affect
        from pm.records.mental_state import FullMentalState
        accepted = FullMentalState.init_from_list(state.mental_state_latent_vec)
        affect = summarize_affect(accepted)
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
    ids.update(int(item["knoxel_id"]) for item in payload.get("stimuli", []) if "knoxel_id" in item)
    config = ghost.ghost_config
    domain_capabilities = capabilities or {
        "text": True, "persistent_memory": True, "human_senses": [], "human_body": False,
        "configured": list(config.supported_capabilities),
        "unavailable": list(map(str, config.unsupported_capabilities)),
        "notes": list(map(str, config.capability_notes)),
        "minecraft": {"status": "unknown", "visual_interpretation": False},
    }
    if state is not None:
        from pm.records.mental_state import FullMentalState
        domain_capabilities["emotion_state"] = FullMentalState.init_from_list(state.mental_state_latent_vec).state_emotions.model_dump()
    rendered_ids = {item["knoxel_id"] for item in originals}
    domain_capabilities["evidence_limit"] = {"records": 48, "record_characters": 1200,
        "source_use": evidence_use, "omitted_ids": sorted(ids - rendered_ids - {getattr(action, "id", None)})}
    return PublicContext(
        companion_name=ghost.get_companion_name(), user_name=ghost.get_user_name(),
        character=ghost.get_simple_character_story_block(), affect=str(affect), feelings=feelings,
        regulation=regulation,
        goals=active_goal_texts(ghost)[:12],
        memories=[item["text"] for item in originals if item["knoxel_id"] != getattr(action, "id", None)]
                 or list(payload.get("memories", [])),
        evidence=originals, workspace=str(payload.get("workspace", "")),
        selected_action=str(action.content if action is not None else payload.get("content", "")),
        temporal_context=ghost.build_body_and_situation_addendum(),
        capabilities=domain_capabilities, ontology=ontology, presentation=presentation,
        action_schema_id=action.id if isinstance(action, ActionSchemaRecord) else None,
        revisions={key: view.revisions[key] for key in ids if key in view.revisions},
        user_frontier=max(users, default=None), task_receipt=dict(payload.get("task_receipt", {})),
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
    people = {context.companion_name, context.user_name}
    rows = []
    for item in context.evidence:
        if int(item["knoxel_id"]) in excluded or item.get("memory_kind") in _HIDDEN_KINDS:
            continue
        if lanes is not None and item.get("lane", "focus") not in lanes:
            continue
        text = str(item.get("text", ""))
        match = _LABEL.match(text)
        label = match.group("label") if match else ""
        body = " ".join((text[match.end():] if match else text).split())
        if not body or body in people or body in shown:
            continue
        source = _SOURCE.search(label)
        speaker = source.group(1).strip() if source else ""
        kind = str(item.get("memory_kind", ""))
        if item.get("lifecycle_event") == "startup":
            phrase = "What I know about my own time away"
        elif speaker in people and item.get("evidence_role") in {"user_report", "companion_utterance"}:
            phrase = f"{speaker} said"
        else:
            # The section title already says what a lane holds. Only kinds that
            # change how real the content is keep a label.
            phrase = _KIND_WORDS.get(kind, "").format(user_name=context.user_name)
        when = format_moment(item.get("timestamp"))
        label = ", ".join(part for part in (when, phrase) if part)
        rows.append((str(item.get("timestamp") or ""), f"- {label + ': ' if label else ''}{body}"))
    return [line for _, line in sorted(rows, key=lambda row: row[0])]


def memory_sections(context: PublicContext, sections: list[dict], *, exclude_ids=(), exclude_texts=()) -> list[str]:
    """Render one titled block per context-plan lane, skipping empty lanes.

    ``sections`` is the ordered list of ``{"lane", "title"}`` mappings from a
    workflow prompt file. Titles may use ``{companion_name}`` and ``{user_name}``.
    """
    names = {"companion_name": context.companion_name, "user_name": context.user_name}
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
    return lines


__all__ = ["DialogueTurn", "Presentation", "PublicContext", "ability_lines", "build_public_context", "describe_elapsed", "feelings_in_words",
           "format_moment", "memory_lines", "memory_sections", "plain_memory_text", "time_lines", "task_lines"]
