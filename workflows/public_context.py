"""One evidence and identity contract for both public dialogue paths.

The host freezes accepted state before generation. Private state guides speech
but does not grant disclosure or establish an external event.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pm.records.memory_records import ActionSchemaKnoxel
from pm.records.cognitive_records import OntologyBindingKnoxel, OntologyTermKnoxel
from workflows.main_cognition.state_changes import active_goal_texts


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


def build_public_context(ghost, payload: dict, *, capabilities: dict | None = None) -> PublicContext:
    """Freeze one accepted snapshot without importing raw private thoughts.

    Only an existing action can grant presentation intent. Invalid deception
    intent raises ValueError before generation. Source revisions guard commit.
    """
    view = ghost.snapshot()
    action = view.get_knoxel_by_id(payload.get("knoxel_id"))
    metadata = action.metadata if isinstance(action, ActionSchemaKnoxel) else {}
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
    evidence = list(metadata.get("memory_evidence", payload.get("memory_evidence", [])))
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
            originals.append(original)
    states = ghost.states
    state = states[-1] if states else None
    affect = metadata.get("affect_summary") or payload.get("affect", "")
    regulation = str(metadata.get("regulation") or payload.get("regulation", ""))
    if state is not None:
        from pm_core.model.affect_dynamics import summarize_affect
        from pm.records.mental_state import FullMentalState
        affect = summarize_affect(FullMentalState.init_from_list(state.mental_state_latent_vec))
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
        if isinstance(item, OntologyBindingKnoxel) and not item.active:
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
        character=ghost.get_simple_character_story_block(), affect=str(affect),
        regulation=regulation,
        goals=active_goal_texts(ghost)[:12],
        memories=[item["text"] for item in originals if item["knoxel_id"] != getattr(action, "id", None)]
                 or list(payload.get("memories", [])),
        evidence=originals, workspace=str(payload.get("workspace", "")),
        selected_action=str(action.content if action is not None else payload.get("content", "")),
        temporal_context=ghost.build_body_and_situation_addendum(),
        capabilities=domain_capabilities, ontology=ontology, presentation=presentation,
        action_schema_id=action.id if isinstance(action, ActionSchemaKnoxel) else None,
        revisions={key: view.revisions[key] for key in ids if key in view.revisions},
        user_frontier=max(users, default=None), task_receipt=dict(payload.get("task_receipt", {})),
        state_tick=state.tick_id if state is not None else None,
    )


def render_public_context(context: PublicContext) -> str:
    """Render the same bounded evidence for Story, Assistant, and claim review."""
    import json
    return json.dumps(context.model_dump(mode="json", exclude={"revisions"}), ensure_ascii=False)


def render_factual_focus(context: PublicContext) -> str:
    """Repeat original evidence at the answer boundary, after voice priors.

    Reports retain actor attribution and uncertainty. This block supplies no
    invented summary and cannot promote prior companion prose into facts.
    """
    import json
    reports = [json.dumps(item, ensure_ascii=False, default=str) for item in context.evidence
               if item.get("evidence_role") in {"user_report", "runtime_event", "world_observation"}]
    return ("Answer the current request from these original reports and the actual task receipt. "
            "Prior companion turns are recorded speech, not a reply prefix to continue. "
            "Preserve each reported actor, timestamp, and uncertainty. Do not copy unrelated prior reply text.\n"
            "These selected reports are a partial snapshot. Do not call them the only event or failure in the whole log. "
            "Attribute a reported suspicion as a suspicion rather than a verified action.\n"
            "A grounding rule is not a failed historical attempt. Do not invent causes that connect unrelated reports.\n"
            "When asked what you actually know, distinguish supplied evidence from general possibilities. "
            "Do not present an unsourced technical explanation or treatment as an established fact. "
            "State uncertainty when the subject, observation, or authoritative source is missing.\n"
            + "\nCurrent clock: " + context.temporal_context + "\n" + "\n".join(reports)
            + "\nOlder reported conditions are not current observations. Attribute an archived condition to its recorded time. "
            "Do not state that it is true today without a current user report or observation. "
            + "\nWhen the user asks what you actually know about advice, first state what the supplied reports establish. "
            "If the object identity or an authoritative source is missing, say that directly and ask for the missing information. "
            "Do not supply a treatment or a scientific explanation from model knowledge. "
            "A selected intent requesting botanical or other technical facts cannot create the missing evidence. "
            "Fulfil that intent with uncertainty instead of invented verification.")


__all__ = ["Presentation", "PublicContext", "build_public_context", "render_public_context", "render_factual_focus"]
