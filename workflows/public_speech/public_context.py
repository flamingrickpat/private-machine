"""One evidence and identity contract for both public dialogue paths.

The host freezes accepted state before generation. Private state guides speech
but does not grant disclosure or establish an external event.
"""
from __future__ import annotations

from workflows.minecraft_body.capabilities import CapabilitySnapshot, CapabilityEvidenceLimit
from workflows.public_speech.task_receipts import TaskReceipt, render_task_receipt, read_optional_task_receipt

import re
from dataclasses import dataclass, field
from datetime import datetime
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator

from pm import vclock
from pm.memory.dependencies import SourceDependency, source_fingerprint
from pm.records.base import RecordScope, RecordBase
from pm.memory.evidence import EvidenceItem as EvidenceItem, EvidenceScoreComponents as EvidenceScoreComponents
from pm.memory.temporal_context import TemporalHistoryInput
from pm.records.enums import FeatureType, InterlocusType, SourceKind
from pm.records.memory_records import Feature
from pm.records.qualia_records import QualeRecord
from pm.records.cognitive_records import OntologyBindingRecord, OntologyTermRecord
from workflows.public_speech.context_trace import ContextPlanningTrace
from workflows.public_speech.ontology_context import PublicOntologyEntry, PublicOntologyTerm, PublicOntologyBinding
from pm.context_render import ContextKind
from pm.context_render.turns import Content
from workflows.entity_context.schemas import EntityContextView
from workflows.entity_context.projection import EntityContextChanged, validate_entity_context
from workflows.main_cognition.cycle_commits import active_goal_texts
from workflows.public_speech.dependencies import (
    InteractionContextChanged, WorkDependencies, capture_dependencies, validate_dependencies,
)
from workflows.public_speech.situational_guidance import (
    SituationalGuidance, build_situational_guidance, render_situational_guidance,
)

if TYPE_CHECKING:
    # main_cognition.schemas imports Presentation from this module.
    from pm.memory.store import MemoryStore
    from pm.records.mental_state import FullMentalState
    from workflows.main_cognition.schemas import ActionRequest
    from workflows.public_speech.context_planning import PlannedContext


ActionOrigin = Literal["main", "interaction", "input_triage", "task_reply"]


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
    known_fact_ids: list[int] = Field(default_factory=list[int])
    shareable_summary: str = ""


@dataclass(order=True, frozen=True, slots=True)
class RankedPublicEvidence:
    negative_score: float
    tie_break: int
    evidence: EvidenceItem = field(compare=False)


def rank_public_evidence(evidence: Sequence[EvidenceItem], *, prefer_recent: bool = False) -> tuple[EvidenceItem, ...]:
    ranked = [RankedPublicEvidence(-item.score, -item.knoxel_id if prefer_recent else item.knoxel_id, item)
              for item in evidence]
    ranked.sort()
    return tuple(item.evidence for item in ranked)


@dataclass(order=True, frozen=True, slots=True)
class RankedMemoryText:
    negative_matches: int
    position: int
    text: str = field(compare=False)


def rank_memory_texts(memories: Sequence[str], focus: str) -> tuple[RankedMemoryText, ...]:
    from pm.memory.retrieval import distinctive_terms
    terms = set(distinctive_terms(focus))
    ranked: list[RankedMemoryText] = []
    seen: set[str] = set()
    for index, text in enumerate(memories):
        normalized = " ".join(text.casefold().split())
        matches = len(terms.intersection(distinctive_terms(text)))
        if matches and normalized not in seen:
            ranked.append(RankedMemoryText(-matches, index, text))
            seen.add(normalized)
    ranked.sort()
    return tuple(ranked)


@dataclass(order=True, frozen=True, slots=True)
class ChronologicalDialogue:
    timestamp: str
    record_id: int
    position: int
    dialogue: DialogueTurn = field(compare=False)


def order_dialogue(dialogue: Sequence[DialogueTurn]) -> list[DialogueTurn]:
    ordered = [ChronologicalDialogue(item.timestamp or "", item.knoxel_id or 0, index, item)
               for index, item in enumerate(dialogue)]
    ordered.sort()
    return [item.dialogue for item in ordered]


class PublicDisclosure(BaseModel):
    model_config = ConfigDict(extra="ignore")
    mode: str


class PublicFeatureMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore")
    origin: str = ""
    event: str = ""
    loop: str = ""
    disclosure: PublicDisclosure | None = None
    memory_kind: str = ""
    regulation: str | None = None


@dataclass(order=True, frozen=True, slots=True)
class PublicRecordRank:
    negative_tick: int
    negative_id: int
    record: RecordBase = field(compare=False)


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
    evidence: list[EvidenceItem] = Field(default_factory=list[EvidenceItem])
    shard_story: str = ""
    workspace: str = ""
    selected_action: str = ""
    temporal_context: str = ""
    temporal_history: TemporalHistoryInput | None = None
    capabilities: CapabilitySnapshot = Field(default_factory=CapabilitySnapshot)
    ontology: list[PublicOntologyEntry] = Field(default_factory=list[PublicOntologyEntry])
    presentation: Presentation = Field(default_factory=Presentation)
    action_schema_id: int | None = None
    revisions: dict[int, int] = Field(default_factory=dict[int, int])
    input_record_ids: list[int] = Field(default_factory=list[int])
    input_commit_ids: list[str] = Field(default_factory=list)
    head_ids: list[int] = Field(default_factory=list[int])
    source_fingerprints: dict[int, str] = Field(default_factory=dict[int, str])
    historical_sources: list[SourceDependency] = Field(default_factory=list[SourceDependency],
        description="Meaning guards for historical guidance, without requiring it to remain current authority.")
    user_frontier: int | None = None
    stimulus_user_ids: list[int] = Field(default_factory=list[int])
    task_receipt: TaskReceipt | None = None

    @field_validator("task_receipt", mode="before")
    @classmethod
    def read_task_receipt(cls, value: object) -> TaskReceipt | None:
        return read_optional_task_receipt(value)

    state_tick: int | None = None
    situational_guidance: SituationalGuidance | None = None
    reconsideration_epoch: int | None = None
    action_origin: ActionOrigin = "main"
    interaction_selection_id: str = ""
    interaction_dependencies: WorkDependencies | None = None
    entity_contexts: list[EntityContextView] = Field(default_factory=list[EntityContextView],
        description="Selected-action and rendered-situation views retained through atomic publication.")
    # The context plan trace (lanes, budgets, semantic status) when planned.
    plan: ContextPlanningTrace | None = None


@dataclass(frozen=True, slots=True)
class PublicContextRenderSettings:
    kinds: ContextKind = ContextKind.All


def render_public_context(context: PublicContext,
                          settings: PublicContextRenderSettings = PublicContextRenderSettings()) -> Content:
    kinds = settings.kinds
    lines = [f"Companion: {context.companion_name}; user: {context.user_name}."]
    if context.partner_name and context.partner_name != context.user_name:
        lines.append(f"Current partner: {context.partner_name}.")
    if kinds & ContextKind.Dialogue:
        lines.extend((context.character, context.workspace, context.selected_action))
        presentation = context.presentation
        lines.append(f"Presentation: {presentation.mode}; purpose: {presentation.purpose}; "
                     f"shareable: {presentation.shareable_summary}")
    if kinds & ContextKind.Emotion:
        lines.append(f"Accepted feelings: {context.affect or context.feelings}; regulation: {context.regulation}")
    if kinds & ContextKind.Facts:
        lines.extend(ability_lines(context))
        lines.extend(f"Goal: {goal}" for goal in context.goals)
        lines.append(context.temporal_context)
        lines.extend(task_lines(context))
        if context.situational_guidance is not None:
            lines.append(render_situational_guidance(context.situational_guidance))
    if kinds & ContextKind.Memories:
        lines.extend(memory_lines(context))
    if kinds & ContextKind.Ontology:
        from workflows.public_speech.ontology import ontology_lines
        lines.extend(ontology_lines(context))
    return Content("\n".join(line for line in lines if line.strip()))


def build_public_context(ghost: MemoryStore, request: ActionRequest | None = None, *,
                         capabilities: CapabilitySnapshot | None = None,
                         planned: PlannedContext | None = None,
                         late_ids: Sequence[int] = ()) -> PublicContext:
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
    for entity_context in request.entity_contexts:
        try:
            validate_entity_context(ghost, entity_context, as_of_utc=vclock.utcnow())
        except EntityContextChanged as error:
            raise InteractionContextChanged(str(error)) from error
    view = ghost.snapshot()
    action = view.get(request.knoxel_id) if request.knoxel_id is not None else None
    metadata = ActionMetadata.model_validate(action.metadata) if action is not None else ActionMetadata()
    if metadata.dispatch_status == "deferred":
        raise InteractionContextChanged("The selected public action already awaits reconsideration.")
    action_id = action.id if action is not None else None
    records = view.all_knoxels
    feature_metadata = {key: PublicFeatureMetadata.model_validate(record.metadata)
                        for key, record in records.items() if isinstance(record, Feature)}
    presentation = metadata.presentation
    from pm.memory.retrieval import candidate, evidence_role
    if presentation.mode == "deceive" and (
        not presentation.purpose or not presentation.known_fact_ids
        or any(key not in view.revisions for key in presentation.known_fact_ids)
        or any(evidence_role(view, view.get(key)) in {"companion_utterance", "intent", "imagined", "interpretation"}
               for key in presentation.known_fact_ids if key in view.revisions)
    ):
        raise ValueError("Deception requires a prior purpose and retained facts.")
    evidence: list[EvidenceItem] = list(planned.evidence) if planned is not None else []
    ranked_evidence = {item.knoxel_id: item for item in evidence}
    lanes = {item.knoxel_id: item.lane for item in evidence if item.lane}
    ids = {key for item in evidence for key in [item.knoxel_id, *item.source_ids]}
    ids.update(presentation.known_fact_ids)
    stimulus_ids = [item.knoxel_id for item in request.stimuli] + list(late_ids)
    ids.update(stimulus_ids)
    if planned is not None:
        ids.update(turn.knoxel_id for turn in planned.dialogue if turn.knoxel_id is not None)
    # Original startup events establish restart chronology beside retained facts.
    startup_ids = sorted(key for key, metadata in feature_metadata.items()
                         if metadata.origin == "lifecycle" and metadata.event == "startup")[-3:]
    ids.update(startup_ids)
    if action_id is not None:
        ids.add(action_id)
    originals: list[EvidenceItem] = []
    selected_ids = [item.knoxel_id for item in evidence]
    # The planned context already budgeted its records. The stimuli, the
    # known facts of a deception, and the startup events come first.
    for key in dict.fromkeys([*stimulus_ids, *presentation.known_fact_ids, *startup_ids, *selected_ids]):
        if key in view.revisions and key != action_id:
            original = candidate(view, key, view.get(key))
            if key in ranked_evidence:
                ranking = ranked_evidence[key]
                original.score = ranking.score
                original.reasons = list(ranking.reasons)
                original.score_components = ranking.score_components.model_copy() if ranking.score_components is not None else None
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
    ranked_records = sorted(PublicRecordRank(-record.tick_id, -key, record) for key, record in records.items())
    for ranked in ranked_records:
        if len(private_state) >= 5:
            break
        record = ranked.record
        key = record.id
        reportable = isinstance(record, QualeRecord) and record.reportable
        candid = False
        if isinstance(record, Feature) and record.feature_type == FeatureType.InnerMonologue:
            private_metadata = feature_metadata[key]
            candid = (private_metadata.loop in {"inner", "inquiry"} and private_metadata.disclosure is not None
                      and private_metadata.disclosure.mode == "candid")
        if not (reportable or candid):
            continue
        item = candidate(view, key, record)
        item.lane = "private_state"
        private_state.append(item)
        ids.add(key)
    originals.extend(private_state)
    states = ghost.read_cycles()
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
        for ranked in ranked_records:
            record = ranked.record
            if not isinstance(record, Feature) or record.tick_id != state.tick_id:
                continue
            state_metadata = feature_metadata[record.id]
            if state_metadata.memory_kind == "internal_state":
                if state_metadata.regulation is not None:
                    regulation = state_metadata.regulation
                break
    ontology: list[PublicOntologyEntry] = []
    for key, item in view.all_knoxels.items():
        if isinstance(item, OntologyTermRecord):
            ontology.append(PublicOntologyTerm(record_id=key, term=item.term, term_kind=item.term_kind,
                definition=item.definition, usage_rule=item.usage_rule))
        elif isinstance(item, OntologyBindingRecord):
            if not item.active:
                continue
            ontology.append(PublicOntologyBinding(record_id=key, term_id=item.term_id,
                fit_score=item.fit_score, explanation=item.explanation))
        else:
            continue
        ids.add(key)
        if len(ontology) == 24:
            break
    users = [key for key, record in records.items() if isinstance(record, Feature)
             and record.feature_type == FeatureType.Dialogue and record.override_source == ghost.get_user_name()]
    config = ghost.ghost_config
    domain_capabilities = capabilities.model_copy(deep=True) if capabilities is not None else CapabilitySnapshot(
        configured=tuple(config.supported_capabilities),
        unavailable=tuple(str(item) for item in config.unsupported_capabilities),
        notes=tuple(str(item) for item in config.capability_notes))
    if state is not None:
        domain_capabilities.emotion_state = state.mental_state.state_emotions.model_copy(deep=True)
    if planned is not None and planned.temporal_history is not None:
        for history_item in planned.temporal_history.items:
            ids.add(history_item.record_id)
            ids.update(history_item.source_ids)
    rendered_ids = {item.knoxel_id for item in originals}
    domain_capabilities.evidence_limit = CapabilityEvidenceLimit(records=48, record_characters=1200,
        omitted_ids=tuple(sorted(ids - rendered_ids - {action_id})))
    guidance = build_situational_guidance(view, as_of_utc=vclock.utcnow(),
        companion_name=ghost.get_companion_name(), user_name=ghost.get_user_name(),
        partner_entity_id=request.partner_entity_id,
        supported_capabilities=config.supported_capabilities,
        scopes=tuple(view.get(key).scope for key in stimulus_ids if key in view.revisions)
               or (action.scope if action is not None else RecordScope(),))
    ids.update(guidance.source_revisions())
    entity_contexts = list(request.entity_contexts)
    if guidance.partner is not None:
        for entity_context in guidance.partner.entity_contexts:
            if entity_context not in entity_contexts:
                entity_contexts.append(entity_context)
    partner_inputs: list[int] = []
    for key in stimulus_ids:
        record = view.get(key)
        if (isinstance(record, Feature) and record.feature_type == FeatureType.Dialogue
                and record.interlocus == InterlocusType.Public and record.provenance is not None
                and record.provenance.source_kind in {SourceKind.User, SourceKind.Agent}):
            partner_inputs.append(key)
    interaction_dependencies: WorkDependencies | None = None
    if request.interaction_dependencies is not None:
        prior = request.interaction_dependencies
        validate_dependencies(ghost, prior, current_partner_input=True)
        # The speech owner freezes current evidence, including later partner
        # lines. It does not reuse the older selection frontier as a new draft.
        interaction_dependencies = capture_dependencies(view, sorted(ids),
            guidance_tick=prior.guidance_tick, critical_epoch=prior.critical_epoch,
            partner_entity_id=request.partner_entity_id, scopes=prior.scopes,
            entity_contexts=entity_contexts)
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
        temporal_history=planned.temporal_history if planned is not None else None,
        capabilities=domain_capabilities, ontology=ontology, presentation=presentation,
        action_schema_id=action_id,
        revisions={key: view.revisions[key] for key in ids if key in view.revisions},
        input_record_ids=list(view.revisions), input_commit_ids=sorted(view.commit_ids),
        head_ids=sorted(ids & set(view.all_knoxels)),
        source_fingerprints={key: source_fingerprint(view.get(key)) for key in ids if key in view.revisions},
        user_frontier=max(users, default=None), task_receipt=request.task_receipt.model_copy(deep=True) if request.task_receipt is not None else None,
        stimulus_user_ids=partner_inputs,
        state_tick=state.tick_id if state is not None else None,
        situational_guidance=guidance,
        action_origin=request.action_origin,
        interaction_selection_id=request.interaction_selection_id,
        interaction_dependencies=interaction_dependencies,
        entity_contexts=entity_contexts,
        plan=planned.trace.model_copy(deep=True) if planned is not None else None,
    )


def feelings_in_words(state: FullMentalState) -> str:
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


def format_moment(value: str | datetime | None) -> str:
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


class TimeTemplates(TypedDict):
    """Required time and conversation-gap templates from a workflow prompt file."""

    time_now: str
    time_since: str
    time_first: str


class MemoryLaneTemplate(TypedDict):
    """One existing prompt section: the retrieval lane and its visible title."""

    lane: str
    title: str


@dataclass(frozen=True, slots=True)
class PromptNames:
    companion_name: str
    partner_name: str


def time_lines(now: str, turns: list[DialogueTurn], templates: TimeTemplates, names: PromptNames) -> list[str]:
    """Describe now and the gap since the last spoken exchange, in words.

    ``templates`` holds ``time_now``, ``time_since``, and ``time_first`` from a
    workflow prompt file. Private thoughts do not count as an exchange.
    """
    from pm.memory.retrieval import utc_time
    lines = [templates["time_now"].format(now=format_moment(now), companion_name=names.companion_name, partner_name=names.partner_name)] if now else []
    stamps = [turn.timestamp for turn in turns if turn.timestamp and turn.kind == "speech"]
    if stamps and now:
        gap = (utc_time(now) - utc_time(max(stamps))).total_seconds()
        lines.append(templates["time_since"].format(elapsed=describe_elapsed(gap), last=format_moment(max(stamps)), companion_name=names.companion_name, partner_name=names.partner_name))
    elif not stamps:
        lines.append(templates["time_first"].format(companion_name=names.companion_name, partner_name=names.partner_name))
    return lines


def memory_lines(context: PublicContext, *, exclude_ids: Iterable[int] = (),
                 exclude_texts: Iterable[str] = (), lanes: set[str] | None = None) -> list[str]:
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
    rows: list[tuple[str, str]] = []
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
        if item.historical:
            phrase = "historical quotation; " + phrase
        label = ", ".join(part for part in (when, phrase) if part)
        rows.append((item.timestamp or "", f"- {label + ': ' if label else ''}{body}"))
    return [line for _, line in sorted(rows, key=lambda row: row[0])]


def memory_sections(context: PublicContext, sections: list[MemoryLaneTemplate], *,
                    exclude_ids: Iterable[int] = (), exclude_texts: Iterable[str] = ()) -> list[str]:
    """Render one titled block per context-plan lane, skipping empty lanes.

    ``sections`` is the ordered list of ``{"lane", "title"}`` mappings from a
    workflow prompt file. Titles may use ``{companion_name}`` and ``{partner_name}``.
    """
    names = PromptNames(context.companion_name, context.partner_name)
    blocks: list[str] = []
    for section in sections:
        lines = memory_lines(context, exclude_ids=exclude_ids, exclude_texts=exclude_texts, lanes={section["lane"]})
        if lines:
            blocks.append(section["title"].format(companion_name=names.companion_name, partner_name=names.partner_name) + "\n" + "\n".join(lines))
    return blocks


def plain_memory_text(text: str) -> str:
    """Remove the machine label from one rendered memory text."""
    match = _LABEL.match(str(text))
    return " ".join((str(text)[match.end():] if match else str(text)).split())


def ability_lines(context: PublicContext) -> list[str]:
    """Render the domain capabilities as short plain sentences."""
    capabilities = context.capabilities
    lines = [f"- {item}" for item in capabilities.configured]
    lines += [f"- Cannot: {item}" for item in capabilities.unavailable]
    lines += [f"- Note: {item}" for item in capabilities.notes]
    if not capabilities.human_body:
        lines.append(f"- {context.companion_name} has no human body and no human senses.")
    if capabilities.runtime_orientation:
        lines.append(capabilities.runtime_orientation)
    minecraft = capabilities.minecraft
    lines.append(f"- Minecraft connection right now: {minecraft.status}.")
    if minecraft.status == "connected":
        lines.append(f"- {context.companion_name} can act inside Minecraft through connected game tools.")
        lines.append("- The tools are reachable now. Connection alone does not establish an observation or completed action.")
    if not minecraft.visual_interpretation:
        lines.append("- Minecraft observations are structured text. Screenshot pixels have no interpretation support in this session.")
    return lines


@dataclass(frozen=True, slots=True)
class ToolEvidence:
    name: str
    executed: bool
    result_text: str
    error: str = ""
    observed_at: str = ""
    source: str = ""


def task_lines(context: PublicContext, receipts: Sequence[ToolEvidence] = ()) -> list[str]:
    """Render task and tool results without removing their verification facts.

    The review receives the same result as the generator. A clipped result
    cannot establish whether a claimed file or world fact was returned.
    """
    lines: list[str] = []
    if context.task_receipt:
        lines.append(render_task_receipt(context.task_receipt))
    for receipt in receipts:
        outcome = "succeeded" if receipt.executed and not receipt.error else "failed"
        lines.append(f"- Tool {receipt.name} {outcome}: {receipt.error or receipt.result_text}")
        if receipt.observed_at or receipt.source:
            lines.append(f"  Observed: {receipt.observed_at}; source: {receipt.source}")
    return lines


__all__ = ["DialogueTurn", "EvidenceItem", "Presentation", "PublicContext", "ability_lines", "build_public_context", "describe_elapsed", "feelings_in_words",
           "format_moment", "memory_lines", "memory_sections", "plain_memory_text", "time_lines", "task_lines"]
