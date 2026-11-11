"""Project accepted records into the situation shared by cognition and speech.

See :mod:`workflows.public_speech.public_dialogue` for publication ownership.
This projection supplies evidence for reasoning about a partner's ability to
receive and respond, current purposes, and moves already made. It never
classifies availability, writes memory, or calls a model. The caller supplies
runtime UTC time; elapsed time never establishes an unobserved state.

The record store remains authoritative. Source IDs, revisions, attribution,
and world scope travel with the selected evidence. Count limits select recent
records without cutting their text; the rendered view reports those limits.
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pm.memory.snapshot import Snapshot
from pm.records.base import RecordBase, RecordScope
from pm.records.cognitive_records import BeliefRecord, Intention
from pm.records.enums import ActionSchemaStatus, EntityClass, FeatureType, FrameEnum, InterlocusType, NarrativeTypes, SourceKind
from pm.records.memory_records import ActionSchemaRecord, Entity, Feature, Narrative
from pm.records.workspace import ShardKey


class SituationSource(BaseModel):
    """One unabridged, attributed source in a frozen situation projection."""

    record_id: int
    revision: int
    occurred_at_utc: datetime = Field(description="Original world time, as naive UTC.")
    source: str
    kind: str = Field(description="Record type and epistemic role, not a truth assertion.")
    text: str
    scope: RecordScope


class RecentMove(BaseModel):
    """A selected purpose and its actual utterance, when one was committed."""

    action: SituationSource
    utterance: SituationSource | None = None
    status: Literal["unknown", "selected", "committed", "delivered", "failed"]


class EntitySituationEvidence(BaseModel):
    """Partner identity, interpretive context, and actual recent interaction."""

    entity_id: int
    revision: int
    name: str
    entity_class: EntityClass
    context: list[SituationSource] = Field(default_factory=list[SituationSource])
    recent_dialogue: list[SituationSource] = Field(default_factory=list[SituationSource])


class SituationalGuidance(BaseModel):
    """Read-only evidence for contextual choices; never an availability gate."""

    as_of_utc: datetime = Field(description="Runtime world time of this projection, as naive UTC.")
    companion_name: str
    supported_capabilities: list[str] = Field(default_factory=list)
    partner: EntitySituationEvidence | None = None
    intentions: list[SituationSource] = Field(default_factory=list[SituationSource])
    recent_moves: list[RecentMove] = Field(default_factory=list[RecentMove])

    def source_revisions(self) -> dict[int, int]:
        """Return every source dependency, including the partner identity anchor."""
        sources = list(self.intentions)
        if self.partner is not None:
            sources.extend(self.partner.context)
            sources.extend(self.partner.recent_dialogue)
        for move in self.recent_moves:
            sources.append(move.action)
            if move.utterance is not None:
                sources.append(move.utterance)
        revisions = {source.record_id: source.revision for source in sources}
        if self.partner is not None:
            revisions[self.partner.entity_id] = self.partner.revision
        return revisions


class _DispatchPartner(BaseModel):
    """Read only the target carried by an existing durable action payload."""

    model_config = ConfigDict(extra="ignore")
    partner_entity_id: int | None = None


class _MoveMetadata(BaseModel):
    """Typed boundary for the existing open action and utterance metadata."""

    model_config = ConfigDict(extra="ignore")
    group: ShardKey | None = None
    dispatch_payload: _DispatchPartner = Field(default_factory=_DispatchPartner)
    public_reply_id: int | None = None
    execution_status: str = ""
    dispatch_status: str = ""
    publication_status: str = ""


def _source(view: Snapshot, record: RecordBase) -> SituationSource:
    """Preserve source text and distinguish reports from interpretations."""
    source = record.provenance.source_kind.value if record.provenance is not None else "unknown origin"
    kind = type(record).__name__
    if isinstance(record, Feature):
        source = record.source
        kind = record.feature_type.value
    elif isinstance(record, Narrative):
        kind = f"interpretive narrative ({record.narrative_type.value})"
    elif isinstance(record, BeliefRecord):
        source = record.source
        kind = f"belief ({record.belief_kind}; confidence {record.confidence:.2f})"
    return SituationSource(record_id=record.id, revision=view.revisions[record.id],
                           occurred_at_utc=record.timestamp_world_begin, source=source,
                           kind=kind, text=record.content, scope=record.scope)


def _dialogue_partner(record: Feature, entities: dict[str, int], companion_id: int | None,
                      user_id: int | None) -> int | None:
    """Apply the existing dialogue listener convention without reading other shards."""
    speaker_id = record.entity_id or entities.get(record.source)
    is_companion = companion_id is not None and speaker_id == companion_id
    if is_companion or record.provenance is not None and record.provenance.source_kind == SourceKind.Self:
        return record.addressee_entity_id or user_id
    return speaker_id


def build_situational_guidance(view: Snapshot, *, as_of_utc: datetime, companion_name: str,
                               user_name: str, partner_entity_id: int | None = None,
                               supported_capabilities: Sequence[str] = (),
                               scopes: Sequence[RecordScope] = (RecordScope(),)) -> SituationalGuidance:
    """Read one partner's evidence without retrieval, inference, or mutation.

    Self and workflow choices use the configured user, matching the existing
    action-request target convention. Entity choices pass their exact partner.
    New stimuli remain in their existing prompt section, not a second event
    list here. Sparse identity data remains unknown rather than invented.
    """
    entities: dict[str, int] = {}
    for record in view.all_knoxels.values():
        if isinstance(record, Entity):
            for name in [record.content, *record.aliases]:
                entities[name] = record.id
    companion_id = entities.get(companion_name)
    user_id = entities.get(user_name)
    target_id = partner_entity_id if partner_entity_id is not None else user_id
    partner: EntitySituationEvidence | None = None
    if target_id is not None:
        identity = view.get(target_id)
        if not isinstance(identity, Entity):
            raise TypeError(f"Situation partner {target_id} is not an Entity.")
        beliefs: list[BeliefRecord] = []
        narratives: dict[NarrativeTypes, Narrative] = {}
        dialogue: list[Feature] = []
        for record in view.all_knoxels.values():
            if record.scope not in scopes:
                continue
            if isinstance(record, Narrative) and record.entity_id == target_id:
                previous = narratives.get(record.narrative_type)
                rank = (record.last_refined_with_tick or -1, record.timestamp_world_begin, record.id)
                if previous is None or rank > (previous.last_refined_with_tick or -1,
                                                previous.timestamp_world_begin, previous.id):
                    narratives[record.narrative_type] = record
            elif isinstance(record, BeliefRecord) and record.active and target_id in record.about_entity_ids:
                beliefs.append(record)
            elif isinstance(record, Feature) and record.feature_type == FeatureType.Dialogue:
                public_origin = record.provenance is not None and record.provenance.source_kind in {
                    SourceKind.User, SourceKind.Agent, SourceKind.Self}
                if (public_origin and record.causal and record.interlocus == InterlocusType.Public
                        and record.scope.frame == FrameEnum.InWorld
                        and _dialogue_partner(record, entities, companion_id, user_id) == target_id):
                    dialogue.append(record)
        # Identity and relationships already have narratives. Select their
        # newest versions rather than letting archived refinements compete.
        beliefs.sort(key=lambda record: (record.timestamp_world_begin, record.id))
        context: list[RecordBase] = []
        for category in (NarrativeTypes.Relations, NarrativeTypes.PsychologicalAnalysis):
            if category in narratives:
                context.append(narratives[category])
        context.extend(beliefs[-2:])
        dialogue.sort(key=lambda record: (record.timestamp_world_begin, record.id))
        partner = EntitySituationEvidence(entity_id=identity.id, revision=view.revisions[identity.id], name=identity.content,
            entity_class=identity.entity_class, context=[_source(view, record) for record in context[-4:]],
            recent_dialogue=[_source(view, record) for record in dialogue[-8:]])
    intentions = [record for record in view.all_knoxels.values()
                  if isinstance(record, Intention) and record.active and record.status == "active"
                  and record.scope in scopes]
    intentions.sort(key=lambda record: (-record.incentive_salience, record.id))
    actions: list[ActionSchemaRecord] = []
    for record in view.all_knoxels.values():
        if not isinstance(record, ActionSchemaRecord) or record.scope not in scopes:
            continue
        metadata = _MoveMetadata.model_validate(record.metadata)
        action_partner = metadata.dispatch_payload.partner_entity_id
        if action_partner is None:
            action_partner = int(metadata.group.ref) if metadata.group is not None and metadata.group.kind == "entity" else user_id
        if action_partner == target_id:
            actions.append(record)
    actions.sort(key=lambda record: (record.timestamp_world_begin, record.id))
    moves: list[RecentMove] = []
    for action in actions[-4:]:
        metadata = _MoveMetadata.model_validate(action.metadata)
        utterance: SituationSource | None = None
        status: Literal["unknown", "selected", "committed", "delivered", "failed"] = "unknown"
        if metadata.dispatch_status in {"pending", "admitted"} or action.status == ActionSchemaStatus.Selected:
            status = "selected"
        if metadata.execution_status in {"failed", "cancelled", "rejected"}:
            status = "failed"
        if metadata.public_reply_id is not None:
            reply = view.get(metadata.public_reply_id)
            if not isinstance(reply, Feature) or reply.feature_type != FeatureType.Dialogue:
                raise TypeError(f"Action {action.id} public reply is not dialogue.")
            if reply.scope not in scopes or reply.interlocus != InterlocusType.Public:
                raise ValueError(f"Action {action.id} public reply contradicts its interaction scope.")
            utterance = _source(view, reply)
            delivery = _MoveMetadata.model_validate(reply.metadata)
            status = "delivered" if delivery.publication_status == "published" else "committed"
        moves.append(RecentMove(action=_source(view, action), utterance=utterance, status=status))
    return SituationalGuidance(as_of_utc=as_of_utc, companion_name=companion_name,
        supported_capabilities=list(supported_capabilities), partner=partner,
        intentions=[_source(view, record) for record in intentions[:8]], recent_moves=moves)


def _render_source(source: SituationSource) -> str:
    """Render provenance alongside the original text rather than a new claim."""
    return (f"[record {source.record_id} rev {source.revision}; {source.occurred_at_utc.isoformat()} UTC; "
            f"{source.kind}; source {source.source}; reality {source.scope.reality_id}/{source.scope.frame.value}] "
            f"{source.text}")


def render_situational_guidance(guidance: SituationalGuidance) -> str:
    """Present evidence for generic situational reasoning without deciding an action."""
    lines = ["Current situational grounding", f"Runtime world time: {guidance.as_of_utc.isoformat()} UTC.",
        f"Configured character: {guidance.companion_name}. Actual available capabilities: "
        + ("; ".join(guidance.supported_capabilities) or "not supplied in this projection."),
        "Infer whether this partner can receive and respond from current evidence, keeping uncertainty explicit. "
        "Presence, activity, interruptibility, and willingness can differ. A statement is attributed evidence, "
        "not proof of the inferred state. Silence and elapsed time do not prove a new state; a timer is only an occasion to reconsider.",
        "Compare any interaction with current purposes and moves already made. Choose among interaction, private activity, "
        "and waiting as this character would; this context sets no automatic permission or silence rule."]
    partner = guidance.partner
    if partner is None:
        lines.append("Partner identity and current ability to respond are unknown.")
    else:
        lines.append(f"Partner: {partner.name}, entity {partner.entity_id}, {partner.entity_class.value}.")
        lines.append("Partner context (up to four latest accepted interpretations; direct new evidence can correct them):")
        lines.extend(_render_source(source) for source in partner.context)
        lines.append("Recent interaction (up to eight actual lines, oldest first; not a complete history):")
        lines.extend(_render_source(source) for source in partner.recent_dialogue)
        if not partner.recent_dialogue:
            lines.append("No recent partner interaction is available; current receptivity remains unknown.")
    lines.append("Accepted active intentions (up to eight, highest salience first):")
    lines.extend(_render_source(source) for source in guidance.intentions)
    lines.append("Recent selected purposes and observed delivery (up to four, oldest first):")
    for move in guidance.recent_moves:
        lines.append(f"{move.status}: {_render_source(move.action)}")
        if move.utterance is not None:
            lines.append("Actual utterance: " + _render_source(move.utterance))
    lines.append("A selected purpose is not an executed effect; delivered text does not establish that the partner read it or can respond.")
    return "\n".join(lines)
