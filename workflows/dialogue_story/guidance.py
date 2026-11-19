"""Anchor accepted main guidance within Story's actual dialogue history.

See :mod:`workflows.dialogue_story.host` for speech ownership. This projection
reads the exact accepted main state and original action records. It does not
consume intentions or invent a new decision for a conversational continuation.
The main snapshot frontier, rather than its later completion time, determines
where its private context belongs. Capabilities remain current in the system
prompt. The host retains ordinary action and delivery provenance separately.
"""
from __future__ import annotations

from pm.constants import CSM_RECENT_DIALOGUE_TURNS
from pm.memory.store import MemoryStore
from pm.memory.dependencies import capture_source
from pm.memory.retrieval import candidate
from pm.records.base import RecordBase
from pm.records.cognitive_records import Intention
from pm.records.enums import FeatureType, InterlocusType, SourceKind
from pm.records.memory_records import ActionSchemaRecord, Feature
from pm.records.workspace import SELF_SHARD, ShardKey
from workflows.dialogue_story.schemas import StoryGuidance
from workflows.main_cognition.affect import regulation_mode
from workflows.main_cognition.action_request import request_shards
from workflows.main_cognition.schemas import ActionRequest, PendingDispatch
from workflows.public_speech.context_planning import PlannedContext
from workflows.public_speech.public_context import DialogueTurn, EvidenceItem, feelings_in_words, order_dialogue
from workflows.public_speech.shard_context import shard_context


def accepted_story_guidance(ghost: MemoryStore, request: ActionRequest) -> StoryGuidance | None:
    """Project this reply's main frame, with only its still-valid speech choice.

    Task receipts and standalone situations keep their existing current
    directive. Runtime main/interaction replies require their actual GWT state;
    missing state or a missing snapshot frontier is an internal contract error.
    """
    if request.action_origin == "task_reply" or request.gwt_tick is None:
        return None
    with ghost.lock:
        view = ghost.snapshot()
        state = next(state for state in ghost.read_cycles() if state.tick_id == request.gwt_tick)
        frontier = state.interval_reviewed_through
        if frontier is None:
            raise ValueError("Story guidance requires the accepted main snapshot frontier.")
        action: ActionSchemaRecord | None = None
        historical_request: ActionRequest | None = None
        scopes = {view.get(item.knoxel_id).scope for item in request.stimuli}
        for dispatch in state.dispatches:
            if dispatch.workflow != "dialogue_story":
                continue
            if dispatch.action_id is None:
                raise ValueError("An accepted main dispatch requires its committed action id.")
            candidate = view.get(dispatch.action_id)
            if not isinstance(candidate, ActionSchemaRecord):
                raise TypeError("A main Story dispatch must name an ActionSchemaRecord.")
            metadata = PendingDispatch.model_validate(candidate.metadata)
            original = metadata.dispatch_payload
            if original.partner_entity_id != request.partner_entity_id:
                continue
            original_scopes = {view.get(item.knoxel_id).scope for item in original.stimuli}
            if scopes and not scopes.intersection(original_scopes or {candidate.scope}):
                continue
            if metadata.dispatch_status == "deferred":
                continue
            action = candidate
            historical_request = original
        if historical_request is None:
            partner = (ShardKey(kind="entity", ref=str(request.partner_entity_id))
                       if request.partner_entity_id is not None else None)
            labels: dict[ShardKey, str] = {}
            for key in (SELF_SHARD, request.group, partner):
                if key is None:
                    continue
                if key.kind == "self":
                    labels[key] = ghost.get_companion_name()
                elif key.kind == "entity":
                    labels[key] = view.get_entity_name(int(key.ref))
                else:
                    labels[key] = key.ref
            shards = request_shards(state.workspace, request.group, partner, labels)
            for item in shards:
                item.shard.items = [entry for entry in item.shard.items
                                    if (not scopes or view.get(entry.record_id).scope in scopes)
                                    and not _subjective_experience(view.get(entry.record_id))]
                item.shard.tail_ids = [key for key in item.shard.tail_ids
                                       if (not scopes or view.get(key).scope in scopes)
                                       and not _subjective_experience(view.get(key))]
            historical_request = request.model_copy(update={"shards": shards, "entity_contexts": [], "workspace": ""})
        planned = shard_context(view, historical_request, ghost.get_companion_name())
        goals: list[str] = []
        source_ids = {frontier}
        for item in historical_request.shards:
            source_ids.update(item.shard.item_ids())
            source_ids.update(item.shard.tail_ids)
        if action is not None:
            source_ids.add(action.id)
            for intention_id in action.intention_ids:
                intention = view.get(intention_id)
                if not isinstance(intention, Intention):
                    raise TypeError("A governing intention must name an Intention record.")
                goals.append(intention.content)
                source_ids.add(intention_id)
        return StoryGuidance(tick_id=state.tick_id, through_record_id=frontier,
            observed_at_utc=view.get(frontier).timestamp_world_begin,
            shard_story=planned.story, workspace=historical_request.workspace,
            feelings=feelings_in_words(state.mental_state),
            regulation=regulation_mode(state.mental_state, state.mental_state),
            goals=tuple(goals), selected_action=action.content if action is not None else "",
            sources=tuple(capture_source(view, record_id, current_head=False) for record_id in sorted(source_ids)))


def _subjective_experience(record: RecordBase) -> bool:
    """Recognize group-private experience that requires its original dispatch projection."""
    return isinstance(record, Feature) and record.feature_type == FeatureType.SubjectiveExperience


def retain_following_dialogue(ghost: MemoryStore, request: ActionRequest, planned: PlannedContext,
                             guidance: StoryGuidance) -> PlannedContext:
    """Retain the frame's first reply and recent speech independently of retrieval.

    The current stimulus is rendered separately. Earlier inputs and companion
    replies remain ordered history for the same partner and scope. Every added
    line is also evidence, so public source guards cover it before publication.
    """
    with ghost.lock:
        view = ghost.snapshot()
        current_ids = {item.knoxel_id for item in request.stimuli}
        scopes = {view.get(record_id).scope for record_id in current_ids}
        if not scopes:
            scopes = {view.get(guidance.through_record_id).scope}
        companion = ghost.get_companion_name()
        user_id = ghost.get_entity_id(ghost.get_user_name())
        partner_id = request.partner_entity_id if request.partner_entity_id is not None else user_id
        partner_name = request.partner_name or ghost.get_user_name()
        turns = list(planned.dialogue)
        evidence = list(planned.evidence)
        seen = {turn.knoxel_id for turn in turns}
        evidenced = {item.knoxel_id for item in evidence}
        following: list[Feature] = []
        for record in view.all_knoxels.values():
            if not isinstance(record, Feature) or record.id <= guidance.through_record_id:
                continue
            if (record.id in current_ids or record.feature_type != FeatureType.Dialogue
                    or not record.causal or record.interlocus != InterlocusType.Public
                    or record.scope not in scopes or record.provenance is None):
                continue
            source_kind = record.provenance.source_kind
            if source_kind == SourceKind.Self:
                belongs = record.source == companion and (record.addressee_entity_id or user_id) == partner_id
            elif source_kind in {SourceKind.User, SourceKind.Agent}:
                belongs = record.entity_id == partner_id if record.entity_id else record.source == partner_name
            else:
                belongs = False
            if not belongs:
                continue
            following.append(record)
        following_by_id = {record.id: record for record in following}
        following = [following_by_id[key] for key in sorted(following_by_id)]
        first_reply = next((record for record in following
                            if record.provenance is not None
                            and record.provenance.source_kind == SourceKind.Self), None)
        retained = following[-CSM_RECENT_DIALOGUE_TURNS:]
        if first_reply is not None and first_reply.id not in {record.id for record in retained}:
            retained.insert(0, first_reply)
        for record in retained:
            if record.id not in seen:
                turns.append(DialogueTurn(speaker=record.source, text=record.content, knoxel_id=record.id,
                    timestamp=record.timestamp_world_begin.isoformat()))
                seen.add(record.id)
            if record.id not in evidenced:
                item = candidate(view, record.id, record)
                item.lane = "conversation"
                evidence.append(item)
                evidenced.add(record.id)
        turns = order_dialogue(turns)
        selected = planned.model_copy(deep=True)
        selected.dialogue = turns
        selected.evidence = evidence
        return selected
