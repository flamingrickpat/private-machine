"""Classify incoming partner input before full cognition and public speech.

This is the canonical overview of early triage. ``InputTriage`` owns one
lightweight kernel invocation per mailbox batch. It reads the accepted
workspace, the relevant partners' recent dialogue, character identity, and
current situational guidance. It reuses main cognition's contextual triage
prompt; it makes no retrieval, embedding, appraisal, or action-selection call.

With its entity owner injected, the same model call also proposes bounded,
attributed entity claims. The host hydrates original evidence and the owner
stages accepted claims beside the classification in one recovery-preserved
batch. A stale immediate proposal holds classification acceptance until fresh
extraction succeeds. Entity claims remain estimates, not speech permissions.

``CognitiveControl`` accepts the assessment into the memory store. Unclassified
input temporarily holds publication; a Critical assessment requires an accepted
main cycle containing its sources. Main cognition consumes that assessment
instead of classifying those inputs again. The publication boundary, rather
than cooperative cancellation, prevents an obsolete draft from becoming speech.

The module runs independently from the main module. Both retain foreground
admission, so a main cycle waiting for classification cannot block its model
call. Provider/model-format failures retain diagnostics and retry with visible
status and cancellation. Internal errors propagate. Restart derives pending
work from durable input and assessment records; no model result is fabricated.

This first boundary covers ``user.input``, including attributed peer input.
Other stimulus kinds retain their existing full-cycle triage.
"""
from __future__ import annotations

import json
import traceback
from typing import TYPE_CHECKING

from pydantic import JsonValue

from pm import vclock
from pm.bus.envelopes import Envelope, new_envelope
from pm.constants import CSM_RECENT_DIALOGUE_TURNS, CYCLE_RETRY_DELAY_SECONDS, CYCLE_RETRY_MAX_DELAY_SECONDS, WORKSPACE_TAIL_TOKENS
from pm.context_render.story import record_view
from pm.llm.http_client import ModelError
from pm.llm.structured import ModelFormatError
from pm.memory.bash_view import MemoryBashEnvironment
from pm.memory.store import MemoryStore
from pm.llm.concurrency import InferenceRole, inference_role
from pm.memory.snapshot import GhostConflict, Snapshot
from pm.records.enums import FeatureType
from pm.records.copying import pickle_copy
from pm.records.memory_records import Entity, Feature
from pm.records.workspace import SELF_SHARD, ShardKey
from pm.runs.workflow_module import CognitiveWorkflow
from workflows.input_triage.schemas import EarlyTriageAssessment, InputSourcePayload, InputTriageRequest, TriageEvidence
from workflows.main_cognition.cycle_commits import active_intentions
from workflows.main_cognition.cycle_work import CycleRequest
from workflows.main_cognition.host import dialogue_partner
from workflows.main_cognition.schemas import ShardStream, Stimulus
from pm.memory.dependencies import source_fingerprint
from workflows.public_speech.situational_guidance import build_situational_guidance
from workflows.entity_context.owner import EntityContextOwner
from workflows.input_triage.entity_updates import capture_entity_extraction
from workflows.main_cognition.entity_shards import add_entity_context_records, reserve_entity_context

if TYPE_CHECKING:
    from workflows.cognitive_control.host import CognitiveControl


class InputTriage(CognitiveWorkflow):
    """Own early classification, leaving state acceptance to cognitive control."""

    DATA_TOPICS = frozenset({"user.input"})
    WORKFLOW_NAME = "input_triage"

    cognitive_control: CognitiveControl
    entity_context_owner: EntityContextOwner | None = None

    def deliver(self, envelope: Envelope) -> None:
        """Register replayed input before queueing its classification."""
        if envelope.kind != "control":
            self.cognitive_control.register_input(InputSourcePayload.model_validate(envelope.payload).knoxel_id)
        super().deliver(envelope)

    def handle_batch(self, batch: list[Envelope]) -> None:
        """Retry external/model-format failures while preserving input identity."""
        remaining = [item for item in batch
                     if not self.cognitive_control.has_assessment(InputSourcePayload.model_validate(item.payload).knoxel_id)]
        if not remaining:
            return
        delay = CYCLE_RETRY_DELAY_SECONDS
        while not self.stop_event.is_set():
            try:
                with inference_role(InferenceRole.Triage):
                    super().handle_batch(remaining)
                return
            except (ModelError, ModelFormatError, GhostConflict) as error:
                diagnostic = {"error_type": type(error).__name__, "reason": str(error),
                              "traceback": traceback.format_exc()}
                path = self.runtime_root / "kernel" / str(self.invocation_id) / "triage-rejected.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")
                self.bus.put(new_envelope(self.module_id, "telemetry", "triage.rejected", "",
                    {"reason": str(error), "artifact_path": str(path), "retry_delay_seconds": delay}))
                if self.stop_event.wait(delay):
                    return
                delay = min(delay * 2, CYCLE_RETRY_MAX_DELAY_SECONDS)

    def prepare_input(self, batch: list[Envelope]) -> dict[str, JsonValue]:
        """Freeze identity, accepted guidance, and only the relevant dialogue."""
        assert self.ghost is not None, "Early triage requires its injected memory store."
        assert isinstance(self.last_environment, MemoryBashEnvironment), "Early triage requires a frozen memory environment."
        view = self.last_environment.ghost_view
        assert view is not None, "Early triage requires a frozen memory snapshot."
        request = build_input_request(self.ghost, view, batch, run_id=self.invocation_id or "")
        if self.entity_context_owner is not None:
            observer_id = request.cycle.self_entity_id
            if observer_id is None:
                raise ValueError("Immediate entity maintenance requires the configured companion identity.")
            request.entity_extraction_input = capture_entity_extraction(self.entity_context_owner, view,
                subject_ids=[item.entity_id for item in request.cycle.stimuli if item.entity_id is not None],
                observer_entity_id=observer_id, source_ids=[item.knoxel_id for item in request.cycle.stimuli],
                evidence_ids=list(request.evidence.record_fingerprints), as_of_utc=request.cycle.cycle_time,
                token_budget=WORKSPACE_TAIL_TOKENS)
        return request.model_dump(mode="json")

    def publish_output(self, report: dict[str, JsonValue], batch: list[Envelope]) -> None:
        """Accept the durable assessment before waking cognition and speech."""
        assessment = EarlyTriageAssessment.model_validate(report)
        accepted = self.cognitive_control.accept_triage(assessment)
        self.bus.put(new_envelope(self.module_id, "telemetry", "triage.accepted", "",
            {"source_ids": [source_id for source_id in accepted.source_ids],
             "reconsideration_epoch": accepted.reconsideration_epoch,
             "guidance_tick": accepted.guidance_tick,
             "result": accepted.result.model_dump(mode="json")}))

def build_input_request(ghost: MemoryStore, view: Snapshot, batch: list[Envelope], *,
                        run_id: str) -> InputTriageRequest:
    """Freeze attributed input and accepted guidance for triage or ordinary selection."""
    with ghost.lock:
        companion = ghost.get_companion_name()
        user = ghost.get_user_name()
        entities = [record for record in view.all_knoxels.values() if isinstance(record, Entity)]
        names = {name: entity.id for entity in entities for name in [entity.content, *entity.aliases]}
        user_id = names.get(user)
        self_id = names.get(companion)
        previous = next((state for state in reversed(ghost.states)
                         if state.tick_id in view.gwt_turns and view.gwt_turns[state.tick_id].slot_id == "main"), None)
        previous = pickle_copy(previous) if previous is not None else None
        sources = [view.all_knoxels[InputSourcePayload.model_validate(item.payload).knoxel_id] for item in batch]
        stimuli = [Stimulus(knoxel_id=source.id, source="user_input", text=source.content,
            correlation_id=envelope.correlation_id,
            entity_id=dialogue_partner(source, names, user_id, self_id)
                      if isinstance(source, Feature) else user_id)
            for source, envelope in zip(sources, batch, strict=True)]
        partner_ids = {item.entity_id for item in stimuli if item.entity_id is not None}
        source_scopes = tuple(source.scope for source in sources)
        partner_scopes = {partner_id: tuple(source.scope for stimulus, source in zip(stimuli, sources, strict=True)
                                            if stimulus.entity_id == partner_id)
                          for partner_id in partner_ids}
        keys = [SELF_SHARD, *(ShardKey(kind="entity", ref=str(key)) for key in sorted(partner_ids))]
        streams: list[ShardStream] = []
        records = {source.id: record_view(source) for source in sources}
        if previous is not None:
            for shard in previous.workspace.all_shards():
                scopes = partner_scopes[int(shard.key.ref)] if shard.key.kind == "entity" and int(shard.key.ref) in partner_scopes else source_scopes
                shard.items = [item for item in shard.items if view.get(item.record_id).scope in scopes]
                shard.tail_ids = [record_id for record_id in shard.tail_ids if view.get(record_id).scope in scopes]
            for record_id in [*previous.workspace.item_ids(), *previous.workspace.tail_ids()]:
                records[record_id] = record_view(view.get(record_id))
        dialogue = sorted((record for record in view.all_knoxels.values()
                           if isinstance(record, Feature) and record.feature_type == FeatureType.Dialogue),
                          key=lambda record: (record.timestamp_world_begin, record.id))
        for key in keys:
            if key.kind != "entity":
                continue
            selected = [record for record in dialogue
                        if dialogue_partner(record, names, user_id, self_id) == int(key.ref)
                        and record.scope in partner_scopes[int(key.ref)]]
            selected = selected[-CSM_RECENT_DIALOGUE_TURNS:]
            streams.append(ShardStream(key=key, record_ids=[record.id for record in selected]))
            for record in selected:
                records[record.id] = record_view(record)
        for record in records.values():
            record.embedding = []  # Triage renders text and performs no similarity calculation.
        now = vclock.utcnow()
        guidance_partners: list[int | None] = list(sorted(partner_ids)) if partner_ids else [user_id]
        guidance = [build_situational_guidance(
            view, as_of_utc=now, companion_name=companion, user_name=user,
            partner_entity_id=partner_id,
            supported_capabilities=ghost.ghost_config.supported_capabilities,
            scopes=partner_scopes[partner_id] if partner_id in partner_scopes else source_scopes)
            for partner_id in guidance_partners]
        goals = [item for item in active_intentions(view) if item.scope in source_scopes][:8]
        add_entity_context_records(view, records, guidance)
        if previous is not None:
            reserve_entity_context(previous.workspace, records, guidance, tick=previous.tick_id)
        cycle = CycleRequest(
            run_id=run_id, correlation_id=batch[-1].correlation_id, companion_name=companion, user_name=user,
            user_entity_id=user_id, self_entity_id=self_id,
            entity_names={key: view.get(key).content for key in partner_ids},
            previous_state=previous, stimuli=stimuli, records=records, streams=streams,
            character_card=ghost.ghost_config.character_card_story,
            state_summary=previous.broadcasts[-1].summary if previous and previous.broadcasts else "",
            active_goals=[item.content for item in goals], cycle_time=now,
            situational_guidance=guidance,
        )
        evidence_ids = set(records)
        evidence_ids.update(item.id for item in goals)
        for item in guidance:
            evidence_ids.update(item.source_revisions())
        request = InputTriageRequest(cycle=cycle,
            source_fingerprints={source.id: source_fingerprint(source) for source in sources},
            guidance_tick=previous.tick_id if previous is not None else None,
            evidence=TriageEvidence(record_fingerprints={key: source_fingerprint(view.get(key))
                                                        for key in evidence_ids},
                                    commit_ids=sorted(view.commit_ids),
                                    current_head_ids=sorted(evidence_ids.intersection(view.all_knoxels))))
    return request
