"""Stage an explicit partner task using the original exchange as its authority.

See :mod:`workflows.input_triage.host` for classification and dispatch ordering.
"""
from __future__ import annotations

from pm.memory.snapshot import Snapshot
from pm.memory.store import MemoryStore
from pm.memory.write_batch import WriteBatch
from pm.records.enums import ActionType, FeatureType, InterlocusType, SourceKind
from pm.records.lineage import RecordLineage
from pm.records.memory_records import ActionSchemaRecord, Feature
from pm.records.workspace import SELF_SHARD, ShardKey
from workflows.input_triage.schemas import EarlyTriageAssessment
from workflows.main_cognition.catalogue import WorkflowCatalogue
from workflows.main_cognition.schemas import ActionArguments, ActionRequest, RecordMetadataView, Stimulus
from workflows.public_speech.dependencies import latest_partner_input


def stage_workflow_requests(ghost: MemoryStore, view: Snapshot, batch: WriteBatch,
                            assessment: EarlyTriageAssessment,
                            catalogue: WorkflowCatalogue, *, allow_start: bool = True) -> dict[int, int]:  # pm-lint: allow PM004 -- action IDs indexed by source knoxel ID
    """Commit one dispatch per current, explicitly authorized source with its triage receipt.

    The caller holds the store lock and has fenced unknown or Critical input.
    The supervisor owns worker admission and resource conflicts.
    """
    admitted: dict[int, int] = {}  # pm-lint: allow PM004 -- action IDs indexed by source knoxel ID
    enabled = {option.name for option in catalogue.direct_request_options()}
    for item in assessment.result.items:
        task = item.workflow_task
        target = assessment.body_target
        if item.body_operation and target:
            if item.body_operation == "resume" and target.state != "cancelled":
                continue
            if item.body_operation == "stop" and target.state == "cancelled":
                continue
            name = "self_management"
            operation = "terminate" if item.body_operation in {"stop", "yield"} else item.body_operation
            arguments = ActionArguments(operation=operation, run_id=target.run_id, expected_epoch=target.execution_epoch,
                                        yield_body=item.body_operation == "yield")
        elif allow_start and task is not None and task.workflow_name in enabled:
            name = task.workflow_name
            arguments = ActionArguments(constraints=list(task.constraints))
        else:
            continue
        source = view.get(item.knoxel_id)
        if not isinstance(source, Feature) or source.feature_type != FeatureType.Dialogue:
            raise TypeError(f"Workflow request {item.knoxel_id} requires a dialogue input.")
        if (source.provenance is None or source.provenance.source_kind != SourceKind.User
                or source.interlocus != InterlocusType.Public or not source.causal):
            continue
        if RecordMetadataView.model_validate(source.metadata).exchange_status != "pending":
            continue
        partner_id = source.entity_id or ghost.get_entity_id(source.source)
        latest = latest_partner_input(view, partner_entity_id=partner_id, partner_name=source.source,
                                      scopes=[source.scope])
        if latest != source.id:
            continue
        correlation = str(source.metadata["exchange_correlation"])
        action_id = f"workflow-request:{source.id}"
        group = ShardKey(kind="entity", ref=str(partner_id)) if partner_id is not None else SELF_SHARD
        arguments.instructions = source.content
        request = ActionRequest(
            action_id=action_id, workflow_name=name, content=source.content,
            arguments=arguments,
            group=group, partner_entity_id=partner_id, partner_name=source.source,
            stimuli=[Stimulus(knoxel_id=source.id, source="user_input", text=source.content,
                              entity_id=partner_id, correlation_id=correlation)],
            action_origin="input_triage",
        )
        action = batch.create(ActionSchemaRecord(
            content=source.content, action_type=ActionType.ToolCallAndReply, scope=source.scope,
            lineage=RecordLineage(episode_id=action_id, operation="workflow_request", source_event_ids=[source.id]),
            metadata={"action_id": action_id, "workflow_name": name,  # pm-lint: allow PM005 -- serialized action record metadata
                      "dispatch_target": name, "dispatch_status": "pending",
                      "dispatch_correlation": correlation, "evidence_ids": [source.id]},
        ))
        request.knoxel_id = action.id
        action.metadata["dispatch_payload"] = request.payload()
        admitted[source.id] = action.id
    return admitted  # pm-lint: allow PM005 -- action IDs indexed by source knoxel ID
