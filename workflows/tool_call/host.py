"""The managed-run binding of the tool_call workflow.

The assembly registers :func:`build_workflow_spec` under ``tool_call`` only
when the model configuration names at least one MCP server. Import and
specification construction open no connection and start no work.

:func:`capture_context` pins the session context to the cycle that chose the
action. Self-management stores the id of that action record in the run
arguments (``_public_action_schema_id``). The record holds the frozen action
request of that cycle: the decision, its stimuli, and its workspace shards.
A launch-time view of the store cannot name that cycle by itself.
"""
from __future__ import annotations

from pm.records.copying import pickle_copy
from pm.context_render.records import RecordEvidence

from collections.abc import Mapping
from dataclasses import dataclass
from pydantic import BaseModel, JsonValue
from pathlib import Path

from pm.llm.mcp_config import ToolCallConfig
from pm.memory.store import MemoryStore
from pm.records.media import alt_text_view, image_ids, metadata_media, RecordImage
from pm.records.memory_records import ActionSchemaRecord
from pm.runs.workflow_spec import WorkflowSpec

from workflows.main_cognition.schemas import ManagedRunView, PendingDispatch
from workflows.tool_call.report import retain_interrupted_report
from workflows.tool_call.schemas import (
    ContextRecord, ToolContextEvidence, PinnedDispatch, SourceAction, SourceStimulus, ToolCallContext, TurnSummary,
)

WORKFLOW_NAME = "tool_call"

WORKFLOW_DESCRIPTION = (
    "Use the companion's own MCP tools for one concrete goal, in one continuous work session with every tool "
    "listed here. The decision text is the goal; arguments.tool_goal can add one short concrete goal. "
    "Returns the result, failures, artifacts, and uncertainty. Use ToolCallAndReply to speak about the result, "
    "ToolCall to use it privately. Tools available now:"
)


@dataclass(frozen=True, slots=True)
class ToolContextSettings:
    turn_limit: int


class ToolCallDispatch(BaseModel):
    tool_call_context: ToolCallContext
    tool_goal: str


@dataclass(frozen=True, slots=True)
class ToolCallContextCapture:
    settings: ToolContextSettings

    def capture(self, store: MemoryStore, arguments: Mapping[str, JsonValue]) -> ToolCallDispatch:
        """Return the ``tool_call_context`` request field for one launch.

        With a pinned source action, the context is that cycle's decision,
        stimuli, workspace shard story, feeling, and the GWT turns up to that
        cycle. Without one, it is the newest GWT turns at launch, labeled
        ``launch``. Image tags in the shard story become alt text, because the
        native session prompt is plain text. Records that the store no longer
        holds are listed as missing. The supervisor calls this under the store
        lock. It makes no commit, model call, or network request.
        """
        pinned = PinnedDispatch.model_validate(arguments)
        companion = store.get_companion_name()
        source = None
        records: list[ContextRecord] = []
        missing: list[int] = []
        if pinned.action_schema_id is not None:
            action = store.get_knoxel_by_id(pinned.action_schema_id)
            if not isinstance(action, ActionSchemaRecord):
                raise ValueError(f"The pinned source action {pinned.action_schema_id} is not an action record.")
            stored = PendingDispatch.model_validate(action.metadata)
            request = stored.dispatch_payload
            ids = list(dict.fromkeys([*(item.knoxel_id for item in request.stimuli),
                                      *(record_id for item in request.shards
                                        for record_id in [*item.shard.item_ids(), *item.shard.tail_ids])]))
            visible = set(ids).intersection(store.revisions)
            missing = [record_id for record_id in ids if record_id not in visible]
            view = store.snapshot(visible)
            records = [ContextRecord(record_id=key, revision=view.revisions[key],
                                     record_class=type(view.get(key)).__name__) for key in sorted(visible)]
            weighted: dict[int, float] = {}
            tail_ids: set[int] = set()
            for shard in request.shards:
                for item in shard.shard.items:
                    weighted[item.record_id] = max(weighted.get(item.record_id, 0.0), item.activation_at_entry)
                tail_ids.update(shard.shard.tail_ids)
            evidence: list[ToolContextEvidence] = []
            stimulus_ids = {item.knoxel_id for item in request.stimuli}
            for record_id in sorted(visible - stimulus_ids):
                record = view.get(record_id)
                evidence.append(ToolContextEvidence(record=RecordEvidence(knoxel_id=record_id,
                    revision=view.revisions[record_id], tick_id=record.tick_id, text=record.content,
                    kind=type(record).__name__, provenance=record.provenance),
                    activation=weighted.get(record_id, 0.0), in_recent_tail=record_id in tail_ids))
            source = SourceAction(
                action_schema_id=action.id, action_id=stored.action_id, tick_id=action.tick_id,
                correlation_id=stored.dispatch_correlation, decision=action.content,
                action_type=action.action_type.value, group=f"{request.group.kind}:{request.group.ref}",
                partner_name=request.partner_name,
                stimuli=[SourceStimulus(knoxel_id=item.knoxel_id, source=item.source, kind=item.kind, text=item.text)
                         for item in request.stimuli],
                shard_names=[item.name for item in request.shards],
                evidence=tuple(evidence),
                workspace=request.workspace, affect=request.affect)
        turns = [store.gwt_turns[tick_id] for tick_id in sorted(store.gwt_turns)]
        if source is not None:
            turns = [turn for turn in turns if turn.tick_id <= source.tick_id]
        required_images: set[str] = set()
        if source is not None:
            for stimulus in source.stimuli:
                required_images.update(image_ids(stimulus.text))
            for entry in source.evidence:
                required_images.update(image_ids(entry.record.text))
            required_images.update(image_ids(source.workspace))
        images: dict[str, RecordImage] = {}
        for record in store.all_knoxels.values():
            for image in metadata_media(record.metadata).images:
                if image.id in required_images:
                    images[image.id] = pickle_copy(image)
        if required_images - images.keys():
            raise ValueError(f"Tool context references missing images: {sorted(required_images - images.keys())}")
        context = ToolCallContext(
            context_source="dispatch" if source is not None else "launch",
            captured_tick_id=store.current_tick_id, companion_name=companion, user_name=store.get_user_name(),
            character_card=store.get_simple_character_story_block(), source_action=source,
            gwt_turns=[TurnSummary(tick_id=turn.tick_id, summary=alt_text_view(turn.summary))
                       for turn in turns[-self.settings.turn_limit:]],
            records=records, missing_record_ids=missing, images=tuple(images[key] for key in sorted(images)))
        if pinned.related_run_id:
            run = ManagedRunView.model_validate(store.read_run_receipts()[pinned.related_run_id])
            context.related_run_id = pinned.related_run_id
            context.original_assignment = run.original_instructions or run.instructions or ""
            context.current_assignment = run.instructions or ""
        return ToolCallDispatch(tool_call_context=context, tool_goal=pinned.tool_goal)




def build_workflow_spec(config: ToolCallConfig, *, private: bool = False) -> WorkflowSpec:
    """Build the specification of the ``tool_call`` workflow.

    Every header reference is resolved once here, so a missing environment
    variable stops the application start with McpConfigError instead of
    failing later inside a session. The specification keeps the references,
    not the secrets; the session resolves them again. Automatic resume stays
    disabled, because an interrupted MCP call can leave external effects
    that a replay would repeat.
    """
    config.check_environment()
    selected = pickle_copy(config)

    return WorkflowSpec(
        manifest=(Path(__file__).parent.parent / "assistant/assistant.workflow.md" if private
                  else Path(__file__).with_name("tool_call.workflow.md")),
        namespace="assistant" if private else WORKFLOW_NAME,
        routine_start=private,
        tool_call_config=selected,
        dispatch_context_owner=ToolCallContextCapture(ToolContextSettings(selected.gwt_turn_limit)),
        on_interrupted=retain_interrupted_report,
        resumable=False,
    )


__all__ = ["WORKFLOW_DESCRIPTION", "WORKFLOW_NAME", "build_workflow_spec", "ToolCallContextCapture"]
