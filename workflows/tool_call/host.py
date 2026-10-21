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

from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any

from pm.llm.mcp_config import ToolCallConfig
from pm.memory.store import MemoryStore
from pm.records.media import alt_text_view
from pm.records.memory_records import ActionSchemaRecord
from pm.runs.workflow_spec import WorkflowSpec

from workflows.main_cognition.schemas import ActionRequest, PendingDispatch
from workflows.public_speech.shard_context import shard_story
from workflows.tool_call.report import reply_evidence, retain_interrupted_report
from workflows.tool_call.schemas import (
    ContextRecord, PinnedDispatch, SourceAction, SourceStimulus, ToolCallContext, TurnSummary,
)

WORKFLOW_NAME = "tool_call"

WORKFLOW_DESCRIPTION = (
    "Use the companion's own MCP tools for one concrete goal, in one continuous work session with every tool "
    "listed here. The decision text is the goal; arguments.tool_goal can add one short concrete goal. "
    "Returns the result, failures, artifacts, and uncertainty. Use ToolCallAndReply to speak about the result, "
    "ToolCall to use it privately. Tools available now:"
)


def _visible_request(request: ActionRequest, visible: set[int]) -> ActionRequest:
    """Return a copy of ``request`` whose shards keep only records that the store still holds."""
    copy = request.model_copy(deep=True)
    for item in copy.shards:
        item.shard.items = [entry for entry in item.shard.items if entry.record_id in visible]
        item.shard.tail_ids = [record_id for record_id in item.shard.tail_ids if record_id in visible]
    return copy


def capture_context(store: MemoryStore, arguments: Mapping[str, Any], *,
                    turn_limit: int) -> dict[str, Any]:
    """Return the ``tool_call_context`` request field for one launch.

    With a pinned source action, the context is that cycle's decision,
    stimuli, workspace shard story, feeling, and the GWT turns up to that
    cycle. Without one, it is the newest GWT turns at launch, labeled
    ``launch``. Image tags in the shard story become alt text, because the
    native session prompt is plain text. Records that the store no longer
    holds are listed as missing. The supervisor calls this under the store
    lock. It makes no commit, model call, or network request.
    """
    pinned = PinnedDispatch.model_validate(dict(arguments))
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
        source = SourceAction(
            action_schema_id=action.id, action_id=stored.action_id, tick_id=action.tick_id,
            correlation_id=stored.dispatch_correlation, decision=action.content,
            action_type=action.action_type.value, group=f"{request.group.kind}:{request.group.ref}",
            partner_name=request.partner_name,
            stimuli=[SourceStimulus(knoxel_id=item.knoxel_id, source=item.source, kind=item.kind, text=item.text)
                     for item in request.stimuli],
            shard_names=[item.name for item in request.shards],
            shard_story=alt_text_view(shard_story(view, _visible_request(request, visible), companion)),
            workspace=request.workspace, affect=request.affect)
    turns = sorted(store.gwt_turns.values(), key=lambda turn: turn.tick_id)
    if source is not None:
        turns = [turn for turn in turns if turn.tick_id <= source.tick_id]
    context = ToolCallContext(
        context_source="dispatch" if source is not None else "launch",
        captured_tick_id=store.current_tick_id, companion_name=companion, user_name=store.get_user_name(),
        character_card=store.get_simple_character_story_block(), source_action=source,
        gwt_turns=[TurnSummary(tick_id=turn.tick_id, summary=alt_text_view(turn.summary))
                   for turn in turns[-turn_limit:]],
        records=records, missing_record_ids=missing)
    return {"tool_call_context": context.model_dump(mode="json", by_alias=True), "tool_goal": pinned.tool_goal}


def build_workflow_spec(config: ToolCallConfig) -> WorkflowSpec:
    """Build the specification of the ``tool_call`` workflow.

    Every header reference is resolved once here, so a missing environment
    variable stops the application start with McpConfigError instead of
    failing later inside a session. The specification keeps the references,
    not the secrets; the session resolves them again. Automatic resume stays
    disabled, because an interrupted MCP call can leave external effects
    that a replay would repeat.
    """
    config.check_environment()
    selected = config.model_copy(deep=True)

    def tools_factory(run_id: str, environment: Any) -> dict[str, Any]:
        def tool_call_settings() -> ToolCallConfig:
            """Return the session configuration of this run."""
            return selected

        return {"tool_call_settings": tool_call_settings}

    return WorkflowSpec(
        manifest=Path(__file__).with_name("tool_call.workflow.md"),
        namespace=WORKFLOW_NAME,
        tools_factory=tools_factory,
        dispatch_context=partial(capture_context, turn_limit=selected.gwt_turn_limit),
        on_interrupted=retain_interrupted_report,
        reply_evidence=reply_evidence,
        resumable=False,
    )


__all__ = ["WORKFLOW_DESCRIPTION", "WORKFLOW_NAME", "build_workflow_spec", "capture_context"]
