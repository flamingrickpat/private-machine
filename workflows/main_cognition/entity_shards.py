"""Reserve compact accepted entity context in its existing workspace shard.

The main host owns acceptance. This adapter changes only the pending request's
record views and workspace references. The accepted receipt remains canonical.
See :mod:`workflows.main_cognition.host` for the subsystem overview.
"""
from __future__ import annotations

from collections.abc import Sequence

from pm.context_render.story import RecordView, record_tokens, record_view
from pm.memory.snapshot import Snapshot
from pm.records.workspace import GlobalWorkspace, ShardKey, WorkspaceItem
from workflows.entity_context.projection import render_entity_context
from workflows.public_speech.situational_guidance import SituationalGuidance


def add_entity_context_records(view: Snapshot, records: dict[int, RecordView],
                                guidance: Sequence[SituationalGuidance]) -> None:
    """Render the frozen compact view in place of an opaque receipt summary."""
    for situation in guidance:
        if situation.partner is None:
            continue
        for context in situation.partner.entity_contexts:
            head = context.head_receipt_id
            if head is None:
                continue
            record = record_view(view.get(head))
            record.content = render_entity_context(context)
            record.entity_ids = [context.key.subject_entity_id]
            record.category = "entity_context"
            records[head] = record


def reserve_entity_context(workspace: GlobalWorkspace, records: dict[int, RecordView],
                            guidance: Sequence[SituationalGuidance], *, tick: int) -> None:
    """Keep the actual receipt reference rather than copying its whole history."""
    for situation in guidance:
        if situation.partner is None:
            continue
        key = ShardKey(kind="entity", ref=str(situation.partner.entity_id))
        shard = workspace.ensure_shard(key)
        shard.items = [item for item in shard.items if item.required_by != "entity_context"]
        for context in situation.partner.entity_contexts:
            head = context.head_receipt_id
            if head is None:
                continue
            shard.items = [item for item in shard.items if item.record_id != head]
            shard.items.append(WorkspaceItem(record_id=head, entered_tick=tick, activation_at_entry=1.0,
                reason="Current accepted observer-specific entity context.", tokens=record_tokens(records, head),
                required_by="entity_context"))
