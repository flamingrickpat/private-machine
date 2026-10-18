"""Route phase: turn the chosen action intents into typed dispatches.

Quiet choices (Ignore, Idle, Sleep) make no dispatch. They stay in the
decision record. Each dispatch gets a durable action id. The same-cycle
ordering check and the success prerequisites then set its status. This phase
has no external effect. The host publishes the released dispatches after the
commit.

Step 8 of A19F moves every routing rule from the decide prompt into a
separate routing call here.
"""
from __future__ import annotations

import hashlib
import json

from workflows.main_cognition.cycle_work import CycleWork
from workflows.main_cognition.dispatch import resolve_prerequisites, resolve_same_cycle
from workflows.main_cognition.phases.decide import QUIET_ACTIONS
from workflows.main_cognition.schemas import ActionDispatch, ActionIntent


def stable_action_id(intent: ActionIntent, index: int) -> str:
    """Return a durable identity for one proposed action.

    The same workflow intent receives the same ID across cycles, so a repeated
    start request can be recognised and suppressed. An action without content
    gets its list position, because it has no stable identity.
    """
    content = intent.content.strip()
    if not content:
        return f"action-{index}"
    material = json.dumps({
        "workflow_name": intent.workflow_name,
        "action_type": intent.action_type,
        "content": content,
        "arguments": intent.arguments.payload(),
    }, sort_keys=True, default=str)
    return "action-" + hashlib.sha1(material.encode("utf-8")).hexdigest()[:12]


def action_dispatches(actions: list[ActionIntent]) -> list[ActionDispatch]:
    """Convert executable choices into dispatch proposals without external effects.

    Quiet choices remain in the decision record. They cannot invoke a body or
    public workflow, even when they carry a valid workflow name.
    """
    dispatches = []
    for index, intent in enumerate(actions):
        if intent.action_type in QUIET_ACTIONS:
            continue
        dispatches.append(ActionDispatch(
            action_id=intent.action_id or stable_action_id(intent, index),
            workflow_name=intent.workflow_name,
            action_type=intent.action_type,
            content=intent.content,
            arguments=intent.arguments.model_copy(deep=True),
            depends_on=list(intent.depends_on),
            requires_completed=list(intent.requires_completed),
            presentation=intent.presentation.model_copy(deep=True),
        ))
    return dispatches


def run(context, work: CycleWork) -> CycleWork:
    """Build and resolve the dispatches.

    No receipts of earlier actions reach the cycle yet, so every success
    prerequisite stays queued.
    """
    dispatches = action_dispatches(work.action_selection.actions)
    dispatches = resolve_same_cycle(dispatches)
    work.dispatches = resolve_prerequisites(dispatches, {})
    released = sum(1 for item in work.dispatches if item.status == "dispatched")
    work.receipt("route", f"released {released} of {len(work.dispatches)} dispatches")
    return work


__all__ = ["action_dispatches", "run", "stable_action_id"]
