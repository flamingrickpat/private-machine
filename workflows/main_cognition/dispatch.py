"""Dependency resolution for typed action dispatches.

Two dependency kinds are kept apart:

- ``depends_on`` only orders dispatch. An action may reach the bus after an
  earlier action in the same selection was released.
- ``requires_completed`` names a real prerequisite. The action stays queued
  until a ``completed`` receipt for that action exists. Missing, failed, or
  cancelled prerequisites keep it queued or reject it, never release it.

A rejected dependency produces a rejected dispatch with a reason, never a
fabricated completion.
"""
from __future__ import annotations

from workflows.main_cognition.schemas import ActionDispatch


def resolve_dispatches(
    dispatches: list[ActionDispatch],
    receipts: dict[str, str],
) -> list[ActionDispatch]:
    """Return dispatches with ``status`` set from the receipt table.

    ``receipts`` maps action ID to a terminal state. Only ``completed``
    releases a dependent dispatch. Missing, failed, or cancelled dependencies
    reject it with an explicit reason.
    """
    resolved: list[ActionDispatch] = []
    for dispatch in dispatches:
        missing: list[str] = []
        failed: list[str] = []
        for dependency in dispatch.depends_on:
            state = receipts.get(dependency)
            if state is None:
                missing.append(dependency)
            elif state != "completed":
                failed.append(f"{dependency}:{state}")
        if failed:
            resolved.append(dispatch.model_copy(update={
                "status": "rejected",
                "rejection_reason": "dependency failed: " + ", ".join(failed),
            }))
        elif missing:
            resolved.append(dispatch.model_copy(update={
                "status": "rejected",
                "rejection_reason": "dependency missing: " + ", ".join(missing),
            }))
        else:
            resolved.append(dispatch.model_copy(update={"status": "dispatched"}))
    return resolved


def resolve_same_cycle(dispatches: list[ActionDispatch]) -> list[ActionDispatch]:
    """Resolve dispatch ordering between actions proposed in one selection.

    An action with no ordering dependency is released. An action is released
    only when every ``depends_on`` names an earlier action ID or a unique
    earlier workflow name that was released. An unknown or ambiguous dependency
    rejects the action explicitly. This function does not confirm success.
    """
    released: dict[str, bool] = {}
    workflows: dict[str, list[str]] = {}
    resolved: list[ActionDispatch] = []
    for dispatch in dispatches:
        bad: list[str] = []
        dependencies: list[str] = []
        for dependency in dispatch.depends_on:
            # The model names workflows before the parent assigns durable IDs.
            # An explicit ID wins. Repeated workflow names need an explicit ID.
            if dependency not in released:
                matches = workflows.get(dependency, [])
                if len(matches) == 1:
                    dependency = matches[0]
                elif len(matches) > 1:
                    bad.append(f"{dependency}:ambiguous")
                    continue
            dependencies.append(dependency)
            if dependency not in released:
                bad.append(f"{dependency}:unknown")
            elif not released[dependency]:
                bad.append(f"{dependency}:rejected")
        if bad:
            released[dispatch.action_id] = False
            resolved.append(dispatch.model_copy(update={
                "status": "rejected",
                "rejection_reason": "dependency not released: " + ", ".join(bad),
                "depends_on": dependencies,
            }))
        else:
            released[dispatch.action_id] = True
            resolved.append(dispatch.model_copy(update={"status": "dispatched", "depends_on": dependencies}))
        workflows.setdefault(dispatch.workflow_name, []).append(dispatch.action_id)
    return resolved


def resolve_prerequisites(
    dispatches: list[ActionDispatch],
    receipts: dict[str, str],
) -> list[ActionDispatch]:
    """Gate success prerequisites without reordering dispatch.

    A dispatch that already failed ordering resolution keeps its rejection.
    Otherwise every ``requires_completed`` entry must have a ``completed``
    receipt. A known failure rejects the dispatch; a missing or unfinished
    receipt leaves it ``queued`` so a later receipt can release it.
    """
    resolved: list[ActionDispatch] = []
    for dispatch in dispatches:
        if dispatch.status == "rejected":
            resolved.append(dispatch)
            continue
        failed: list[str] = []
        pending: list[str] = []
        for prerequisite in dispatch.requires_completed:
            state = receipts.get(prerequisite)
            if state == "completed":
                continue
            if state is None:
                pending.append(prerequisite)
            else:
                failed.append(f"{prerequisite}:{state}")
        if failed:
            resolved.append(dispatch.model_copy(update={
                "status": "rejected",
                "rejection_reason": "prerequisite failed: " + ", ".join(failed),
            }))
        elif pending:
            resolved.append(dispatch.model_copy(update={
                "status": "queued",
                "rejection_reason": "prerequisite pending: " + ", ".join(pending),
            }))
        else:
            resolved.append(dispatch)
    return resolved


__all__ = ["resolve_dispatches", "resolve_prerequisites", "resolve_same_cycle"]
