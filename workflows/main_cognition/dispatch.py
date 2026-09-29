"""Dependency resolution for typed action dispatches.

A dispatch that names ``depends_on`` action IDs is only released when every
dependency has a successful receipt. A rejected dependency produces a rejected
dispatch with a reason, never a fabricated completion.
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
    """Resolve dependencies between actions proposed in one selection.

    An action with no dependency is released as a scheduling intent. An
    action is released only when every dependency names an earlier action in
    this same list that was itself released. An unknown or rejected
    dependency rejects the action explicitly.
    """
    released: dict[str, bool] = {}
    resolved: list[ActionDispatch] = []
    for dispatch in dispatches:
        bad: list[str] = []
        for dependency in dispatch.depends_on:
            if dependency not in released:
                bad.append(f"{dependency}:unknown")
            elif not released[dependency]:
                bad.append(f"{dependency}:rejected")
        if bad:
            released[dispatch.action_id] = False
            resolved.append(dispatch.model_copy(update={
                "status": "rejected",
                "rejection_reason": "dependency not released: " + ", ".join(bad),
            }))
        else:
            released[dispatch.action_id] = True
            resolved.append(dispatch.model_copy(update={"status": "dispatched"}))
    return resolved


__all__ = ["resolve_dispatches", "resolve_same_cycle"]
