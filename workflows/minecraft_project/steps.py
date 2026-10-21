"""Kernel entry of the minecraft_project roles: route each phase to its module in ``planning/``."""
from __future__ import annotations

from pm_workflows.python_role import RoleContext

from workflows.minecraft_project.planning.run_context import request, store_for
from workflows.minecraft_project.planning.retention import retain
from workflows.minecraft_project.planning.plan import plan
from workflows.minecraft_project.planning.survey import survey
from workflows.minecraft_project.planning.review import review
from workflows.minecraft_project.planning.selection import select
from workflows.minecraft_project.planning.dispatch import act
from workflows.minecraft_project.planning.verify import verify
from workflows.minecraft_project.planning.integration import finish


def _run(context: RoleContext) -> dict:
    """Route one declared role and retain its explicit failure or feedback."""
    store = store_for(context)
    if context.phase == "replan":
        return plan(context, store)
    if context.phase == "review":
        result = review(context, store)
        if result["status"] == "accepted" and request(context).plan_only:
            result["status"] = "inspect"
        return result
    operation = {"retain": retain, "plan": plan, "survey": survey, "select": select, "act": act, "verify": verify, "finish": finish}[context.phase]
    result = operation(context, store)
    if result["status"] in {"replan", "waiting", "blocked"}:
        store.state["feedback"] = result["summary"]
        if result["status"] == "replan":
            store.state["require_leaf_change"] = True
        store.save()
    return result


def run(context: RoleContext) -> dict:
    """Run a role without the outer runner replaying it for a file lock.

    Local evidence commits retry only their file operation. Other persistent
    permission errors fail the project. Its original world receipts remain.
    """
    try:
        return _run(context)
    except PermissionError as error:
        raise RuntimeError(f"A11 file access failed in {context.phase}. Do not replay world work: {error}") from error

