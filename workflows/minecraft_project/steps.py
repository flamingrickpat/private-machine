"""Kernel entry of the minecraft_project roles: route each phase to its module in ``planning/``."""
from __future__ import annotations
from enum import StrEnum
from typing import TypedDict
from workflows.minecraft_project.schemas import ProjectPhaseResult

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



class ProjectPhase(StrEnum):
    Retain = "retain"
    Plan = "plan"
    Replan = "replan"
    Survey = "survey"
    Review = "review"
    Select = "select"
    Act = "act"
    Verify = "verify"
    Finish = "finish"


class ProjectRoleWire(TypedDict):
    status: str
    summary: str

def _run(context: RoleContext) -> ProjectPhaseResult:
    """Route one declared role and retain its explicit failure or feedback."""
    store = store_for(context)
    phase = ProjectPhase(context.phase)
    if phase is ProjectPhase.Replan:
        return plan(context, store)
    if phase is ProjectPhase.Review:
        result = review(context, store)
        if result.status == "accepted" and request(context).plan_only:
            result.status = "inspect"
        return result
    match phase:
        case ProjectPhase.Retain:
            result = retain(context, store)
        case ProjectPhase.Plan:
            result = plan(context, store)
        case ProjectPhase.Survey:
            result = survey(context, store)
        case ProjectPhase.Select:
            result = select(context, store)
        case ProjectPhase.Act:
            result = act(context, store)
        case ProjectPhase.Verify:
            result = verify(context, store)
        case ProjectPhase.Finish:
            result = finish(context, store)
        case _:
            raise ValueError(f"Unrouted Minecraft project phase: {phase}")
    if result.status in {"replan", "waiting", "blocked"}:
        store.state["feedback"] = result.summary
        if result.status == "replan":
            store.state["require_leaf_change"] = True
        store.save()
    return result


def run(context: RoleContext) -> ProjectRoleWire:
    """Run a role without the outer runner replaying it for a file lock.

    Local evidence commits retry only their file operation. Other persistent
    permission errors fail the project. Its original world receipts remain.
    """
    try:
        result = _run(context)
        return ProjectRoleWire(status=result.status, summary=result.summary)
    except PermissionError as error:
        raise RuntimeError(f"A11 file access failed in {context.phase}. Do not replay world work: {error}") from error

