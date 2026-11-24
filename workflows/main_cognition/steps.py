"""Run one cognitive cycle on one in-memory typed state.

The kernel invokes this role once. Intake validates the external request;
existing phase functions then share the same CycleWork without intermediate
JSON files or reconstruction. Commit writes the final report for the host's
atomic memory acceptance. A cancelled or interrupted cycle has no resumable
phase state. The host.py overview describes memory and publication ownership.
"""
from __future__ import annotations

from pm.llm.role_model import role_model

from pm_workflows.python_role import RoleContext
from collections.abc import Sequence
from dataclasses import replace
import logging

from pm.utils import yaml_util

from workflows.main_cognition.cycle_work import CycleRequest, CycleWork
from workflows.main_cognition.phases import (
    appraise, attention, codelets, commit, decide, experience, goals, intake, retrieve, route, structures, triage,
)

#: Phase modules by manifest phase name, in manifest order.
PHASES = {
    "triage": triage,
    "appraise": appraise,
    "retrieve": retrieve,
    "codelets": codelets,
    "structures": structures,
    "attention": attention,
    "experience": experience,
    "goals": goals,
    "decide": decide,
    "route": route,
    "commit": commit,
}


def run_selected_phases(context: RoleContext, work: CycleWork, names: Sequence[str]) -> CycleWork:
    """Execute dependent phase functions in order on the same state object."""
    runtime = context.runtime
    assert runtime is not None
    for name in names:
        runtime.check_cancelled()
        logging.getLogger(__name__).info("Cognitive phase start run=%s phase=%s", context.run_id, name)
        work = PHASES[name].run(replace(context, phase=name), work)
        logging.getLogger(__name__).info("Cognitive phase end run=%s phase=%s", context.run_id, name)
    return work


def run(context: RoleContext) -> dict[str, str]:
    """Validate intake once and execute the whole cycle without state round trips."""
    runtime = context.runtime
    assert runtime is not None
    request = CycleRequest.model_validate(yaml_util.safe_load(runtime.environment.read_text("inbox/request.yaml")))
    work = intake.start(request)
    work.prepare_stimulus_summaries(role_model(runtime))
    work = run_selected_phases(context, work, list(PHASES))
    return {"status": "done", "summary": work.phases[-1].summary}


__all__ = ["PHASES", "run", "run_selected_phases"]
