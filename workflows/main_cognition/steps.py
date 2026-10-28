"""The Python role of every main cognition phase.

Entry point: ``run(context)``. The kernel calls it once for each phase of
``main_cognition.workflow.md``. ``context.phase`` selects the phase module.

- The intake phase validates ``inbox/request.yaml`` and creates the shared
  :class:`~workflows.main_cognition.cycle_work.CycleWork`.
- Every other phase validates ``CycleWork`` from the working target at
  entry, runs, and writes the returned object back.
- The commit phase also writes the final report to ``outbox/report.yaml``.

Phases call the model only through the run's ``llm_complete`` tool.
"""
from __future__ import annotations

from pm_workflows.python_role import RoleContext

from pm.utils import yaml_util

from workflows.main_cognition.cycle_work import CycleRequest, read_work, write_work
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


def run(context: RoleContext) -> dict[str, str]:
    """Run one phase and return its role result for the kernel."""
    environment = context.runtime.environment
    if context.phase == "intake":
        request = CycleRequest.model_validate(yaml_util.safe_load(environment.read_text("inbox/request.yaml")))
        work = intake.start(request)
    else:
        work = PHASES[context.phase].run(context, read_work(environment))
    write_work(environment, work)
    return {"status": "done", "summary": work.phases[-1].summary}


__all__ = ["PHASES", "run"]
