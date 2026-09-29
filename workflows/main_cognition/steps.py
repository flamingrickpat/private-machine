"""Parent main-cognition role.

Entry point: ``run(context)``. Calls the injected ``run_stage`` tool for each
ordered child and writes the typed parent report. It owns no model calls and
no prompts; the children do.
"""
from __future__ import annotations

from workflows.main_cognition.pipeline import run_cycle
from workflows.main_cognition.stage_support import read_request, write_report


def run(context) -> dict:
    request = read_request(context.runtime.environment)
    run_stage = context.runtime.tools["run_stage"]
    report = run_cycle(request, run_stage)
    write_report(context.runtime.environment, report)
    return {"status": "done", "summary": report.summary}


__all__ = ["run"]
