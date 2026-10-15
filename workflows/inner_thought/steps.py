"""Parent role of the private thought loop.

Entry point: ``run(context)``. Reads the request, runs the private cycle
through the ``run_stage`` tool, and writes the typed parent report. The cycle
is pure orchestration; this role adds no policy of its own.
"""
from __future__ import annotations

from workflows.inner_thought.pipeline import run_cycle
from workflows.inner_thought.schemas import InnerReport, InnerRequest
from workflows.main_cognition.stage_support import read_request, write_report


def run(context) -> dict:
    request = InnerRequest.model_validate(read_request(context.runtime.environment))
    report = run_cycle(request, context.runtime.tools["run_stage"])
    write_report(context.runtime.environment, report)
    return {"status": "done", "summary": report.summary}


__all__ = ["run"]
