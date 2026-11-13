"""Parent role of the private thought loop.

Entry point: ``run(context)``. Reads the request and writes the typed parent
report. It owns one routing rule and no cognitive policy: a cycle with no
retained linear episode goes to ``run_private_inquiry``, so the inquiry graph
decides what private work is worth doing. A retained linear episode, which
predates the graph, finishes on the trigger/thought/integrate pipeline through
``run_stage``. See ``workflows/inner_thought/host.py`` for the loop overview.
"""
from __future__ import annotations

from typing import Any

from pm_workflows.python_role import RoleContext

from workflows.inner_thought.pipeline import run_cycle
from workflows.inner_thought.schemas import InnerReport, InnerRequest
from workflows.main_cognition.model_json import read_request, write_report


def run(context: RoleContext) -> dict[str, Any]:
    """Run one private cycle and write its typed parent report."""
    request = InnerRequest.model_validate(read_request(context.runtime.environment))
    # Every fresh private opportunity goes through the inquiry graph. Only a
    # retained linear episode, which predates the graph, finishes on the old
    # trigger/thought/integrate pipeline.
    if request.episode.state is None:
        from workflows.inner_thought.graph_execution import PrivateGraphResult
        graph_report = PrivateGraphResult.model_validate(
            context.runtime.tools["run_private_inquiry"](request.model_dump(mode="json")))
        write_report(context.runtime.environment, graph_report)
        return {"status": "done", "summary": graph_report.summary}
    report = run_cycle(request, context.runtime.tools["run_stage"],
                       context.runtime.tools.get("retrieve_memory"))
    write_report(context.runtime.environment, report)
    return {"status": "done", "summary": report.summary}


__all__ = ["run"]
