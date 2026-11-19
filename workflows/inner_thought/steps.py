"""Parent role of the private thought loop.

Entry point: ``run(context)``. Reads the request and writes the typed parent
report. It owns one routing rule and no cognitive policy: a cycle with no
retained linear episode goes to ``run_private_inquiry``, so the inquiry graph
decides what private work is worth doing. A retained linear episode, which
predates the graph, finishes on the trigger/thought/integrate pipeline through
``run_stage``. See ``workflows/inner_thought/host.py`` for the loop overview.
"""
from __future__ import annotations

from workflows.inner_thought.host import PrivateThoughtRuntime

from pm_workflows.python_role import RoleContext

from workflows.inner_thought.schemas import InnerRequest, PrivateRoleResult
from pm.runs.role_io import read_request, write_report


def run(context: RoleContext) -> PrivateRoleResult:
    """Run one private cycle and write its typed parent report."""
    assert isinstance(context.runtime, PrivateThoughtRuntime)
    invocation = context.runtime.private_invocation
    request = read_request(context.runtime.environment, InnerRequest)
    # Every fresh private opportunity goes through the inquiry graph. Only a
    # retained linear episode, which predates the graph, finishes on the old
    # trigger/thought/integrate pipeline.
    if request.episode.state is None:
        graph_report = invocation.run_private_inquiry(request)
        write_report(context.runtime.environment, graph_report)
        return {"status": "done", "summary": graph_report.summary}
    report = invocation.run_legacy_cycle(request)
    write_report(context.runtime.environment, report)
    return {"status": "done", "summary": report.summary}


__all__ = ["run"]
