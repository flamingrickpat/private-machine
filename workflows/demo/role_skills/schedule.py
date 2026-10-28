"""Dummy scheduler role: advance the in-memory task table and report due transitions.

The kernel loads this file by path. The behavior lives in
``workflows.demo.roles``; this module only forwards the context.
"""
from __future__ import annotations

from typing import Any

from pm_workflows.python_role import RoleContext

from workflows.demo.roles import schedule


def run(context: RoleContext) -> dict[str, Any]:
    """Run the demonstration role and return its kernel result mapping."""
    return schedule(context)
