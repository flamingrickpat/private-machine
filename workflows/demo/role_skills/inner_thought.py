"""Dummy private-thought role: emit the fixed demonstration thought for the tick.

The kernel loads this file by path. The behavior lives in
``workflows.demo.roles``; this module only forwards the context.
"""
from __future__ import annotations

from typing import Any

from pm_workflows.python_role import RoleContext

from workflows.demo.roles import inner_thought


def run(context: RoleContext) -> dict[str, Any]:
    """Run the demonstration role and return its kernel result mapping."""
    return inner_thought(context)
