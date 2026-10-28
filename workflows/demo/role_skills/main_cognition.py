"""Dummy main-cognition role: route user input by fixed keyword rules, with no model call.

The kernel loads this file by path. The behavior lives in
``workflows.demo.roles``; this module only forwards the context.
"""
from __future__ import annotations

from typing import Any

from pm_workflows.python_role import RoleContext

from workflows.demo.roles import main_cognition


def run(context: RoleContext) -> dict[str, Any]:
    """Run the demonstration role and return its kernel result mapping."""
    return main_cognition(context)
