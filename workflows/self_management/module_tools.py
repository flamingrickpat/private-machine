"""Module-level tool factory for the SelfManagement workflow.

The application supplies one managed-run supervisor and one Ghost. The read
tools resolve the supervisor's durable table; the ghost tools use a scoped
view. The same function set is registered on the Assistant MCP server, so
Python steps and MCP clients share one implementation.
"""
from __future__ import annotations

from typing import Callable

from pm.runs.management_api import build_management_tools


def build_self_management_tools(supervisor) -> Callable[[object, object], dict]:
    """Return ``tools_factory(environment, batch)`` for SelfManagement."""

    def factory(environment, batch) -> dict:
        return build_management_tools(supervisor)

    return factory


__all__ = ["build_self_management_tools"]
