"""Shared managed-run operations for Python roles, MCP, and the steward.

One function set serves the SelfManagement workflow, the Assistant's
tool-calling session, and the steward. Each function returns a plain mapping
so it can cross both a Python call boundary and a JSON tool boundary.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping

from pm.runs.supervisor import ManagedRunError, ManagedRunSupervisor


def build_management_tools(supervisor: ManagedRunSupervisor) -> dict[str, Callable[..., Any]]:
    """Return the scoped managed-run operations bound to one supervisor."""

    def managed_start(
        action_id: str,
        workflow_name: str,
        instructions: str = "",
        parent_correlation: str = "",
        artifact_path: str = "",
        arguments: dict | None = None,
    ) -> dict:
        """Start one managed run. A repeated action ID returns the first run."""
        return supervisor.start(
            action_id=action_id,
            workflow_name=workflow_name,
            parent_correlation=parent_correlation,
            instructions=instructions,
            artifact_path=artifact_path,
            arguments=arguments,
        )

    def managed_status(run_id: str) -> dict:
        """Return the latest durable receipt for one run."""
        return supervisor.status(run_id)

    def managed_list() -> list[dict]:
        """Return every receipt in the one managed-run table."""
        return list(supervisor.receipts().values())

    def managed_update(run_id: str, instructions: str, request_id: str = "") -> dict:
        """Queue one instruction revision at the next safe boundary."""
        return supervisor.update(run_id, instructions, request_id=request_id)

    def managed_terminate(run_id: str, request_id: str = "") -> dict:
        """Set the run's code-owned cancellation signal. Files are untouched."""
        return supervisor.terminate(run_id, request_id=request_id)

    def managed_answer(run_id: str, question_id: str, answer: str) -> dict:
        """Deliver a correlated answer to a waiting run and resume it."""
        return supervisor.answer(run_id, question_id, answer)

    def managed_resume(run_id: str, request_id: str = "") -> dict:
        """Resume retained body work after cleanup and fresh observation."""
        return supervisor.resume(run_id, request_id=request_id)

    return {
        "managed_start": managed_start,
        "managed_status": managed_status,
        "managed_list": managed_list,
        "managed_update": managed_update,
        "managed_terminate": managed_terminate,
        "managed_answer": managed_answer,
        "managed_resume": managed_resume,
    }


def management_mcp(supervisor: ManagedRunSupervisor):
    """Register the same operations on a FastMCP server for one run."""
    from fastmcp import FastMCP

    server = FastMCP("workflow_management")
    for fn in build_management_tools(supervisor).values():
        server.add_tool(fn)
    return server


__all__ = ["build_management_tools", "management_mcp", "ManagedRunError"]
