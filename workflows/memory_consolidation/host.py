"""Module-level tool factory for memory consolidation."""

from __future__ import annotations

from typing import Callable

from workflows.memory_consolidation.artifact_consolidation import build_consolidation_tools


def build_consolidation_module_tools(consolidator) -> Callable[[object, object], dict]:
    """Return ``tools_factory(environment, batch)`` for consolidation."""

    def factory(environment, batch) -> dict:
        return build_consolidation_tools(consolidator)

    return factory


__all__ = ["build_consolidation_module_tools"]
