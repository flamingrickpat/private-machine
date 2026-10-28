"""Module-level tool factory for memory consolidation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from workflows.memory_consolidation.artifact_consolidation import build_consolidation_tools

if TYPE_CHECKING:
    from workflows.memory_consolidation.artifact_consolidation import Consolidator


def build_consolidation_module_tools(
    consolidator: "Consolidator",
) -> Callable[[object, object], dict[str, Callable[..., object]]]:
    """Return ``tools_factory(environment, batch)`` for consolidation."""

    def factory(environment: object, batch: object) -> dict[str, Callable[..., object]]:
        """Return the consolidation tools of one invocation."""
        return build_consolidation_tools(consolidator)

    return factory


__all__ = ["build_consolidation_module_tools"]
