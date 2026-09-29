"""Module-level tool factory for the steward workflow."""

from __future__ import annotations

from typing import Callable

from runtime.steward import build_steward_tools


def build_steward_module_tools(steward) -> Callable[[object, object], dict]:
    """Return ``tools_factory(environment, batch)`` for the steward module."""

    def factory(environment, batch) -> dict:
        return build_steward_tools(steward)

    return factory


__all__ = ["build_steward_module_tools"]
