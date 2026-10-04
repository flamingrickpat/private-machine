"""Module-level tool factory for the real main-cognition parent.

The application supplies one lazily-activated completion callable. This module
binds it, the Ghost, and the runtime root into the ``run_stage`` tool the
parent Python role names in its manifest. Each ``run_stage`` call builds a
fresh read-only Ghost projection and invokes the real child kernel exactly as
the child's standalone tests do.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from workflows.main_cognition.engine import build_stage_runner


def build_main_cognition_tools(
    ghost,
    runtime_root: Path,
    llm_complete: Callable[..., str],
) -> Callable[[object, object], dict]:
    """Return the ``tools_factory(environment, batch)`` for the parent module."""

    def factory(environment, batch) -> dict:
        return {"run_stage": build_stage_runner(ghost, llm_complete, Path(runtime_root),
                                               read_environment=environment)}

    return factory


__all__ = ["build_main_cognition_tools"]
