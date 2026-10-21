"""Names of the modules that the console runs, and their loader.

The console imports the assembly and the operator hooks by name at run time.
This keeps ``pm`` free of static imports of ``workflows``.
"""
from __future__ import annotations

import importlib
from types import ModuleType

#: The assembly whose ``build_app`` each console worker runs.
DEFAULT_ASSEMBLY = "workflows.companion"
#: The module with ``prepare_environment(values)`` and async ``probes(values)``.
DEFAULT_OPERATOR_HOOKS = "workflows.minecraft_body.operator_hooks"


def load(name: str) -> ModuleType:
    """Import one configured module."""
    return importlib.import_module(name)
