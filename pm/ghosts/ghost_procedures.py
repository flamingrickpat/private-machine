"""
Compatibility orchestrator module.

The legacy ghost implementation imports `pm.ghosts.ghost_procedures.BaseProcMain`.
The newer modular implementation lives in `pm.ghosts.procedure_main`.
This module keeps that public import stable and avoids runtime import failures.
"""

from pm.ghosts.procedure_main import BaseProcMain

__all__ = ["BaseProcMain"]

