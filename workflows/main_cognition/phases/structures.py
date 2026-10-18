"""Structures phase: build structures from the CSM.

Step 6 of A19F adds the two structure builders here. Until then the phase
changes nothing. It exists so the manifest already has the final phase order.
"""
from __future__ import annotations

from workflows.main_cognition.cycle_work import CycleWork


def run(context, work: CycleWork) -> CycleWork:
    work.receipt("structures", "no structure builders yet")
    return work


__all__ = ["run"]
