"""Workflow-side glue for the A18 self-model.

The package turns the live application context into a capability overlay and a
compact public projection. The pure evidence rules live in ``pm.introspection``.
"""
from workflows.introspection.snapshot import (
    build_capability_overlay,
    enabled_capability_names,
    observed_capability_receipts,
    overlay_public,
)

__all__ = [
    "build_capability_overlay",
    "enabled_capability_names",
    "observed_capability_receipts",
    "overlay_public",
]
