from __future__ import annotations

from pathlib import Path
from pm.host.application import Application
from pm.introspection.capabilities import CapabilityOverlay
from workflows.inner_thought.request_views import CapabilityInspection, InspectedCapability
from pm.memory.store import MemoryStore
from workflows.introspection.snapshot import build_capability_overlay


class CapabilityInspectionOwner:
    def __init__(self, store: MemoryStore, active_module_ids: frozenset[str],
                 config_path: Path, application: Application) -> None:
        self.store = store
        self.active_module_ids = active_module_ids
        self.config_path = config_path
        self.application = application

    def refresh(self) -> CapabilityOverlay:
        overlay = build_capability_overlay(ghost=self.store, active_modules=self.active_module_ids,
            config_path=self.config_path)
        overlay.previous = self.application.capability_overlay
        self.application.capability_overlay = overlay
        return overlay


def capability_inspection(overlay: CapabilityOverlay | None) -> CapabilityInspection:
    if overlay is None:
        return CapabilityInspection()
    return CapabilityInspection(available=True, checked_at=overlay.built_at,
        stale=overlay.stale_names(), capabilities=[InspectedCapability(name=finding.name,
            state=finding.strongest.value, summary=finding.summary, revision=finding.revision)
            for finding in overlay.findings.values()])
