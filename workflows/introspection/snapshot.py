"""Build the A18 capability overlay from the live application context.

The application calls this module at assembly time and inside a private cycle.
It reads the real repository through an explicit source grant, the real active
wiring, and the real accepted records. It never substitutes a plan for a
wiring fact, and it never turns a tool name into an observed execution.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path

from pm.introspection.capabilities import (
    CapabilityContext,
    CapabilityOverlay,
    build_overlay,
    default_capability_declarations,
)
from pm.introspection.grants import SourceInspector, build_default_grant, git_revision
from pm.records.cognitive_records import BeliefRecord
from pm.records.enums import FeatureType

__all__ = [
    "build_capability_overlay",
    "enabled_capability_names",
    "observed_capability_receipts",
    "overlay_public",
]


def _config_revision(config_path: str | Path | None) -> str:
    if not config_path:
        return ""
    path = Path(config_path)
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def enabled_capability_names(active_modules: frozenset[str] | set[str]) -> frozenset[str]:
    """Return the capabilities their active wiring turns on.

    ``active_modules`` names the modules the assembly actually registered. A
    capability is enabled only when its owner module is present.
    """
    mapping = {
        "public_dialogue": {"dialogue", "dialogue_story", "assistant"},
        "private_thought": {"inner_thought"},
        "memory_tools": {"inner_thought", "assistant"},
        "source_inspection": {"inner_thought"},
        "minecraft_body": {"minecraft", "minecraft_project", "world_events"},
    }
    active = set(active_modules)
    return frozenset(name for name, owners in mapping.items() if owners & active)


def observed_capability_receipts(ghost) -> dict[str, list[dict]]:
    """Derive observed-execution rows from real accepted records.

    The rows come from committed features and managed-run receipts. A planned
    or merely registered capability produces no row here.
    """
    observed: dict[str, list[dict]] = {}

    dialogue = [record for record in ghost.all_features
                if record.feature_type == FeatureType.Dialogue
                and (record.provenance is not None and record.provenance.source_kind.value == "Self")]
    if dialogue:
        newest = max(dialogue, key=lambda record: record.tick_id)
        observed["public_dialogue"] = [{
            "reference": f"feature:{newest.id}", "detail": "A public reply was committed.",
            "revision": str(ghost.revisions.get(newest.id, 1)),
        }]

    thoughts = [record for record in ghost.all_features
                if record.feature_type == FeatureType.InnerMonologue
                and (record.metadata or {}).get("loop") == "inner"]
    if thoughts:
        newest = max(thoughts, key=lambda record: record.tick_id)
        observed["private_thought"] = [{
            "reference": f"feature:{newest.id}", "detail": "A private cycle committed a thought.",
            "revision": str(ghost.revisions.get(newest.id, 1)),
        }]

    inspections = [record for record in ghost.all_knoxels.values()
                   if isinstance(record, BeliefRecord)
                   and (record.metadata or {}).get("memory_kind") == "capability_belief"]
    if inspections:
        newest = max(inspections, key=lambda record: record.id)
        observed["source_inspection"] = [{
            "reference": f"belief:{newest.id}",
            "detail": "A capability belief cites inspected evidence.",
            "revision": str(ghost.revisions.get(newest.id, 1)),
        }]

    body_rows = []
    for run_id, receipt in ghost.managed_runs.items():
        if not str(receipt.get("workflow_name", "")).startswith("minecraft"):
            continue
        if receipt.get("state") != "completed":
            continue
        body_rows.append({"reference": f"run:{run_id}",
                          "detail": "A Minecraft run completed with retained evidence.",
                          "revision": str(receipt.get("revision", 0))})
    if body_rows:
        observed["minecraft_body"] = body_rows[:4]
    return observed


def _repository_revision(root: Path) -> str:
    """Return the cached working-tree revision for one repository root.

    The revision changes only on a commit, so one lookup per process is
    enough. A private cycle can then rebuild the overlay without starting a
    subprocess every tick.
    """
    key = str(root)
    if key not in _REVISION_CACHE:
        _REVISION_CACHE[key] = git_revision(root)
    return _REVISION_CACHE[key]


_REVISION_CACHE: dict[str, str] = {}


def build_capability_overlay(*, ghost, active_modules: set[str] | frozenset[str],
                             config_path: str | Path | None = None,
                             repo_root: str | Path | None = None) -> CapabilityOverlay:
    """Build one dated capability overlay for the running instance."""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    inspector = SourceInspector(
        build_default_grant(root),
        repo_revision=_repository_revision(root),
        configuration_revision=_config_revision(config_path),
    )
    return build_overlay(default_capability_declarations(), CapabilityContext(
        inspector=inspector,
        enabled_names=enabled_capability_names(active_modules),
        observed_receipts=observed_capability_receipts(ghost),
    ))


def overlay_public(overlay: CapabilityOverlay | None) -> dict:
    """Return a compact, speech-facing projection of the overlay.

    Ordinary speech uses the summary and the strongest state. Raw evidence
    lists stay in the debug view and in the belief records.
    """
    if overlay is None:
        return {"available": False, "checked_at": None, "capabilities": []}
    return {
        "available": True,
        "checked_at": overlay.built_at.astimezone(timezone.utc).isoformat(),
        "stale": overlay.stale_names(),
        "capabilities": [
            {"name": finding.name, "state": finding.strongest.value, "summary": finding.summary,
             "revision": finding.revision}
            for finding in overlay.findings.values()
        ],
    }
