"""Evidence-backed conversion of managed-run text artifacts into memory.

A consolidation reads a stable inventory of real task files, converts the
observations, outcome, and reusable experience into typed knoxels, and records
one receipt keyed by the run and the inventory hash. Repeating the request
returns the original receipt instead of duplicating memories. A failed read or
commit preserves the artifacts and leaves no partial accepted state.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from pm_core.model.knoxel_common import Feature
from pm_core.model.knoxel_enums import FeatureType, InterlocusType

TEXT_SUFFIXES = frozenset({".md", ".txt", ".yaml", ".yml", ".json"})
_TEXT_DIRECTORIES = ("outbox", "scratch")


@dataclass
class ConsolidationReceipt:
    key: str
    run_id: str
    artifact_path: str
    evidence: list[dict] = field(default_factory=list)
    source_event_ids: list[int] = field(default_factory=list)
    memory_ids: list[int] = field(default_factory=list)
    state: str = "consolidated"
    created_at: float = 0.0

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "run_id": self.run_id,
            "artifact_path": self.artifact_path,
            "evidence": list(self.evidence),
            "source_event_ids": list(self.source_event_ids),
            "memory_ids": list(self.memory_ids),
            "state": self.state,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConsolidationReceipt":
        known = {name: data[name] for name in cls.__dataclass_fields__ if name in data}
        return cls(**known)


class Consolidator:
    """One consolidator bound to one Ghost."""

    def __init__(self, ghost, *, clock: Callable[[], float]) -> None:
        self.ghost = ghost
        self.clock = clock

    # --------------------------------------------------------- inventory

    def inventory(self, artifact_path: str | Path) -> list[dict]:
        """Return the real text artifacts with their content hashes."""
        root = Path(artifact_path)
        items: list[dict] = []
        for directory in _TEXT_DIRECTORIES:
            base = root / directory
            if not base.is_dir():
                continue
            for path in sorted(base.iterdir()):
                if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
                if not text.strip():
                    continue
                items.append({
                    "path": f"{directory}/{path.name}",
                    "text": text,
                    "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                })
        return items

    def _key(self, run_id: str, inventory: list[dict]) -> str:
        digest = hashlib.sha256(
            "\n".join(f"{item['path']}:{item['sha256']}" for item in inventory).encode("utf-8")
        ).hexdigest()
        return f"{run_id}:{digest}"

    # ------------------------------------------------------- proposals

    def proposals(self, run_id: str, inventory: list[dict]) -> list[Feature]:
        """Convert the inventory into typed features. No model is involved."""
        by_path = {item["path"]: item for item in inventory}
        proposals: list[Feature] = []
        metadata = {"run_id": run_id, "evidence": [item["path"] for item in inventory]}

        report = by_path.get("outbox/report.yaml")
        if report is not None:
            summary = self._report_summary(report["text"])
            proposals.append(Feature(
                content=f"Task {run_id} outcome: {summary}",
                feature_type=FeatureType.WorldEvent,
                interlocus=InterlocusType.PrivateInternal,
                causal=True,
                override_source="memory_consolidation",
                metadata={**metadata, "kind": "outcome"},
            ))
        observations = by_path.get("scratch/observations.md")
        if observations is not None:
            proposals.append(Feature(
                content=self._clip(f"Task {run_id} observations: {observations['text']}"),
                feature_type=FeatureType.WorldEvent,
                interlocus=InterlocusType.PrivateInternal,
                causal=True,
                override_source="memory_consolidation",
                metadata={**metadata, "kind": "observation"},
            ))
        result = by_path.get("scratch/result.md")
        if result is not None:
            proposals.append(Feature(
                content=self._clip(f"Task {run_id} experience: {result['text']}"),
                feature_type=FeatureType.InnerMonologue,
                interlocus=InterlocusType.PrivateReportable,
                causal=True,
                override_source="memory_consolidation",
                metadata={**metadata, "kind": "experience"},
            ))
        return proposals

    @staticmethod
    def _report_summary(text: str) -> str:
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError:
            return Consolidator._clip(text)
        if isinstance(data, dict):
            for key in ("summary", "result", "outcome"):
                value = data.get(key)
                if value:
                    return str(value)
        return Consolidator._clip(text)

    @staticmethod
    def _clip(text: str, limit: int = 500) -> str:
        text = str(text).strip()
        return text if len(text) <= limit else text[: limit - 1].rstrip() + "\u2026"

    # -------------------------------------------------------- operation

    def consolidate(
        self,
        run_id: str,
        artifact_path: str | Path,
        *,
        source_event_ids: list[int] | None = None,
    ) -> dict:
        """Commit one evidence-backed consolidation, or reuse its receipt."""
        inventory = self.inventory(artifact_path)
        key = self._key(run_id, inventory)
        with self.ghost.lock:
            existing = self.ghost.consolidations.get(key)
        if existing is not None:
            return dict(existing)

        proposals = self.proposals(run_id, inventory)
        batch = self.ghost.begin_batch(f"consolidation:{key}", new_tick=True)
        memory_ids: list[int] = []
        for proposal in proposals:
            created = batch.create(proposal)
            memory_ids.append(created.id)
        receipt = ConsolidationReceipt(
            key=key,
            run_id=run_id,
            artifact_path=str(artifact_path),
            evidence=[{"path": item["path"], "sha256": item["sha256"]} for item in inventory],
            source_event_ids=list(source_event_ids or []),
            memory_ids=memory_ids,
            state="consolidated",
            created_at=self.clock(),
        )
        batch.set_consolidation_receipt(key, receipt.to_dict())
        self.ghost.commit(batch)
        return receipt.to_dict()


def build_consolidation_tools(consolidator: Consolidator) -> dict[str, Callable[..., object]]:
    """Return the consolidation operations as plain functions."""

    def consolidated_artifacts(run_id: str, artifact_path: str,
                               source_event_ids: list[int] | None = None) -> dict:
        """Consolidate one run's text artifacts into evidence-backed memory."""
        return consolidator.consolidate(
            run_id, artifact_path, source_event_ids=source_event_ids,
        )

    def list_consolidations() -> list[dict]:
        """List every consolidation receipt in the one ledger."""
        with consolidator.ghost.lock:
            return [dict(item) for item in consolidator.ghost.consolidations.values()]

    return {
        "consolidated_artifacts": consolidated_artifacts,
        "list_consolidations": list_consolidations,
    }


__all__ = [
    "ConsolidationReceipt",
    "Consolidator",
    "build_consolidation_tools",
]
