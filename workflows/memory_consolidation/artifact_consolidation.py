"""Evidence-backed conversion of managed-run artifacts into memory.

A consolidation reads a stable inventory of real task files, converts the
observations, outcome, reusable experience, spatial records, and screenshot
evidence into typed knoxels, and records one receipt keyed by the run and the
inventory hash. Repeating the request returns the original receipt instead of
duplicating memories. A failed read or commit preserves the artifacts and
leaves no partial accepted state.

Text content is the only thing that may reach an embedding or a dialogue
prompt. Screenshot base64 lives in knoxel ``evidence`` metadata and is never
placed in a knoxel's ``content``.
"""
from __future__ import annotations

import base64
import hashlib
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from pm.utils import yaml_util

from pm.records.memory_records import Feature
from pm.records.base import Provenance
from pm.records.world_records import BodyEpisodeRecord, WorldObjectRecord, WorldPlaceRecord
from pm.records.enums import FeatureType, InterlocusType, SourceKind

TEXT_SUFFIXES = frozenset({".md", ".txt", ".yaml", ".yml", ".json"})
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif"})
_TEXT_DIRECTORIES = ("outbox", "scratch")
_IMAGE_DIRECTORIES = ("screenshots", "frames", "images")


@dataclass
class ConsolidationReceipt:
    key: str
    run_id: str
    artifact_path: str
    evidence: list[dict] = field(default_factory=list)
    image_evidence: list[dict] = field(default_factory=list)
    source_event_ids: list[int] = field(default_factory=list)
    memory_ids: list[int] = field(default_factory=list)
    state: str = "consolidated"
    created_at: float = 0.0
    memory_keys: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "run_id": self.run_id,
            "artifact_path": self.artifact_path,
            "evidence": list(self.evidence),
            "image_evidence": list(self.image_evidence),
            "source_event_ids": list(self.source_event_ids),
            "memory_ids": list(self.memory_ids),
            "state": self.state,
            "created_at": self.created_at,
            "memory_keys": list(self.memory_keys),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConsolidationReceipt":
        known = {name: data[name] for name in cls.__dataclass_fields__ if name in data}
        return cls(**known)


class Consolidator:
    """One consolidator bound to one MemoryStore."""

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

    def image_inventory(self, artifact_path: str | Path) -> list[dict]:
        """Return real image artifacts with media type, hash, and base64."""
        root = Path(artifact_path)
        items: list[dict] = []
        for directory in _IMAGE_DIRECTORIES:
            base = root / directory
            if not base.is_dir():
                continue
            for path in sorted(base.iterdir()):
                if not path.is_file():
                    continue
                suffix = path.suffix.lower()
                if suffix == ".b64":
                    try:
                        raw = base64.b64decode(path.read_text(encoding="ascii").strip())
                    except (ValueError, UnicodeError):
                        continue
                    name = path.with_suffix("").name
                elif suffix in IMAGE_SUFFIXES:
                    raw = path.read_bytes()
                    name = path.name
                else:
                    continue
                if not raw:
                    continue
                media_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
                items.append({
                    "path": f"{directory}/{path.name}",
                    "media_type": media_type,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "size": len(raw),
                    "observed_at": self.clock(),
                    "base64": base64.b64encode(raw).decode("ascii"),
                })
        return items

    def _key(self, run_id: str, inventory: list[dict], images: list[dict]) -> str:
        digest = hashlib.sha256(
            "\n".join(
                [*(f"{item['path']}:{item['sha256']}" for item in inventory),
                 *(f"{item['path']}:{item['sha256']}" for item in images)]
            ).encode("utf-8")
        ).hexdigest()
        return f"{run_id}:{digest}"

    # ------------------------------------------------------- proposals

    def proposals(
        self,
        run_id: str,
        inventory: list[dict],
        images: list[dict] | None = None,
    ) -> list[Any]:
        """Convert the inventory into typed knoxels. No model is involved."""
        images = images or []
        by_path = {item["path"]: item for item in inventory}
        metadata = {"run_id": run_id, "evidence": [item["path"] for item in inventory],
                    "memory_kind": "artifact_report", "factual_promotion": False}
        proposals: list[Any] = []

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

        proposals.extend(self._spatial_proposals(run_id, by_path, images))
        return proposals

    def _spatial_proposals(self, run_id: str, by_path: dict, images: list[dict]) -> list[Any]:
        world = by_path.get("outbox/world.yaml")
        if world is None and not images:
            return []
        data = {}
        if world is not None:
            try:
                data = yaml_util.safe_load(world["text"]) or {}
            except yaml_util.YAMLError:
                data = {}
        if not isinstance(data, dict):
            data = {}

        proposals: list[Any] = []
        for place in data.get("places", []) or []:
            proposals.append(WorldPlaceRecord(
                content=str(place.get("content") or place.get("place_key") or "known place"),
                body_id=str(place.get("body_id", "body")),
                place_key=str(place.get("place_key", "")),
                position=dict(place.get("position", {})),
                dimension=str(place.get("dimension", "unknown")),
                landmarks=[str(item) for item in place.get("landmarks", [])],
                confidence=float(place.get("confidence", 0.0)),
                metadata={"run_id": run_id, "kind": "place"},
            ))
        for obj in data.get("objects", []) or []:
            proposals.append(WorldObjectRecord(
                content=str(obj.get("content") or obj.get("object_key") or "known object"),
                body_id=str(obj.get("body_id", "body")),
                object_key=str(obj.get("object_key", "")),
                object_kind=str(obj.get("object_kind", "unknown")),
                position=dict(obj.get("position", {})),
                dimension=str(obj.get("dimension", "unknown")),
                confidence=float(obj.get("confidence", 0.0)),
                metadata={"run_id": run_id, "kind": "object"},
            ))

        images_by_path = {item["path"]: item for item in images}
        episodes = list(data.get("episodes", []) or [])
        if not episodes and images:
            episodes = [{
                "body_id": str(data.get("body_id", "body")),
                "goal": str(data.get("goal", "")),
                "outcome": str(data.get("outcome", "observed")),
                "action_count": 0,
                "images": [item["path"] for item in images],
            }]
        for episode in episodes:
            gameplay = episode.get("gameplay", {})
            recent = gameplay.get("recent_outcomes", [])
            failures = [row for row in recent if row.get("unsuccessful")]
            consequence = (f" Retained {gameplay.get('outcome_count', 0)} gameplay outcomes."
                           f" Latest failure: {failures[-1].get('reason')}." if failures else "")
            if gameplay.get("recovery"):
                consequence += f" Recovery remains active: {gameplay['recovery']['causes']}."
            attached = [
                self._image_evidence(images_by_path[path])
                for path in (episode.get("images") or [])
                if path in images_by_path
            ]
            if not attached and images:
                attached = [self._image_evidence(item) for item in images]
            proposals.append(BodyEpisodeRecord(
                content=self._clip(
                    f"Body episode for {episode.get('body_id', 'body')}: "
                    f"{episode.get('outcome', 'observed')}. {episode.get('goal', '')}{consequence}"
                ),
                body_id=str(episode.get("body_id", "body")),
                goal=episode.get("goal"),
                outcome=str(episode.get("outcome", "unknown")),
                action_count=int(episode.get("action_count", 0)),
                evidence={"images": attached, "run_id": run_id, "gameplay": episode.get("gameplay", {})},
                metadata={"run_id": run_id, "kind": "episode"},
            ))
        return proposals

    def _image_evidence(self, item: dict) -> dict:
        return {
            "media_type": item["media_type"],
            "base64": item["base64"],
            "source_artifact": item["path"],
            "sha256": item["sha256"],
            "observed_at": item["observed_at"],
        }

    @staticmethod
    def _report_summary(text: str) -> str:
        try:
            data = yaml_util.safe_load(text)
        except yaml_util.YAMLError:
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
        images = self.image_inventory(artifact_path)
        key = self._key(run_id, inventory, images)
        with self.ghost.lock:
            existing = self.ghost.consolidations.get(key)
        if existing is not None:
            return dict(existing)

        batch = self.ghost.begin_batch(f"consolidation:{key}", new_tick=True)
        batch.guard_consolidation_run(run_id)
        existing_keys = {memory_key for receipt in batch._initial_consolidations.values()
                         if receipt.get("run_id") == run_id for memory_key in receipt.get("memory_keys", [])}
        proposals = self.proposals(run_id, inventory, images)
        memory_ids: list[int] = []
        memory_keys: list[str] = []
        for proposal in proposals:
            kind = proposal.metadata.get("kind", "")
            paths = {"outcome": ["outbox/report.yaml"], "observation": ["scratch/observations.md"],
                     "experience": ["scratch/result.md"], "place": ["outbox/world.yaml"],
                     "object": ["outbox/world.yaml"], "episode": ["outbox/world.yaml"]}.get(kind, [])
            sources = [{"path": item["path"], "sha256": item["sha256"]}
                       for item in inventory if item["path"] in paths]
            if kind == "episode":
                sources.extend({"path": item["path"], "sha256": item["sha256"]} for item in images)
            # Identity includes the typed payload and its exact artifact hashes.
            # Unchanged report and observation files survive progress snapshots
            # without producing another memory of the same material.
            material = f"{type(proposal).__name__}:{proposal.content}:{sources}"
            memory_key = hashlib.sha256(material.encode("utf-8")).hexdigest()
            if memory_key in existing_keys:
                continue
            existing_keys.add(memory_key)
            memory_keys.append(memory_key)
            proposal.metadata.update(memory_key=memory_key, evidence=[item["path"] for item in sources],
                                     artifact_evidence=sources,
                                     memory_kind="artifact_report", factual_promotion=False)
            proposal.provenance = Provenance(source_kind=SourceKind.Workflow, source_ref=run_id)
            if hasattr(proposal, "source_event_ids"):
                proposal.source_event_ids = [event_id for event_id in source_event_ids or []
                                             if event_id in batch.view().revisions]
            created = batch.create(proposal)
            memory_ids.append(created.id)
        receipt = ConsolidationReceipt(
            key=key,
            run_id=run_id,
            artifact_path=str(artifact_path),
            evidence=[{"path": item["path"], "sha256": item["sha256"]} for item in inventory],
            image_evidence=[
                {
                    "path": item["path"], "sha256": item["sha256"],
                    "media_type": item["media_type"],
                }
                for item in images
            ],
            source_event_ids=list(source_event_ids or []),
            memory_ids=memory_ids,
            state="consolidated",
            created_at=self.clock(),
            memory_keys=memory_keys,
        )
        batch.set_consolidation_receipt(key, receipt.to_dict())
        self.ghost.commit(batch)
        return receipt.to_dict()


def build_consolidation_tools(consolidator: Consolidator) -> dict[str, Callable[..., object]]:
    """Return the consolidation operations as plain functions."""

    def consolidated_artifacts(run_id: str, artifact_path: str,
                               source_event_ids: list[int] | None = None) -> dict:
        """Consolidate one run's text and image artifacts into memory."""
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
