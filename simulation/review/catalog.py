"""Register immutable captures and keep human feedback in separate files.

A capture is frozen once: ``register`` hashes the bundle, writes ``capture.json``
and ``registration.json`` under the capture's own folder, and refuses a different
payload for an ID that already exists. Owner feedback lives in a separate
``feedback.json`` beside it, so a mutable note can never alter captured evidence.

Owns the capture registry. Bundle
construction lives in ``simulation.review.cognitive_trace``. Record presentation
lives in ``pm.introspection.record_format``.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from pydantic import JsonValue

from pm.host.durable_files import ProcessLock, atomic_json
from pm.paths import REPO_ROOT
from simulation.personas import DEFAULT_RUNS


def identifier(value: str) -> str:
    """Require a portable capture ID without path separators."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("Capture IDs need letters, numbers, underscores, or hyphens.")
    return value


def captures(root: Path = DEFAULT_RUNS) -> list[dict[str, JsonValue]]:
    """List the registration receipts under the selected run data root."""
    rows: list[dict[str, JsonValue]] = []
    for path in sorted((root / "reviews").glob("*/registration.json")):
        rows.append({**json.loads(path.read_text(encoding="utf-8")), "available": True, "legacy": False})
    return rows


def register(bundle: dict[str, JsonValue], *, root: Path = DEFAULT_RUNS) -> str:
    """Freeze a capture. A different payload cannot replace an existing capture ID.

    The payload is presented once before freezing, so a bundle that the browser
    could not render is rejected here rather than discovered later. Writes happen
    under the registry lock: ``capture.json`` first, then the receipt that makes
    the capture visible. Returns the browser URL of the registered capture.
    """
    from pm.introspection.record_format import present_packages
    for item in bundle.get("items", []):
        present_packages(item)
    capture_id = identifier(bundle["id"])
    destination = root / "reviews" / capture_id
    encoded = json.dumps(bundle, ensure_ascii=False, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    with ProcessLock(root / "reviews" / ".registry.lock"):
        receipt = destination / "registration.json"
        if receipt.exists():
            previous = json.loads(receipt.read_text(encoding="utf-8"))
            if previous["sha256"] != digest:
                raise ValueError("This capture ID already names different immutable evidence.")
        else:
            atomic_json(destination / "capture.json", bundle)
            atomic_json(receipt, {"id": capture_id, "title": bundle.get("title", capture_id),
                "work_item": bundle.get("work_item", "Debug"), "sha256": digest,
                "created_at": datetime.now(timezone.utc).isoformat(), "path": str((destination / "capture.json").resolve())})
    return f"http://127.0.0.1:8095/#capture/{capture_id}"


def load_capture(capture_id: str, *, root: Path = DEFAULT_RUNS) -> dict[str, JsonValue]:
    """Read one registered capture or raise if the ID is unavailable."""
    identifier(capture_id)
    row = next((item for item in reversed(captures(root)) if item["id"] == capture_id), None)
    if row is None or not row["available"]:
        raise ValueError("This capture is unavailable in the current checkout.")
    return json.loads((REPO_ROOT / row["path"]).read_text(encoding="utf-8"))
