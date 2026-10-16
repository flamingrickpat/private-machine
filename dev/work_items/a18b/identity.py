"""Freeze the suite contract beside each run's evidence.

Reports use the saved contract, not a later checkout's cases or thresholds.
Saved review preserves that contract while recording the current reviewer
prompt separately. A missing or changed snapshot fails before inference.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from dev.work_items.a18b.policy import Suite
from dev.work_items.shared.evidence import save_json, sha256_file


def save_suite(root: Path, suite: Suite) -> str:
    """Write one suite snapshot and return its SHA-256 identity."""
    path = root / "suite.json"
    if path.exists():
        raise FileExistsError(f"The suite snapshot already exists: {path}")
    save_json(path, suite.model_dump(mode="json"))
    return sha256_file(path)


def save_review_origin(root: Path) -> None:
    """Retain the current reviewer implementation separately from PM generation."""
    source = Path(__file__).resolve().parent
    target = root / "review-source"
    target.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name in ("cli.py", "policy.py", "review.py", "constants.py", "prompts.yaml"):
        data = (source / name).read_bytes()
        (target / name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    save_json(root / "review-origin.json", {"scope": "reviewer implementation, not companion runtime execution",
                                           "sources_sha256": hashes})


def saved_suite(root: Path) -> Suite:
    """Load the original suite contract. Reject absent or changed evidence.

    Earlier runs without a snapshot require their original checkout. This
    function never substitutes a later manifest or acceptance threshold.
    """
    origin = json.loads((root / "run-origin.json").read_text(encoding="utf-8"))
    path = root / "suite.json"
    if not path.is_file() or "suite_snapshot_sha256" not in origin:
        raise ValueError("This run has no frozen suite snapshot. Use its original checkout to inspect it.")
    if sha256_file(path) != origin["suite_snapshot_sha256"]:
        raise ValueError("The frozen suite snapshot changed. Do not rescore altered evidence.")
    suite = Suite.model_validate_json(path.read_bytes())
    if origin["suite_version"] != suite.version:
        raise ValueError("The run origin and frozen suite versions differ.")
    return suite
