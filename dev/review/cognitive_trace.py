"""Export captured conversation and accepted cognitive records for human review.

This offline tool reads a completed checkpoint. The review server reads only
the resulting immutable bundle. Recorded references supply the links; record
order alone never establishes that one thought caused another thought.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dev.review.record_format import record_card
from pm.introspection.record_packages import capture_records, capture_gwt, dump_package
from pm.memory.snapshot import Snapshot


def read(path: Path) -> dict:
    """Read a UTF-8 evidence object. Missing or invalid evidence raises."""
    return json.loads(path.read_text(encoding="utf-8"))


def build_bundle(checkpoint: Path, receipt_path: Path, *, bundle_id: str,
                 first_record: int = 11203) -> dict:
    """Build a review bundle from a completed run with actual public speech.

    Refuse failed receipts and unresolved user exchanges. Keep all accepted
    records in the selected range and their transitive recorded sources.
    Human scores remain absent. The caller owns registration and feedback.
    """
    receipt = read(receipt_path)
    if not receipt.get("ok") or receipt.get("runtime_error"):
        raise ValueError("Repair the run before registering it for human review.")
    checkpoint_data = read(checkpoint)
    records = {row["data"]["id"]: row for row in checkpoint_data["state"]["knoxels"]}
    messages = receipt["messages"]
    if not any(row["speaker"] == "Spamuel" for row in messages):
        raise ValueError("A conversation review requires actual delivered companion speech.")
    for wrapped in records.values():
        data = wrapped["data"]
        if data["id"] >= first_record and data.get("metadata", {}).get("exchange_status") == "pending":
            raise ValueError(f"User exchange {data['id']} still has no completed public outcome.")
    selected = {key for key in records if key >= first_record}
    pending = list(selected)
    cards = {}
    while pending:
        key = pending.pop()
        if key in cards:
            continue
        if key not in records:
            raise ValueError(f"A recorded cognitive source is missing: {key}")
        card = record_card(records[key])
        cards[key] = card
        pending.extend(link["id"] for link in card["links"] if link["id"] not in cards)

    delivered = {row["id"] for row in messages}
    conversation = []
    for key in sorted(selected):
        data = records[key]["data"]
        if (data.get("feature_type") == "Dialogue" and key not in delivered
                and (data.get("provenance") or {}).get("source_kind") == "User"):
            conversation.append({"speaker": "Jesse", "text": str(data["content"]), "id": key,
                                 "fixture": bool(data.get("metadata", {}).get("gate_supplied"))})
    conversation.extend({key: row[key] for key in ("speaker", "text", "id")} for row in messages)

    view = Snapshot(records, {int(key): value for key, value in checkpoint_data["gwt_turns"].items()},
                    {int(key): value for key, value in checkpoint_data["revisions"].items()})
    packages = [capture_records(view, sorted(selected), title="Private activity, memory, and routing",
                                related_ids=sorted(set(cards) - selected)),
                capture_records(view, [row["id"] for row in conversation], view="chat", perspective="Jesse",
                                title="Chat log from Jesse's perspective")]
    packages.extend(capture_gwt(view, tick) for tick in sorted(view.serialize_gwt())
                    if any(key in selected for key in view.serialize_gwt()[tick]["knoxel_ids"]))
    return {"schema": "pm.dev-review.bundle.v1", "id": bundle_id, "work_item": "A19",
            "title": "A19 · What ‘later’ means to Spamuel", "created_at": receipt["ended_at"],
            "summary": "Read the conversation, then follow the private revision, codelet output, memory, and routing.",
            "items": [{"id": "story-and-boundaries", "title": "A pause is not a rejection",
                       "persona": "Spamuel / full history", "scenario": "Story, boundaries, and a revised belief",
                       "context": [{"role": "Situation", "text": "Jesse wants to leave both story endings open. "
                                    "The setup supplied a mistaken private belief that ‘later’ meant lost interest. "
                                    "Spamuel's private revision and every reply below came from real inference. "
                                    "The first replies resume the two queued story requests."}],
                       "content": "", "reviewable": True, "notes": [], "references": [],
                       "packages": packages}]}


def main() -> None:
    """Export one successful run to a review bundle under ignored evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--id", required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    if not run.is_relative_to(ROOT / ".runtime"):
        raise ValueError("Captured review evidence belongs under .runtime.")
    bundle = build_bundle(run / "ghost.json", run / "receipt.json", bundle_id=args.id)
    path = run / "review-bundle.json"
    path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for index, package in enumerate(bundle["items"][0]["packages"]):
        dump_package(package, run / "record-packages" / f"{index:02}-{package['view']}.json")
    print(path)


if __name__ == "__main__":
    main()
