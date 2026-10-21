"""Export the public dialogue of a seed or checkpoint as a chat log CSV.

The CSV has the columns ``name,message,timestamp`` that
``python -m pm.app.character_cli import`` reads. Only public ``Dialogue``
lines of the user and the companion are written, in time order. Thoughts,
summaries, and other records stay out.

``--days D --per-day N`` keeps the first N lines of each of the first D
calendar days, so a small test import spans several days.

Run:
    uv run --locked python dev/tools/export_chat_csv.py --seed test_data/personas/spamuel/canon.yaml --out chat.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pm.memory.store import MemoryStore  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export public dialogue as name,message,timestamp CSV")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--seed", type=Path, help="Persona seed YAML")
    source.add_argument("--checkpoint", type=Path, help="MemoryStore checkpoint JSON")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--days", type=int, default=None, help="Keep only the first D calendar days.")
    parser.add_argument("--per-day", type=int, default=None, help="Keep only the first N lines of each day.")
    args = parser.parse_args(argv)
    store = MemoryStore.from_yaml(args.seed) if args.seed else MemoryStore.from_checkpoint(args.checkpoint)
    rows = store.dialogue_history(user=store.ghost_config.user_name)
    kept, per_day = [], {}
    for row in rows:
        day = datetime.fromisoformat(str(row["timestamp"])).date()
        if args.days is not None and day not in per_day and len(per_day) == args.days:
            continue
        per_day[day] = per_day.get(day, 0) + 1
        if args.per_day is not None and per_day[day] > args.per_day:
            continue
        kept.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["name", "message", "timestamp"])
        for row in kept:
            writer.writerow([row["speaker"], row["text"], datetime.fromisoformat(str(row["timestamp"])).isoformat()])
    print(f"{len(kept)} of {len(rows)} dialogue lines over {len(per_day)} days -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
