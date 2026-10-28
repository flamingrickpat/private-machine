"""Read the append-only body journal of one specialist session.

The gateway writes a dispatch row before each body call and a result row
after it. A torn final line is an uncertain tail, not corruption.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_actor_journal(path: Path) -> list[dict[str, Any]]:
    """Read complete receipts. Retain a torn final append as an uncertain tail."""
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                marker = json.loads(lines[index + 1])
                if marker != {"event": "torn_tail", "text": line}:
                    raise
            rows.append({"event": "torn_tail", "text": line})
    return rows

