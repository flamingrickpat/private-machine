"""Re-run the W7 conversation evaluator over saved transcripts.

    python -m tests.live.evaluate_saved .runtime/w7/conversations/<run>

Generation is never re-run. This reads the saved exchange artifacts, scores
each trajectory and each reply, and rewrites ``evaluations.json`` and
``summary.json`` in place.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pm.llm.activation import ModelActivation
from pm.llm.config import load_config
from tests.live import conversation as conv


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Score a saved W7 conversation run.")
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--scenarios", default="", help="Comma-separated subset.")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--trajectory-only", action="store_true",
        help="Re-score only the whole-trajectory judgment; keep saved reply scores.",
    )
    args = parser.parse_args(argv)

    default = args.config or (
        "config.local.yaml" if Path("config.local.yaml").is_file() else "config.example.yaml"
    )
    config = load_config(default)
    selected = [name for name in args.scenarios.split(",") if name] or None
    summary = conv.run_evaluation(
        args.run_root, config=config, owner=ModelActivation(), selected=selected,
        trajectory_only=args.trajectory_only,
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
