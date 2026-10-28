"""Register runtime packages or read owner feedback without the retired review server.

Two paths matter to existing gates: ``register``/``register-package`` freeze a
capture through ``simulation.review.catalog``, and ``feedback`` prints every
stored owner note so a review pass can be read without the browser.
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

from simulation.personas import DEFAULT_RUNS
from pm.paths import REPO_ROOT
from simulation.review.catalog import register


def main(argv: Sequence[str] | None = None) -> int:
    """Keep capture registration available to existing work-item gates."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_RUNS)
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("register")
    add.add_argument("bundle", type=Path)
    package = commands.add_parser("register-package")
    package.add_argument("packages", nargs="+", type=Path)
    package.add_argument("--id", required=True)
    package.add_argument("--work-item", default="Debug")
    package.add_argument("--title")
    package.add_argument("--combine", action="store_true")
    commands.add_parser("feedback")
    args = parser.parse_args(argv)
    if args.command == "feedback":
        files = [*args.data.glob("runs/*/feedback.json"), *args.data.glob("reviews/*/feedback.json"), * (REPO_ROOT / ".runtime/dev-review/feedback").glob("*.json")]
        print(json.dumps([{"path": str(path), "feedback": json.loads(path.read_text(encoding="utf-8"))}
                          for path in files], ensure_ascii=False, indent=2))
        return 0
    if args.command == "register":
        bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    else:
        packages = [json.loads(path.read_text(encoding="utf-8")) for path in args.packages]
        groups = [packages] if args.combine else [[item] for item in packages]
        bundle = {"schema": "pm.simulation-capture.v1", "id": args.id, "work_item": args.work_item,
                  "title": args.title or packages[0]["title"],
                  "items": [{"id": f"records-{index}", "title": args.title or group[0]["title"], "packages": group}
                            for index, group in enumerate(groups)]}
    print(register(bundle, root=args.data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
