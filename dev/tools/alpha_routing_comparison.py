"""Save current-chat answers for development cases without treating them as truth."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(ROOT / ".runtime"):
        raise ValueError("Store comparison evidence under .runtime.")
    out.mkdir(parents=True, exist_ok=False)
    fixture = ROOT / "test_data/captures/alpha/routing-development.json"
    cases = json.loads(fixture.read_text(encoding="utf-8"))
    from pm.llm.config import load_config, resolve_connection
    from pm.llm.activation import ChatClient, ModelActivation
    import asyncio
    connection = resolve_connection(load_config(args.config))
    owner = ModelActivation()
    asyncio.run(owner.activate(connection, live_test=True))
    client = ChatClient(connection, owner, live_test=True)
    results = []
    try:
        for case in cases["cases"]:
            start = time.monotonic()
            answer = asyncio.run(client.complete([
                {"role": "system", "content": cases["question"] + "\n" + json.dumps(cases["criteria"]) + "\nReply with one label only."},
                {"role": "user", "content": case["state"]}]))
            text = answer.choices[0].message.content
            result = {"id": case["id"], "answer": text, "seconds": time.monotonic() - start,
                "proposed_label": case["expected"], "response": answer.model_dump(mode="json")}
            results.append(result)
            print(json.dumps({key: result[key] for key in ("id", "answer", "seconds")}), flush=True)
    finally:
        receipt = {"model": connection.model, "base_url": connection.base_url,
            "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(), "review_status": cases["review_status"],
            "holdout_evaluated": False, "results": results,
            "scope": "Comparison answers only. Human facts define acceptance; no prompt or threshold tuning in A01."}
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
