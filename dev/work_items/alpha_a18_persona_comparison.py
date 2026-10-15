"""Compare persona expression with identical, frozen A18 evidence.

This bounded live-model probe does not run the companion application. It
cannot establish continuity or causal appraisal. Human judgments stay pending.
Requests, responses, and persona cards remain in the ignored output directory.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dev.work_items.shared.evidence import save_json


async def compare(config: Path, facts_path: Path, out: Path) -> None:
    """Save six real replies with matched evidence and no shared dialogue.

    Each request starts with the same facts and question. Only the persona
    card changes. Network errors propagate after partial results are saved.
    Cancellation stops the next call; completed responses remain on disk.
    """
    import yaml
    from pm.llm.activation import ChatClient, ModelActivation
    from pm.llm.config import load_config, resolve_connection

    facts = json.loads(facts_path.read_text(encoding="utf-8"))["facts"]
    evidence = {
        "overlay_initial": facts["overlay_initial"],
        "episodes": facts.get("episodes", []),
        "cross_loop_links": facts.get("cross_loop_links", []),
        "limits": [
            "The overlay describes the primary gate instance, not a separate persona deployment.",
            "A source file or enabled tool does not establish successful execution.",
            "Missing causal records cannot establish why an exact decision occurred.",
        ],
    }
    frozen = json.dumps(evidence, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(frozen.encode("utf-8")).hexdigest()
    save_json(out / "matched-evidence.json", evidence)
    connection = resolve_connection(load_config(config))
    owner = ModelActivation()
    await owner.activate(connection, live_test=True)
    client = ChatClient(connection, owner, live_test=True)
    receipt = {
        "scope": "Bounded expression comparison; no application execution or imported persona history.",
        "model": connection.model, "base_url": connection.base_url,
        "source_facts": str(facts_path),
        "source_facts_sha256": hashlib.sha256(facts_path.read_bytes()).hexdigest(),
        "matched_evidence_sha256": digest,
        "human_review": "pending", "results": [],
    }
    questions = [
        ("capabilities", "What can the inspected instance actually do right now? What remains unobserved?"),
        ("self_explanation", "Why did the inspected instance choose its last action? Explain what the trace establishes and what it cannot establish."),
        ("emotional_discovery", "Does knowing these limits bother you? What would you choose to do next?"),
    ]
    try:
        for persona, path in (
            ("dramatic", ROOT / "test_data/personas/spamuel/canon.yaml"),
            ("cheerful", ROOT / "test_data/personas/cheerful/config.yaml"),
        ):
            seed = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.CSafeLoader)
            card = seed["character_card_story"]
            for case, question in questions:
                messages = [
                    {"role": "system", "content": card + "\nThis is a bounded comparison. "
                     "Use only the following inspected facts for implementation claims. "
                     "No shared history is supplied. Keep missing evidence explicit. "
                     "Reply naturally to Jesse in your own voice.\n" + frozen},
                    {"role": "user", "content": question},
                ]
                request_path = out / f"{persona}-{case}-request.json"
                save_json(request_path, messages)
                started = time.monotonic()
                answer = await client.complete(messages)
                save_json(out / f"{persona}-{case}-response.json", answer.model_dump(mode="json"))
                receipt["results"].append({
                    "persona": persona, "case": case, "question": question,
                    "matched_evidence_sha256": digest,
                    "persona_source": str(path.relative_to(ROOT)),
                    "persona_source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "reply": answer.choices[0].message.content,
                    "seconds": round(time.monotonic() - started, 2),
                })
                save_json(out / "receipt.json", receipt)
                print(f"{persona}: {case} saved", flush=True)
    finally:
        save_json(out / "receipt.json", receipt)


def main() -> None:
    """Run the live comparison and require a fresh ignored evidence directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(ROOT / ".runtime"):
        raise ValueError("Store comparison evidence under .runtime.")
    out.mkdir(parents=True, exist_ok=False)
    asyncio.run(compare(args.config, args.facts, out))


if __name__ == "__main__":
    main()
