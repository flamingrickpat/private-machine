"""A19F gate: an anger-heavy stimulus and the self narrative facet in the self shard.

The gate copies the story_continuation situation into a fresh run directory
and hands one hostile user message to the real main cognition module (one
cycle, real model). It then compares the self narratives in the self shard
with those of a baseline: the story_continuation replay of the same start
state with the recorded, friendly turn.

The Verify row of A19F asks that an anger-heavy stimulus replaces the self
narrative facet. The receipt records both sets, the narrative types, the
reasons, and the CSM activation. ``replaced`` is True when the reserved self
narrative differs from the baseline. The gate does not decide whether the
chosen facet fits the anger; that is for the owner review.

Usage::

    uv run --locked python dev/work_items/alpha_a19f_narrative_gate.py --baseline .runtime/situations/story_continuation/<run>

The receipt goes to ``.runtime/alpha/<time>-a19f-narrative/receipt.json``.
A provider failure leaves the receipt as unverified.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pm.bus.envelopes import new_envelope  # noqa: E402
from pm.memory.store import MemoryStore  # noqa: E402
from pm.records.enums import FeatureType, InterlocusType  # noqa: E402
from pm.records.memory_records import Narrative, new_feature  # noqa: E402
from workflows.companion import DEFAULT_SEED, build_app  # noqa: E402

SITUATION = ROOT / "test_data" / "situations" / "story_continuation"
#: An anger-heavy message from the user. Test data, not product data.
ANGRY_TEXT = ("no. honestly your story ideas are lazy and i'm done pretending they're clever. you're a cheap "
              "chatbot doing a bad impression of someone with taste, and i'm bored of you.")


def self_narratives(ghost, state) -> list[dict]:
    """Return the self narratives in the self shard of ``state``, with their type and reason."""
    rows = []
    for item in state.workspace.self_shard.items:
        record = ghost.all_knoxels[item.record_id]
        if isinstance(record, Narrative):
            rows.append({"id": record.id, "type": record.narrative_type.value, "required_by": item.required_by,
                         "reason": item.reason, "content": record.content[:300]})
    return rows


def main() -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((SITUATION / "situation.json").read_text(encoding="utf-8"))["config"]
    out = ROOT / ".runtime" / "alpha" / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-a19f-narrative")
    out.mkdir(parents=True)
    shutil.copy2(SITUATION / "ghost.pickle", out / "ghost.pickle")
    os.environ["PM_PROMPT_LOG_DIR"] = str(out / "llm-logs")
    baseline_ghost = MemoryStore.from_checkpoint(args.baseline.resolve() / "ghost.pickle")
    receipt: dict = {"run_directory": str(out), "status": "started", "baseline": str(args.baseline),
                     "baseline_self_narratives": self_narratives(baseline_ghost, baseline_ghost.states[-1])}
    app = build_app(DEFAULT_SEED, out / "app", cognition="real", config_path=str(ROOT / config),
                    checkpoint_path=out / "ghost.pickle", thought_period=3600, world_poll_seconds=None,
                    record_lifecycle=False, emit_reply=lambda text: None)
    try:
        ghost = app.ghost
        user = new_feature(ghost, ANGRY_TEXT, FeatureType.Dialogue, source=ghost.get_user_name(),
                           interlocus=InterlocusType.Public, metadata={"session": "a19f-narrative"})
        app.main_cog.handle_batch([new_envelope("user_input", "data", "user.input", "corr-angry",
                                                {"knoxel_id": user.id})])
        app.raise_if_error()
        state = ghost.states[-1]
        receipt["stimulus"] = user.id
        receipt["mental_state"] = {"anger": state.mental_state.state_emotions.anger,
                                   "resentment": state.mental_state.state_emotions.resentment,
                                   "valence": state.mental_state.state_core.valence}
        receipt["self_narratives"] = self_narratives(ghost, state)
        receipt["decisions"] = [item.text for item in state.decisions]
        reserved = {row["id"] for row in receipt["self_narratives"] if row["required_by"] == "self_narrative"}
        before = {row["id"] for row in receipt["baseline_self_narratives"] if row["required_by"] == "self_narrative"}
        receipt["replaced"] = bool(reserved) and reserved != before
        receipt["status"] = "passed" if receipt["replaced"] else "failed"
    except Exception as error:  # A provider failure keeps the receipt as unverified evidence.
        receipt["status"] = "unverified"
        receipt["error"] = f"{type(error).__name__}: {error}"
        receipt["traceback"] = traceback.format_exc()
        raise
    finally:
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
        app.shutdown()
        app.join()
        print(f"receipt: {out / 'receipt.json'} ({receipt['status']})")
    return out


if __name__ == "__main__":
    main()
