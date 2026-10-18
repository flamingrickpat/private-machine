"""Remove the old ledger fields from saved cycle states (A19F step 10).

Usage::

    uv run --locked python dev/tools/migrate_cycle_ledger_a19f.py PATH [PATH ...]

A path that ends in ``.json`` is a MemoryStore checkpoint. A path that ends
in ``.yaml`` is a persona seed such as ``canon.yaml``.

A19F step 10 removed four fields from ``CycleState``. Each saved cycle
changes as follows:

- ``stimuli_ids`` of a main cycle (``"main"`` in ``published_slots``) become
  one stimulus group with the self key, when the cycle has no groups yet.
  The real group keys are not known for old cycles. ``CycleState.claimed_ids``
  then still names the stimuli that the cycle appraised, so old thoughts and
  results are not replayed.
- ``published_slots``, ``attention_frame_id``, and ``commit_hash`` are
  dropped. The GWT turn record names the loop. No cycle filled
  ``commit_hash``.

The tool does not change records. Old ``AttentionFrameKnoxel`` records stay,
because ``AttentionFrameRecord`` stays as a read-only legacy class. Old GWT
turns stay v1; their ``attention_frame_id`` key is ignored on load.

Every result is validated with ``CycleState.model_validate``, which forbids
unknown keys. A cycle without the old fields stops the tool, so a file is
not migrated twice.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pm.memory.checkpoint import Checkpoint  # noqa: E402
from pm.memory.yaml_codec import _GhostStateYamlDumper  # noqa: E402
from pm.records.cycle_records import StimulusGroup  # noqa: E402
from pm.records.cycle_state import CycleState  # noqa: E402
from pm.utils import yaml_util  # noqa: E402


class LegacyLedger(BaseModel):
    """The four ledger fields of a cycle before A19F step 10."""

    model_config = ConfigDict(extra="forbid")

    stimuli_ids: list[int] = Field(default_factory=list)
    published_slots: list[str] = Field(default_factory=list)
    attention_frame_id: int | None = None
    commit_hash: str | None = None


LEDGER_KEYS = tuple(LegacyLedger.model_fields)


def migrate_cycle(raw: dict) -> dict:
    """Return one cycle without the ledger fields, as a JSON-ready dict.

    A cycle without any ledger key raises ValueError: it is already migrated.
    """
    if not any(key in raw for key in LEDGER_KEYS):
        raise ValueError(f"Cycle {raw['tick_id']} has no ledger fields. It is already migrated.")
    ledger = LegacyLedger.model_validate({key: raw[key] for key in LEDGER_KEYS if key in raw})
    state = CycleState.model_validate({key: value for key, value in raw.items() if key not in LEDGER_KEYS})
    if "main" in ledger.published_slots and ledger.stimuli_ids and not state.stimulus_groups:
        state.stimulus_groups = [StimulusGroup(stimulus_ids=list(ledger.stimuli_ids))]
    return state.model_dump(mode="json")


def migrate_checkpoint(path: Path) -> int:
    """Rewrite the cycles of one checkpoint. Returns the number of cycles."""
    data = json.loads(path.read_text(encoding="utf-8"))
    data["state"]["cycles"] = [migrate_cycle(raw) for raw in data["state"]["cycles"]]
    checkpoint = Checkpoint.model_validate(data)
    path.write_bytes(json.dumps(checkpoint.model_dump(mode="json"), ensure_ascii=False, allow_nan=False).encode("utf-8"))
    return len(data["state"]["cycles"])


_TOP_LEVEL_KEY = re.compile(r"^[A-Za-z_]+:", re.MULTILINE)


def migrate_seed(path: Path) -> int:
    """Rewrite the leading ``cycles`` block of one seed. Returns the number of cycles.

    The knoxel text after the block stays byte for byte, as in
    ``migrate_cycle_state_a19f.py``.
    """
    text = path.read_bytes().decode("utf-8")
    if not text.startswith("cycles:"):
        raise ValueError(f"{path} does not start with the cycles block")
    eol = "\r\n" if text.startswith("cycles:\r\n") else "\n"
    end = _TOP_LEVEL_KEY.search(text, len("cycles:")).start()
    cycles = [migrate_cycle(raw) for raw in yaml_util.safe_load(text[:end])["cycles"]]
    block = yaml.dump({"cycles": cycles}, Dumper=_GhostStateYamlDumper, sort_keys=False, allow_unicode=True,
                      default_flow_style=False, width=2**31 - 1)
    path.write_bytes((block.replace("\n", eol) + text[end:]).encode("utf-8"))
    return len(cycles)


def main(paths: list[str]) -> None:
    for name in paths:
        path = Path(name)
        count = migrate_checkpoint(path) if path.suffix == ".json" else migrate_seed(path)
        print(f"{path}: migrated {count} cycles")


if __name__ == "__main__":
    main(sys.argv[1:])
