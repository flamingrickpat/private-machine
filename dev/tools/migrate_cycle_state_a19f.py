"""Rewrite saved cycle states into the typed A19F CycleState format.

Usage::

    uv run --locked python dev/tools/migrate_cycle_state_a19f.py PATH [PATH ...]

A path that ends in ``.json`` is a MemoryStore checkpoint. A path that ends
in ``.yaml`` is a persona seed such as ``canon.yaml``.

Each legacy cycle changes as follows:

- ``mental_state_latent_vec`` becomes ``mental_state``. The vector is padded
  or cut to ``VectorModelReservedSize`` and read with
  ``FullMentalState.init_from_list``, as the old ``mental_state_latent``
  property did.
- ``internal_data.control_generation`` and ``internal_data.qualia_channels``
  move to the typed fields. ``internal_data.operation`` is dropped, because
  the private thought record already stores it in its metadata.
- ``previous_tick_id``, ``rating``, ``state_needs``, and ``state_cognition``
  are dropped. Nothing read them.

Every result is validated with ``CycleState.model_validate``. The legacy
models forbid unknown keys, so an unexpected field stops the tool before it
writes. An already migrated file also stops the tool, because its cycles
have no legacy shape. GWT turn records are not changed: old v1 turns stay
v1 (A19F, "Turn record").

The checkpoint keeps the compact JSON of ``JsonCheckpointSerializer``. The
seed keeps its knoxel text byte for byte. Only the ``cycles`` block at the
top of the file is written again with the seed YAML dumper.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pm.memory.checkpoint import Checkpoint  # noqa: E402
from pm.memory.yaml_codec import _GhostStateYamlDumper  # noqa: E402
from pm.records.cycle_state import CycleState  # noqa: E402
from pm.records.mental_state import FullMentalState, VectorModelReservedSize  # noqa: E402
from pm.records.qualia_records import QualeChannelState  # noqa: E402
from pm.utils import yaml_util  # noqa: E402


class LegacyInternalData(BaseModel):
    """The keys that the old ``CycleState.internal_data`` held."""

    model_config = ConfigDict(extra="forbid")

    control_generation: int = 0
    qualia_channels: dict[str, QualeChannelState] = Field(default_factory=dict)
    operation: str = ""


class LegacyCycleState(BaseModel):
    """The ``CycleState`` shape before A19F step 2."""

    model_config = ConfigDict(extra="forbid")

    tick_id: int
    previous_tick_id: int = -1
    timestamp: datetime
    rating: int = 0
    mental_state_latent_vec: list[float] = Field(default_factory=list)
    state_needs: dict = Field(default_factory=dict)
    state_cognition: dict = Field(default_factory=dict)
    stimuli_ids: list[int] = Field(default_factory=list)
    published_slots: list[str] = Field(default_factory=list)
    attention_frame_id: int | None = None
    commit_hash: str | None = None
    internal_data: LegacyInternalData = Field(default_factory=LegacyInternalData)

    def migrate(self) -> CycleState:
        """Return the typed cycle state with the same content."""
        vector = list(self.mental_state_latent_vec)[:VectorModelReservedSize]
        vector.extend([0.0] * (VectorModelReservedSize - len(vector)))
        return CycleState(
            tick_id=self.tick_id,
            timestamp=self.timestamp,
            mental_state=FullMentalState.init_from_list(vector),
            control_generation=self.internal_data.control_generation,
            qualia_channels=self.internal_data.qualia_channels,
            stimuli_ids=self.stimuli_ids,
            published_slots=self.published_slots,
            attention_frame_id=self.attention_frame_id,
            commit_hash=self.commit_hash,
        )


def migrate_cycles(raw_cycles: list) -> list[dict]:
    """Validate legacy cycles, migrate them, and return JSON-ready dicts."""
    result = []
    for raw in raw_cycles:
        state = LegacyCycleState.model_validate(raw).migrate()
        result.append(CycleState.model_validate(state.model_dump(mode="json")).model_dump(mode="json"))
    return result


def migrate_checkpoint(path: Path) -> int:
    """Rewrite the cycles of one checkpoint. Returns the number of cycles."""
    data = json.loads(path.read_text(encoding="utf-8"))
    data["state"]["cycles"] = migrate_cycles(data["state"]["cycles"])
    checkpoint = Checkpoint.model_validate(data)
    path.write_bytes(json.dumps(checkpoint.model_dump(mode="json"), ensure_ascii=False, allow_nan=False).encode("utf-8"))
    return len(data["state"]["cycles"])


_TOP_LEVEL_KEY = re.compile(r"^[A-Za-z_]+:", re.MULTILINE)


def migrate_seed(path: Path) -> int:
    """Rewrite the leading ``cycles`` block of one seed. Returns the number of cycles."""
    # Read bytes, so the knoxel text keeps its line endings exactly. A Windows
    # checkout with core.autocrlf has CRLF line endings.
    text = path.read_bytes().decode("utf-8")
    if not text.startswith("cycles:"):
        raise ValueError(f"{path} does not start with the cycles block")
    eol = "\r\n" if text.startswith("cycles:\r\n") else "\n"
    end = _TOP_LEVEL_KEY.search(text, len("cycles:")).start()
    cycles = migrate_cycles(yaml_util.safe_load(text[:end])["cycles"])
    # The same dumper options as pm.memory.yaml_codec.dump_ghost_state_to_yaml.
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
