"""Ghost ownership, scoped snapshots, and atomic accepted memory.

Invalid proposals and revision conflicts reject a complete batch. A failed
checkpoint leaves accepted memory unchanged. Errors reach the debugger.
No model or network call runs under the commit lock. Optional checkpoint=None
means an explicit in-memory session, not durable storage.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from threading import Event, RLock
from uuid import uuid4

from pm_core.ghost.batch import GhostCancelled, GhostConflict, GhostView, GhostWriteBatch
from pm_core.ghost.ghost_base import BaseGhost
from pm_core.ghost.references import check_references
from pm_core.model.ghost_state import CycleStateModel, GhostConfig
from pm_core.model.gwt import GwtTurnRecord
from pm_core.model.knoxel_core import KnoxelBase
from pm_core.persist.dataprovider_yaml import (
    _deserialize_knoxel, _serialize_knoxel, ghost_state_from_dict, load_ghost_state_from_yaml,
)
from pm_core.persist.snapshot import CommitReceipt, GhostCheckpoint, read_checkpoint, write_checkpoint


serialize_knoxel = _serialize_knoxel


class Ghost(BaseGhost):
    def __init__(self, config: GhostConfig, *, checkpoint: Path | None = None):
        super().__init__(None, config.model_copy(deep=True))
        self.lock = RLock()
        self.checkpoint = Path(checkpoint) if checkpoint is not None else None
        self.gwt_turns: dict[int, GwtTurnRecord] = {}
        self.revisions: dict[int, int] = {}
        self.applied_commits: dict[str, CommitReceipt] = {}
        self.managed_runs: dict[str, dict] = {}
        self.previous_emb_model = ""
        self._reserved_tick_id = 0

    @classmethod
    def from_state(cls, state: CycleStateModel, *, checkpoint: Path | None = None) -> "Ghost":
        config = GhostConfig.model_validate(state.model_dump(exclude={"knoxels", "cycles", "previous_emb_model"}))
        ghost = cls(config, checkpoint=checkpoint)
        ghost.states = [cycle.model_copy(deep=True) for cycle in state.cycles]
        ghost._all_knoxels = {knoxel.id: _deserialize_knoxel(_serialize_knoxel(knoxel)) for knoxel in state.knoxels}
        if len(ghost._all_knoxels) != len(state.knoxels):
            raise ValueError("The seed contains duplicate knoxel IDs.")
        ghost.previous_emb_model = state.previous_emb_model
        ghost.current_tick_id = max([0] + [cycle.tick_id for cycle in state.cycles] + [k.tick_id for k in state.knoxels])
        ghost._reserved_tick_id = ghost.current_tick_id
        ghost.revisions = {record_id: 1 for record_id in ghost._all_knoxels}
        ghost.rebuild_knoxel_indexes()
        return ghost

    @classmethod
    def from_yaml(cls, path: str | Path, *, checkpoint: Path | None = None) -> "Ghost":
        return cls.from_state(load_ghost_state_from_yaml(path), checkpoint=checkpoint)

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "Ghost":
        ghost = cls._restore(read_checkpoint(Path(path)), Path(path))
        for receipt in ghost.managed_runs.values():
            if receipt["state"] in {"accepted", "running", "waiting_for_input"}:
                receipt["interrupted_from"] = receipt["state"]
                receipt["state"] = "interrupted"
        return ghost

    @classmethod
    def _restore(cls, data: GhostCheckpoint, path: Path | None) -> "Ghost":
        ghost = cls.from_state(ghost_state_from_dict(data.state), checkpoint=path)
        if set(data.revisions) != set(ghost._all_knoxels) or any(revision < 1 for revision in data.revisions.values()):
            raise ValueError("Checkpoint record revisions do not match stored memory.")
        if (data.current_knoxel_id < ghost.current_knoxel_id
                or data.current_tick_id < ghost.current_tick_id
                or data.reserved_tick_id < data.current_tick_id):
            raise ValueError("Checkpoint counters would reuse accepted IDs.")
        if any(tick != record.tick_id or tick > data.current_tick_id for tick, record in data.gwt_turns.items()):
            raise ValueError("Checkpoint GWT keys do not match accepted ticks.")
        if any(key != receipt.commit_id for key, receipt in data.applied_commits.items()):
            raise ValueError("Checkpoint receipt keys do not match commit IDs.")
        ghost.current_knoxel_id = data.current_knoxel_id
        ghost.current_tick_id = data.current_tick_id
        ghost._reserved_tick_id = data.reserved_tick_id
        ghost.revisions = dict(data.revisions)
        ghost.gwt_turns = {tick: record.model_copy(deep=True) for tick, record in data.gwt_turns.items()}
        ghost.applied_commits = dict(data.applied_commits)
        ghost.managed_runs = deepcopy(data.managed_runs)
        return ghost

    def state_dict(self) -> dict:
        with self.lock:
            return {
                **self.ghost_config.model_dump(mode="json"),
                "previous_emb_model": self.previous_emb_model,
                "cycles": [cycle.model_dump(mode="json") for cycle in self.states],
                "knoxels": [_serialize_knoxel(knoxel) for knoxel in self._all_knoxels.values()],
            }

    def snapshot(self, visible_ids=None) -> GhostView:
        with self.lock:
            ids = set(self._all_knoxels) if visible_ids is None else set(visible_ids)
            records = {record_id: _serialize_knoxel(self._all_knoxels[record_id]) for record_id in sorted(ids)}
            gwt = {tick: record.model_dump(mode="json") for tick, record in self.gwt_turns.items()
                   if set(record.knoxel_ids).issubset(ids)
                   and (record.attention_frame_id is None or record.attention_frame_id in ids)}
            return GhostView(records, gwt, {record_id: self.revisions[record_id] for record_id in ids})

    def begin_batch(self, commit_id: str, *, visible_ids=None, new_tick: bool = False, cancel: Event | None = None) -> GhostWriteBatch:
        """Capture a stable view. A new tick reserves an ID without accepting memory."""
        with self.lock:
            tick = self.current_tick_id
            if new_tick:
                self._reserved_tick_id = max(self._reserved_tick_id, self.current_tick_id) + 1
                tick = self._reserved_tick_id
            return GhostWriteBatch(self, commit_id, self.snapshot(visible_ids), tick, cancel)

    def commit(self, batch: GhostWriteBatch) -> CommitReceipt:
        """Save and accept a whole batch, or raise without accepting its proposals.

        Repeated commit IDs return their original receipt. Revision conflicts
        raise GhostConflict. A cancellation before replacement raises GhostCancelled.
        """
        with self.lock:
            if batch._ghost is not self:
                raise ValueError("The batch belongs to another Ghost.")
            if batch.commit_id in self.applied_commits:
                batch._closed = True
                return self.applied_commits[batch.commit_id]
            batch.check_cancelled()
            conflicts = [record_id for record_id, revision in batch._expected.items() if self.revisions[record_id] != revision]
            run_conflicts = [run_id for run_id in batch._runs if self.managed_runs.get(run_id) != batch._initial_runs.get(run_id)]
            if conflicts or run_conflicts:
                raise GhostConflict(f"Stale memory revisions: {conflicts}. Stale run receipts: {run_conflicts}.")
            allowed_ids = set(batch._snapshot._records) | set(batch._new)
            for raw in (*batch._new.values(), *batch._replacements.values()):
                check_references(_deserialize_knoxel(raw), allowed_ids)
            if batch._gwt is not None:
                if batch.tick_id in self.gwt_turns:
                    raise GhostConflict(f"GWT tick {batch.tick_id} is already committed.")
                check_references(batch._gwt, allowed_ids)
            if batch._cycle is not None:
                if any(cycle.tick_id == batch.tick_id for cycle in self.states):
                    raise GhostConflict(f"Cycle tick {batch.tick_id} is already committed.")
                check_references(batch._cycle, allowed_ids)

            state = self.state_dict()
            records = {raw["data"]["id"]: raw for raw in state["knoxels"]}
            records.update(deepcopy(batch._new))
            records.update(deepcopy(batch._replacements))
            # Semantic vectors belong in the shared cache, not each checkpoint.
            for raw in records.values():
                raw["data"]["embedding"] = []
            state["knoxels"] = list(records.values())
            if batch._cycle is not None:
                state["cycles"].append(batch._cycle.model_dump(mode="json"))
            revisions = dict(self.revisions)
            for record_id in (*batch._new, *batch._replacements):
                revisions[record_id] = revisions.get(record_id, 0) + 1
            receipt = CommitReceipt(commit_id=batch.commit_id, added_ids=tuple(batch._new),
                                    updated_ids=tuple(batch._replacements), tick_id=batch.tick_id)
            gwt = dict(self.gwt_turns)
            if batch._gwt is not None:
                gwt[batch.tick_id] = batch._gwt
            data = GhostCheckpoint(
                state=state, current_knoxel_id=self.current_knoxel_id,
                current_tick_id=max(self.current_tick_id, batch.tick_id), reserved_tick_id=self._reserved_tick_id,
                revisions=revisions, gwt_turns=gwt,
                applied_commits={**self.applied_commits, batch.commit_id: receipt},
                managed_runs={**self.managed_runs, **batch._runs},
            )
            candidate = self._restore(data, self.checkpoint)
            batch.check_cancelled()
            if self.checkpoint is not None:
                write_checkpoint(self.checkpoint, data, batch.check_cancelled)
            # All validation and I/O precede canonical installation.
            for name in ("ghost_config", "states", "_all_knoxels", "_knoxels_by_type", "_knoxels_by_subtype", "_knoxels_by_class",
                         "current_knoxel_id", "current_tick_id", "_reserved_tick_id", "revisions", "gwt_turns",
                         "applied_commits", "managed_runs", "previous_emb_model"):
                setattr(self, name, getattr(candidate, name))
            for knoxel in self._all_knoxels.values():
                knoxel._owner = self
            batch._closed = True
            return receipt

    def store_knoxel(self, knoxel: KnoxelBase) -> KnoxelBase:
        """Commit one trusted host input through the same batch boundary."""
        with self.lock:
            batch = self.begin_batch(uuid4().hex)
            record = batch.create(knoxel)
            self.commit(batch)
            return self.get_knoxel_by_id(record.id)

    def get_knoxel_by_id(self, knoxel_id: int) -> KnoxelBase | None:
        with self.lock:
            original = self._all_knoxels.get(knoxel_id)
            if original is None:
                return None
            record = _deserialize_knoxel(_serialize_knoxel(original))
            record._owner = self
            return record

    def add_knoxel(self, knoxel: KnoxelBase, generate_embedding: bool = True) -> KnoxelBase:
        raise NotImplementedError("Use store_knoxel or GhostWriteBatch.create to add memory.")

    def replace_knoxel(self, knoxel: KnoxelBase) -> KnoxelBase:
        raise NotImplementedError("Use GhostWriteBatch.replace to update memory.")

    def create_bash_environment(self, mode: str = "read", visible_ids=None):
        from working_target import GhostBashWrapper
        return GhostBashWrapper(self, mode=mode, visible_ids=visible_ids)
