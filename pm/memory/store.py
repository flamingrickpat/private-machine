"""MemoryStore ownership, scoped snapshots, and atomic accepted memory.

Invalid proposals and revision conflicts reject a complete batch. A failed
checkpoint leaves accepted memory unchanged. Errors reach the debugger.
No model or network call runs under the commit lock. Optional checkpoint=None
means an explicit in-memory session, not durable storage.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from threading import Event, RLock
from uuid import uuid4

from pm.memory.snapshot import GhostCancelled, GhostConflict, Snapshot
from pm.memory.write_batch import WriteBatch
from pm.memory.record_index import RecordIndex
from pm.memory.references import check_references
from pm.records.cycle_state import CycleStateModel, PersonaConfig
from pm.records.global_workspace import GwtTurnRecord
from pm.records.base import RecordBase
from pm.memory.yaml_codec import _deserialize_knoxel, _serialize_knoxel, ghost_state_from_dict, load_ghost_state_from_yaml
from pm.memory.checkpoint import CommitReceipt, Checkpoint, read_checkpoint, write_checkpoint


serialize_knoxel = _serialize_knoxel


class MemoryStore(RecordIndex):
    def __init__(self, config: PersonaConfig, *, checkpoint: Path | None = None):
        super().__init__(None, config.model_copy(deep=True))
        self.lock = RLock()
        self.checkpoint = Path(checkpoint) if checkpoint is not None else None
        self.gwt_turns: dict[int, GwtTurnRecord] = {}
        self.revisions: dict[int, int] = {}
        self.applied_commits: dict[str, CommitReceipt] = {}
        self.managed_runs: dict[str, dict] = {}
        self.consolidations: dict[str, dict] = {}
        self.previous_emb_model = ""
        self._reserved_tick_id = 0

    @classmethod
    def from_state(cls, state: CycleStateModel, *, checkpoint: Path | None = None) -> "MemoryStore":
        config = PersonaConfig.model_validate(state.model_dump(exclude={"knoxels", "cycles", "previous_emb_model"}))
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
    def from_yaml(cls, path: str | Path, *, checkpoint: Path | None = None) -> "MemoryStore":
        """Import the seed as attributed background without rewriting its file.

        Imported dialogue establishes what the source says. It does not prove
        that this runtime performed an action or had a sensory capability.
        The original provenance remains in metadata for later inspection.
        """
        path = Path(path)
        ghost = cls.from_state(load_ghost_state_from_yaml(path), checkpoint=checkpoint)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        from pm.records.base import Provenance
        from pm.records.enums import SourceKind
        for record in ghost._all_knoxels.values():
            record.metadata = {**record.metadata, "memory_kind": "background",
                               "seed_sha256": digest, "seed_source": str(path.resolve()),
                               "imported_provenance": record.provenance.model_dump(mode="json") if record.provenance else None}
            record.provenance = Provenance(source_kind=SourceKind.Migration, source_ref=digest)
        return ghost

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "MemoryStore":
        ghost = cls._restore(read_checkpoint(Path(path)), Path(path))
        for receipt in ghost.managed_runs.values():
            if receipt["state"] in {"accepted", "running", "waiting_for_input"}:
                receipt["interrupted_from"] = receipt["state"]
                receipt["state"] = "interrupted"
        return ghost

    @classmethod
    def _restore(cls, data: Checkpoint, path: Path | None) -> "MemoryStore":
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
        ghost.consolidations = deepcopy(data.consolidations)
        return ghost

    def state_dict(self) -> dict:
        with self.lock:
            return {
                **self.ghost_config.model_dump(mode="json"),
                "previous_emb_model": self.previous_emb_model,
                "cycles": [cycle.model_dump(mode="json") for cycle in self.states],
                "knoxels": [_serialize_knoxel(knoxel) for knoxel in self._all_knoxels.values()],
            }

    def snapshot(self, visible_ids=None) -> Snapshot:
        with self.lock:
            ids = set(self._all_knoxels) if visible_ids is None else set(visible_ids)
            records = {record_id: _serialize_knoxel(self._all_knoxels[record_id]) for record_id in sorted(ids)}
            gwt = {tick: record.model_dump(mode="json") for tick, record in self.gwt_turns.items()
                   if set(record.knoxel_ids).issubset(ids)
                   and (record.attention_frame_id is None or record.attention_frame_id in ids)}
            return Snapshot(records, gwt, {record_id: self.revisions[record_id] for record_id in ids})

    def dialogue_history(self, *, visible_ids=None, user: str | None = None) -> list[dict]:
        """Return public dialogue from accepted memory, ordered by world time.

        ``visible_ids`` limits the read scope. ``user`` selects that speaker
        and the companion. This method reads no trace files or network state.
        """
        records = self.snapshot(visible_ids).serialize_knoxels()
        names = {key: raw["data"]["content"] for key, raw in records.items() if raw["__type__"] == "Entity"}
        result = []
        for key, raw in records.items():
            data = raw["data"]
            if raw["__type__"] != "Feature" or data.get("feature_type") != "Dialogue" or data.get("interlocus") != 1:
                continue
            speaker = data.get("override_source") or names.get(data.get("entity_id"), "Unknown")
            if user is not None and speaker not in {user, self.ghost_config.companion_name}:
                continue
            result.append({"id": key, "speaker": speaker, "text": data["content"],
                           "timestamp": data.get("timestamp_world_begin") or data.get("timestamp_creation"),
                           "tick_id": data.get("tick_id")})
        return sorted(result, key=lambda row: (row["timestamp"] or "", row["id"]))

    async def retrieve_experiences(self, query: str, embeddings=None, *, view=None,
                                   visible_ids=None, frontier=None, scopes=None,
                                   limit: int = 24, cancel=None) -> dict:
        """Retrieve from the supplied stable view or a fresh permitted snapshot.

        Scope and frontier filters run before embedding calls. The returned
        revisions let the caller reject evidence that changed before commit.
        """
        from pm.memory.retrieval import retrieve
        return await retrieve(view if view is not None else self.snapshot(visible_ids), query,
                              embeddings, frontier=frontier, scopes=scopes, limit=limit, cancel=cancel)

    def activity_summary(self, start, end, *, visible_ids=None) -> dict:
        """Read actual activity and original evidence IDs in one UTC interval."""
        from pm.memory.retrieval import activity
        return activity(self.snapshot(visible_ids), start, end)

    async def search_memory(self, query: str, embeddings, *, visible_ids=None, limit: int = 50,
                            prepare: bool = False, cancel=None, progress=None,
                            record_type: str = "", subtype: str = "") -> dict:
        """Search accepted text with the supplied embedding service.

        Search uses cached vectors and reports missing coverage. ``prepare``
        explicitly embeds missing records before search. Network work never
        holds the MemoryStore lock. Type and subtype filters apply before indexing and ranking.
        Callers must pass their permitted record IDs.
        """
        view = self.snapshot(visible_ids)
        records = {key: raw for key, raw in view.serialize_knoxels().items()
                   if (not record_type or raw["__type__"] == record_type)
                   and (not subtype or subtype in [f"{field}: {value}"
                        for field, value in raw["data"].items() if field.endswith("_type")])}
        texts = {key: raw["data"]["content"] for key, raw in records.items() if raw["data"].get("content", "").strip()}
        if prepare:
            await embeddings.prepare_index(texts, cancel=cancel, progress=progress)
        plan = embeddings.inspect_index(texts)
        indexed = {key: texts[key] for key in plan.hits}
        results = await embeddings.search(query, indexed, cancel=cancel) if indexed else []
        return {"indexed": len(indexed), "total": len(texts), "missing": len(plan.missing),
                "model": embeddings.config.model,
                "matches": [{"id": key, "score": score, "record": records[key],
                             "revision": view.revisions[key]} for key, score in results[:max(1, min(limit, 250))]]}

    def begin_batch(self, commit_id: str, *, visible_ids=None, new_tick: bool = False, cancel: Event | None = None) -> WriteBatch:
        """Capture a stable view. A new tick reserves an ID without accepting memory."""
        with self.lock:
            tick = self.current_tick_id
            if new_tick:
                self._reserved_tick_id = max(self._reserved_tick_id, self.current_tick_id) + 1
                tick = self._reserved_tick_id
            return WriteBatch(self, commit_id, self.snapshot(visible_ids), tick, cancel)

    def commit(self, batch: WriteBatch) -> CommitReceipt:
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
            consolidation_conflicts = [
                key for key in batch._consolidations
                if self.consolidations.get(key) != batch._initial_consolidations.get(key)
            ]
            for run_id in batch._consolidation_run_guards:
                before = {key: value for key, value in batch._initial_consolidations.items() if value.get("run_id") == run_id}
                current = {key: value for key, value in self.consolidations.items() if value.get("run_id") == run_id}
                if before != current:
                    consolidation_conflicts.append(run_id)
            if conflicts or run_conflicts or consolidation_conflicts:
                raise GhostConflict(
                    f"Stale memory revisions: {conflicts}. Stale run receipts: {run_conflicts}. "
                    f"Stale consolidation receipts: {consolidation_conflicts}."
                )
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
            data = Checkpoint(
                state=state, current_knoxel_id=self.current_knoxel_id,
                current_tick_id=max(self.current_tick_id, batch.tick_id), reserved_tick_id=self._reserved_tick_id,
                revisions=revisions, gwt_turns=gwt,
                applied_commits={**self.applied_commits, batch.commit_id: receipt},
                managed_runs={**self.managed_runs, **batch._runs},
                consolidations={**self.consolidations, **batch._consolidations},
            )
            # Run receipts can change without changing any record or tick.
            # Reuse validated records for that boundary. Rebuilding the entire
            # persona for every run event delays both control and public replies.
            metadata_only = (not batch._new and not batch._replacements
                             and batch._cycle is None and batch._gwt is None
                             and batch.tick_id == self.current_tick_id
                             and all(not record.embedding for record in self._all_knoxels.values()))
            if metadata_only:
                metadata = {"managed_runs": deepcopy(data.managed_runs),
                            "consolidations": deepcopy(data.consolidations),
                            "applied_commits": dict(data.applied_commits)}
                candidate = None
            else:
                candidate = self._restore(data, self.checkpoint)
            batch.check_cancelled()
            if self.checkpoint is not None:
                write_checkpoint(self.checkpoint, data, batch.check_cancelled)
            # All validation and I/O precede canonical installation.
            if metadata_only:
                for name, value in metadata.items():
                    setattr(self, name, value)
                batch._closed = True
                return receipt
            for name in ("ghost_config", "states", "_all_knoxels", "_knoxels_by_type", "_knoxels_by_subtype", "_knoxels_by_class",
                         "current_knoxel_id", "current_tick_id", "_reserved_tick_id", "revisions", "gwt_turns",
                         "applied_commits", "managed_runs", "consolidations", "previous_emb_model"):
                setattr(self, name, getattr(candidate, name))
            for knoxel in self._all_knoxels.values():
                knoxel._owner = self
            batch._closed = True
            return receipt

    def store_knoxel(self, knoxel: RecordBase) -> RecordBase:
        """Commit one trusted host input through the same batch boundary."""
        with self.lock:
            batch = self.begin_batch(uuid4().hex)
            record = batch.create(knoxel)
            self.commit(batch)
            return self.get_knoxel_by_id(record.id)

    def get_knoxel_by_id(self, knoxel_id: int) -> RecordBase | None:
        with self.lock:
            original = self._all_knoxels.get(knoxel_id)
            if original is None:
                return None
            record = _deserialize_knoxel(_serialize_knoxel(original))
            record._owner = self
            return record

    def add_knoxel(self, knoxel: RecordBase, generate_embedding: bool = True) -> RecordBase:
        raise NotImplementedError("Use store_knoxel or GhostWriteBatch.create to add memory.")

    def replace_knoxel(self, knoxel: RecordBase) -> RecordBase:
        raise NotImplementedError("Use GhostWriteBatch.replace to update memory.")

    def create_bash_environment(self, mode: str = "read", visible_ids=None):
        from pm.memory.bash_view import MemoryBashTarget
        return MemoryBashTarget(self, mode=mode, visible_ids=visible_ids)
