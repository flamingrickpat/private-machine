"""MemoryStore ownership, scoped snapshots, and atomic accepted memory.

Invalid proposals and revision conflicts reject a complete batch. A failed
checkpoint leaves accepted memory unchanged. Errors reach the debugger.
No model or network call runs under the commit lock. Optional checkpoint=None
means an explicit in-memory session, not durable storage.
"""
from __future__ import annotations


import hashlib
from collections.abc import Callable
from pathlib import Path
import json
import pickle
from threading import Event, RLock
from uuid import uuid4

from pm.memory.snapshot import GhostCancelled, GhostConflict, Snapshot
from pm.memory.write_batch import WriteBatch
from pm.memory.record_index import RecordIndex
from pm.memory.references import check_references
from pm.records.cycle_state import CycleState, CycleStateModel, PersonaConfig
from pm.records.global_workspace import GwtTurnRecord
from pm.records.base import RecordBase
from pm.memory.yaml_codec import ghost_state_to_dict, load_ghost_state_from_yaml
from pm.memory.vector_index import VectorIndex
from pm.memory.embeddings import EmbeddingService, run_embedding_work
from pm.memory.checkpoint import CommitReceipt, Checkpoint, read_checkpoint, write_checkpoint
from pm.records.media import MEDIA_KEY, RecordImage, merge_alt_text, metadata_media


class MemoryStore(RecordIndex):
    """Own one central record map. Reads share records and writes use the store lock."""
    def __init__(self, config: PersonaConfig, *, checkpoint: Path | None = None):
        super().__init__(None, config)
        self.lock = RLock()
        self.embedding_index = VectorIndex()
        self.embeddings: EmbeddingService | None = None
        self.checkpoint = Path(checkpoint) if checkpoint is not None else None
        if self.checkpoint is not None and self.checkpoint.suffix.lower() in {".json", ".yaml", ".yml"}:
            self.checkpoint = self.checkpoint.with_suffix(".pickle")
        self.gwt_turns: dict[int, GwtTurnRecord] = {}
        self.revisions: dict[int, int] = {}
        self.applied_commits: dict[str, CommitReceipt] = {}
        self.managed_runs: dict[str, dict] = {}
        self.consolidations: dict[str, dict] = {}
        self.previous_emb_model = ""
        self._reserved_tick_id = 0

    @classmethod
    def load(cls, path: str | Path, *, checkpoint: Path | None = None) -> "MemoryStore":
        """Load pickle, JSON, or a legacy YAML seed. Save future changes with pickle."""
        from pm.memory.yaml_codec import ghost_state_from_dict
        path = Path(path)
        if path.suffix.lower() in {".yaml", ".yml"}:
            return cls.from_yaml(path, checkpoint=checkpoint)
        data = path.read_bytes()
        if data.startswith(b"\x80"):
            loaded = pickle.loads(data)
            if isinstance(loaded, Checkpoint):
                return cls._restore(loaded, checkpoint or path.with_suffix(".pickle"))
            return cls.from_state(loaded, checkpoint=checkpoint or path.with_suffix(".pickle"))
        loaded = json.loads(data)
        if "schema_id" in loaded:
            return cls._restore(Checkpoint.model_validate(loaded), checkpoint or path.with_suffix(".pickle"))
        return cls.from_state(ghost_state_from_dict(loaded), checkpoint=checkpoint)

    @classmethod
    def from_state(cls, state: CycleStateModel, *, checkpoint: Path | None = None) -> "MemoryStore":
        config = PersonaConfig.model_validate(state.model_dump(exclude={"knoxels", "cycles", "previous_emb_model"}))
        ghost = cls(config, checkpoint=checkpoint)
        ghost.states = state.cycles
        ghost._all_knoxels = {knoxel.id: knoxel for knoxel in state.knoxels}
        if len(ghost._all_knoxels) != len(state.knoxels):
            raise ValueError("The seed contains duplicate knoxel IDs.")
        ghost.previous_emb_model = state.previous_emb_model
        ghost.current_tick_id = max([0] + [cycle.tick_id for cycle in state.cycles] + [k.tick_id for k in state.knoxels])
        ghost._reserved_tick_id = ghost.current_tick_id
        ghost.revisions = {record_id: 1 for record_id in ghost._all_knoxels}
        ghost.rebuild_knoxel_indexes()
        ghost.embedding_index.rebuild(state.knoxels)
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
        ghost = cls.from_state(data.state, checkpoint=path)
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
        ghost.gwt_turns = data.gwt_turns
        ghost.applied_commits = dict(data.applied_commits)
        ghost.managed_runs = data.managed_runs
        ghost.consolidations = data.consolidations
        return ghost

    def state(self, records: dict[int, RecordBase] | None = None, cycles: list[CycleState] | None = None) -> CycleStateModel:
        """Return typed state with shared records. This method copies no record values."""
        return CycleStateModel.model_construct(
            **self.ghost_config.model_dump(), previous_emb_model=self.previous_emb_model,
            cycles=self.states if cycles is None else cycles,
            knoxels=list((self._all_knoxels if records is None else records).values()),
        )

    def state_dict(self) -> dict:
        """Render an explicit JSON export. Normal commits use typed pickle state."""
        with self.lock:
            return ghost_state_to_dict(self.state())

    def snapshot(self, visible_ids=None) -> Snapshot:
        """Capture membership and revisions while sharing the central record instances."""
        with self.lock:
            ids = self._all_knoxels.keys() if visible_ids is None else visible_ids
            records = {key: self._all_knoxels[key] for key in ids}
            gwt = {tick: record for tick, record in self.gwt_turns.items()
                   if set(record.knoxel_ids).issubset(records)}
            return Snapshot(records, gwt, {key: self.revisions[key] for key in records}, self.embedding_index)

    def index_embedding(self, record: RecordBase) -> None:
        """Index a vector update only when this record is the accepted central instance."""
        if self._all_knoxels.get(record.id) is not record:
            return
        with self.lock:
            if self._all_knoxels.get(record.id) is record:
                self.embedding_index.update(record)

    def check_embeddings(self, embeddings: EmbeddingService) -> dict[str, float]:
        """Probe the embedding endpoint before this store writes any vector.

        Returns the quality report. A degenerate endpoint returns vectors that
        all point nearly the same way. Those vectors are durable, and every
        merge, retrieval, coalition, and expectation rule reads them, so the
        damage is silent and permanent. This method raises
        ``EmbeddingQualityError`` instead. Record embeddings are the shared
        vector space of the whole store, so one probe per service is enough.
        """
        return run_embedding_work(embeddings.smoke_test())

    def rebuild_embeddings(self, embeddings: EmbeddingService,
                           progress: Callable[[int, int], None] | None = None) -> int:
        """Recompute every stored vector with the current model, then save.

        This is the ``--embed`` path. It drops all vectors first, so no vector
        from the old model survives the run. The endpoint probe runs before
        anything is dropped: a bad endpoint leaves the checkpoint untouched.
        Records keep their IDs, so links, states, and story text do not change.
        Returns the number of records that carry a vector afterwards.
        """
        self.embeddings = embeddings
        self.check_embeddings(embeddings)
        records = self.snapshot().all_knoxels
        for record in records.values():
            record.embedding = []
        run_embedding_work(embeddings.prepare_index(records, progress=progress))
        self.previous_emb_model = embeddings.config.model
        embedded = sum(1 for record in records.values() if record.embedding)
        if self.checkpoint is not None:
            self.save()
        return embedded

    def initialize_embeddings(self, embeddings: EmbeddingService) -> None:
        """Embed all missing records on load. Save once when vectors or the format change.

        The endpoint probe runs first, so a store never absorbs vectors from a
        broken encoder. A model change or a dimension change drops the old
        vectors, and the vector index rebuilds itself at the new size.
        """
        self.embeddings = embeddings
        self.check_embeddings(embeddings)
        records = self.snapshot().all_knoxels
        changed_model = bool(self.previous_emb_model and self.previous_emb_model != embeddings.config.model)
        for record in records.values():
            if changed_model or len(record.embedding) != embeddings.dimensions:
                record.embedding = []
        missing = any(not record.embedding for record in records.values())
        run_embedding_work(embeddings.prepare_index(records))
        self.previous_emb_model = embeddings.config.model
        if self.checkpoint is not None:
            migrate = self.checkpoint.suffix.lower() not in {".pkl", ".pickle"}
            if migrate:
                self.checkpoint = self.checkpoint.with_suffix(".pickle")
            if missing or changed_model or migrate or not self.checkpoint.exists():
                self.save()

    def checkpoint_data(self) -> Checkpoint:
        """Collect the full typed store for a pickle save."""
        return Checkpoint(
            state=self.state(), current_knoxel_id=self.current_knoxel_id,
            current_tick_id=self.current_tick_id, reserved_tick_id=self._reserved_tick_id,
            revisions=self.revisions, gwt_turns=self.gwt_turns,
            applied_commits=self.applied_commits, managed_runs=self.managed_runs,
            consolidations=self.consolidations,
        )

    def save(self) -> None:
        """Save all records and vectors. Filesystem errors propagate."""
        with self.lock:
            if self.checkpoint is not None:
                write_checkpoint(self.checkpoint, self.checkpoint_data(), lambda: None)

    def backup(self) -> Path | None:
        """Save accepted memory beside the checkpoint and return its backup path.

        Main cognition calls this method after a successful cycle commit.
        The commit lock keeps all fields at one stable revision. Backups have
        no retention limit. I/O errors propagate without undoing accepted
        memory. An ephemeral store returns None and writes no file.
        """
        from datetime import datetime, timezone
        from uuid import uuid4
        import logging

        if self.checkpoint is None:
            return None
        with self.lock:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            path = self.checkpoint.with_name(f"ghost_{self.current_tick_id}_{stamp}_{uuid4().hex[:8]}.pickle")
            data = self.checkpoint_data()
            write_checkpoint(path, data, lambda: None)
        logging.getLogger(__name__).info("Memory backup tick=%s path=%s", data.current_tick_id, path)
        return path

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
        """Search shared records with FAISS after embedding any missing vectors.

        Type and subtype filters apply before ranking. Pass the permitted
        record IDs. The legacy ``prepare`` argument has no effect.
        """
        view = self.snapshot(visible_ids)
        records = {key: record for key, record in view.all_knoxels.items()
                   if (not record_type or record.type.value == record_type)
                   and (not subtype or subtype in [f"{field}: {getattr(record, field)}"
                        for field in type(record).model_fields if field.endswith("_type")])}
        missing = any(not record.embedding for record in records.values())
        results = await embeddings.search(query, records, index=view.embedding_index, cancel=cancel)
        if missing and self.checkpoint is not None:
            self.save()
        return {"indexed": len(records), "total": len(records), "missing": 0,
                "model": embeddings.config.model,
                "matches": [{"id": key, "score": score, "record": records[key].as_payload(include_embedding=False),
                             "revision": view.revisions[key]} for key, score in results[:max(1, min(limit, 250))]]}

    def gwt_turns_of(self, slot_id: str, limit: int = 8) -> list[GwtTurnRecord]:
        """Return the newest published turns of one loop, newest first.

        Both conscious loops publish into the same keyed tick map. The slot id
        of a turn names its loop (``main`` or ``inner``), so one loop can read
        the latest turn of the other without a second store.
        """
        with self.lock:
            turns = [record for record in self.gwt_turns.values() if record.slot_id == slot_id]
        turns.sort(key=lambda record: record.tick_id, reverse=True)
        return turns[:max(1, limit)]

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
        if self.embeddings is not None and batch.commit_id not in self.applied_commits:
            batch.check_cancelled()
            run_embedding_work(self.embeddings.prepare_index(
                {**batch._new, **batch._replacements}, cancel=batch.cancel))
        with self.lock:
            if batch._ghost is not self:
                raise ValueError("The batch belongs to another Ghost.")
            if batch.commit_id in self.applied_commits:
                batch._closed = True
                return self.applied_commits[batch.commit_id]
            batch.check_cancelled()
            conflicts = [record_id for record_id, revision in batch._expected.items() if self.revisions[record_id] != revision]
            family_conflicts = [key for key, value, expected in batch._metadata_guards
                                if {record.id: self.revisions[record.id]
                                    for record in self._all_knoxels.values()
                                    if record.metadata.get(key) == value} != expected]
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
            if conflicts or run_conflicts or consolidation_conflicts or family_conflicts:
                raise GhostConflict(
                    f"Stale memory revisions: {conflicts}. Stale run receipts: {run_conflicts}. "
                    f"Stale consolidation receipts: {consolidation_conflicts}. Stale record families: {family_conflicts}."
                )
            allowed_ids = set(batch._snapshot._records) | set(batch._new)
            for record in (*batch._new.values(), *batch._replacements.values()):
                check_references(record, allowed_ids)
            if batch._gwt is not None:
                if batch.tick_id in self.gwt_turns:
                    raise GhostConflict(f"GWT tick {batch.tick_id} is already committed.")
                check_references(batch._gwt, allowed_ids)
            if batch._cycle is not None:
                if any(cycle.tick_id == batch.tick_id for cycle in self.states):
                    raise GhostConflict(f"Cycle tick {batch.tick_id} is already committed.")
                check_references(batch._cycle, allowed_ids)

            records = {**self._all_knoxels, **batch._new, **batch._replacements}
            for record_id, replacement in batch._replacements.items():
                merge_alt_text(replacement.metadata, self._all_knoxels[record_id].metadata)
            cycles = self.states if batch._cycle is None else [*self.states, batch._cycle]
            revisions = dict(self.revisions)
            for record_id in (*batch._new, *batch._replacements):
                revisions[record_id] = revisions.get(record_id, 0) + 1
            receipt = CommitReceipt(commit_id=batch.commit_id, added_ids=tuple(batch._new),
                                    updated_ids=tuple(batch._replacements), tick_id=batch.tick_id)
            gwt = dict(self.gwt_turns)
            if batch._gwt is not None:
                gwt[batch.tick_id] = batch._gwt
            data = Checkpoint(
                state=self.state(records, cycles), current_knoxel_id=self.current_knoxel_id,
                current_tick_id=max(self.current_tick_id, batch.tick_id), reserved_tick_id=self._reserved_tick_id,
                revisions=revisions, gwt_turns=gwt,
                applied_commits={**self.applied_commits, batch.commit_id: receipt},
                managed_runs={**self.managed_runs, **batch._runs},
                consolidations={**self.consolidations, **batch._consolidations},
            )
            batch.check_cancelled()
            if self.checkpoint is not None:
                write_checkpoint(self.checkpoint, data, batch.check_cancelled)
            # Save before installation. A failed file replacement leaves proposals private.
            for record_id, record in {**batch._new, **batch._replacements}.items():
                previous = self._all_knoxels.get(record_id)
                if previous is not None:
                    self._deindex_knoxel(previous)
                record._owner = self
                self._all_knoxels[record_id] = record
                self._index_knoxel(record)
                self.embedding_index.update(record)
            self.states = cycles
            self.current_tick_id = data.current_tick_id
            self.revisions = revisions
            self.gwt_turns = gwt
            self.applied_commits = data.applied_commits
            self.managed_runs = data.managed_runs
            self.consolidations = data.consolidations
            batch._closed = True
            import logging
            logging.getLogger(__name__).debug("Memory commit id=%s tick=%s added=%s updated=%s", receipt.commit_id,
                                              receipt.tick_id, len(receipt.added_ids), len(receipt.updated_ids))
            return receipt

    def find_image(self, image_id: str) -> RecordImage | None:
        """Return the image with this id from the newest accepted record that holds it.

        Image ids are content addressed, so every record that holds the id
        holds the same bytes. The newest record has the newest alt text.
        Returns None when no accepted record holds the image.
        """
        with self.lock:
            for record_id in sorted(self._all_knoxels, reverse=True):
                metadata = self._all_knoxels[record_id].metadata
                if MEDIA_KEY not in metadata:
                    continue
                for image in metadata_media(metadata).images:
                    if image.id == image_id:
                        return image
        return None

    def annotate_alt_text(self, image_id: str, alt_text: str, source: str) -> list[int]:
        """Store an alt text in every accepted record that holds the image, and save.

        The alt text is a derived description, like an embedding: it does not
        change what the record says. The write therefore keeps the record
        revisions, so open write batches that read these records stay valid.
        Their later replacements keep this alt text (see ``commit``). An
        image that already has an alt text keeps it. Returns the changed ids.
        """
        changed = []
        with self.lock:
            for record_id, record in self._all_knoxels.items():
                if MEDIA_KEY not in record.metadata:
                    continue
                media = metadata_media(record.metadata)
                hit = False
                for image in media.images:
                    if image.id == image_id and not image.alt_text:
                        image.alt_text, image.alt_text_source = alt_text, source
                        hit = True
                if hit:
                    record.metadata = {**record.metadata, MEDIA_KEY: media.model_dump(mode="json")}
                    changed.append(record_id)
            if changed and self.checkpoint is not None:
                self.save()
        return changed

    def store_knoxel(self, knoxel: RecordBase) -> RecordBase:
        """Commit one trusted host input through the same batch boundary."""
        batch = self.begin_batch(uuid4().hex)
        record = batch.create(knoxel)
        self.commit(batch)
        return self.get_knoxel_by_id(record.id)

    def get_knoxel_by_id(self, knoxel_id: int) -> RecordBase | None:
        with self.lock:
            return self._all_knoxels.get(knoxel_id)

    def add_knoxel(self, knoxel: RecordBase, generate_embedding: bool = True) -> RecordBase:
        raise NotImplementedError("Use store_knoxel or GhostWriteBatch.create to add memory.")

    def replace_knoxel(self, knoxel: RecordBase) -> RecordBase:
        raise NotImplementedError("Use GhostWriteBatch.replace to update memory.")

    def create_bash_environment(self, mode: str = "read", visible_ids=None):
        from pm.memory.bash_view import MemoryBashTarget
        return MemoryBashTarget(self, mode=mode, visible_ids=visible_ids)
