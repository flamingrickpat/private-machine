"""Import a chat log into a memory store as lived-looking, marked history.

Most companions exist only as a character card and a chat log. This module
reads such a log (``name,message,timestamp`` CSV) and writes the records that
the live loop reads, so the first live cycle after the import needs no
special case:

- an ``Entity`` for the user (Human) and one for the companion (Self);
- for each exchange (user messages, then companion replies) one write batch
  with a new tick: the user ``Dialogue`` features (provenance User, as
  ``UserInputProxy.submit`` writes them, but already resolved), the
  companion replies (provenance Self, addressed to the user, published), a
  neutral ``CycleState``, and a minimal v2 ``GwtTurnRecord``;
- after reconsolidation, a few synthetic records about the move
  (:func:`add_transfer_records`).

The import invents no thought, feeling, appraisal, or expectation. Every
record carries an :class:`~pm.records.import_marks.ImportMark`.
"""
from __future__ import annotations

import csv
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from pm.constants import (IMPORT_MESSAGE_SPACING_SECONDS, IMPORT_SESSION_GAP_SECONDS, IMPORT_SESSION_SPLIT_SECONDS,
                          IMPORT_SOURCE_PREFIX, TRANSFER_EVENT_SOURCE, TRANSFER_EVENT_TEXT, TRANSFER_NARRATIVE_LINE,
                          TRANSFER_SELF_MODEL_TEXT)
from pm.records.base import Provenance
from pm.records.cycle_records import StimulusGroup
from pm.records.cycle_state import CycleState
from pm.records.enums import EntityClass, FeatureType, InterlocusType, NarrativeTypes, SourceKind
from pm.records.global_workspace import GwtTurnRecord
from pm.records.import_marks import ImportMark, ImportOrigin, mark_metadata
from pm.records.lineage import RecordLineage
from pm.records.memory_records import Entity, Feature, Narrative, SemanticRecord
from pm.records.mental_state import FullMentalState, PersonalityProfile
from pm.records.taxonomy import EpistemicStatus, MemoryTaxonomy, RecordType
from pm.records.workspace import GlobalWorkspace, ShardKey
from pm.memory.store import MemoryStore
from pydantic import JsonValue

COLUMNS = ("name", "message", "timestamp")


class ChatRow(BaseModel):
    """One message of the log. ``row`` is the 1-based data row; the header is row 0."""

    row: int = 0
    speaker: Literal["user", "companion"] = "user"
    message: str = ""
    timestamp: datetime | None = None


class Exchange(BaseModel):
    """User messages and the companion replies that follow them.

    The first exchange of a log can have no user message, when the
    companion spoke first.
    """

    user_rows: list[ChatRow] = Field(default_factory=list)
    companion_rows: list[ChatRow] = Field(default_factory=list)

    @property
    def rows(self) -> list[ChatRow]:
        """Return the user rows first, then the companion rows."""
        return [*self.user_rows, *self.companion_rows]


def parse_timestamp(text: str, row: int) -> datetime:
    """Return a CSV timestamp as naive UTC.

    ISO 8601 (with ``T`` or a space, optional seconds, optional offset or
    ``Z``) is accepted. A time without an offset is taken as UTC. Another
    format raises ValueError naming the row.
    """
    value = text.strip()
    try:
        moment = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError:
        raise ValueError(f"Row {row}: timestamp {value!r} is not ISO 8601 or YYYY-MM-DD HH:MM[:SS].") from None
    if moment.tzinfo is not None:
        moment = moment.astimezone(timezone.utc).replace(tzinfo=None)
    return moment


def read_chat_csv(path: Path, *, companion_name: str, user_name: str) -> list[ChatRow]:
    """Read and check a chat log.

    The file is UTF-8 with or without a byte order mark. The columns are
    ``name`` and ``message``, and optionally ``timestamp``. Quoted fields can
    span lines. A name must be the companion or the user name (any case).
    The timestamps are either all empty or all set, and they must not
    decrease. Every violation raises ValueError naming the row.
    """
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        header = [name.strip().casefold() for name in reader.fieldnames or []]
        unknown = [name for name in header if name not in COLUMNS]
        if unknown or "name" not in header or "message" not in header:
            raise ValueError(f"The CSV header must be name,message[,timestamp]; found {reader.fieldnames}.")
        reader.fieldnames = header
        names = {companion_name.casefold(): "companion", user_name.casefold(): "user"}
        rows: list[ChatRow] = []
        for number, record in enumerate(reader, 1):
            name = (record["name"] or "").strip()
            if name.casefold() not in names:
                raise ValueError(f"Row {number}: name {name!r} is neither {companion_name!r} nor {user_name!r}.")
            message = (record["message"] or "").strip()
            if not message:
                raise ValueError(f"Row {number}: the message is empty.")
            stamp = (record["timestamp"] or "").strip() if "timestamp" in record else ""
            rows.append(ChatRow(row=number, speaker=names[name.casefold()], message=message,
                                timestamp=parse_timestamp(stamp, number) if stamp else None))
    stamped = [row.timestamp is not None for row in rows]
    if any(stamped) and not all(stamped):
        missing = next(row.row for row in rows if row.timestamp is None)
        raise ValueError(f"Row {missing}: the timestamp is empty. Give every row a timestamp or none.")
    for previous, row in zip(rows, rows[1:]):
        if row.timestamp is not None and row.timestamp < previous.timestamp:
            raise ValueError(f"Row {row.row}: timestamp {row.timestamp} is earlier than row {previous.row}.")
    return rows


def synthesize_times(rows: list[ChatRow], now: datetime) -> list[ChatRow]:
    """Give a log without timestamps ascending times that end before ``now``.

    The messages are ``IMPORT_MESSAGE_SPACING_SECONDS`` apart. The last one
    is ``IMPORT_SESSION_GAP_SECONDS`` before ``now``, so the first live
    message starts a new session. Rows with timestamps stay unchanged.
    """
    if not rows or rows[0].timestamp is not None:
        return rows
    last = now - timedelta(seconds=IMPORT_SESSION_GAP_SECONDS)
    count = len(rows)
    return [row.model_copy(update={"timestamp": last - timedelta(seconds=IMPORT_MESSAGE_SPACING_SECONDS * (count - 1 - index))})
            for index, row in enumerate(rows)]


def group_exchanges(rows: list[ChatRow]) -> list[Exchange]:
    """Group the rows: each exchange is user messages, then companion replies.

    A user message after a companion reply starts the next exchange.
    """
    exchanges: list[Exchange] = []
    for row in rows:
        if not exchanges or (row.speaker == "user" and exchanges[-1].companion_rows):
            exchanges.append(Exchange())
        if row.speaker == "user":
            exchanges[-1].user_rows.append(row)
        else:
            exchanges[-1].companion_rows.append(row)
    return exchanges


def session_numbers(rows: list[ChatRow]) -> list[int]:
    """Return a session number for each row. A silence of ``IMPORT_SESSION_SPLIT_SECONDS`` starts a new session."""
    numbers = [0] * len(rows)
    for index in range(1, len(rows)):
        gap = (rows[index].timestamp - rows[index - 1].timestamp).total_seconds()
        numbers[index] = numbers[index - 1] + (gap >= IMPORT_SESSION_SPLIT_SECONDS)
    return numbers


def file_sha256(path: Path) -> str:
    """Return the SHA-256 of the file's exact bytes, as a lowercase hex string."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ensure_persona_entities(store: MemoryStore, mark: ImportMark) -> tuple[int, int]:
    """Create the user and companion entities when they are missing.

    Returns ``(user_entity_id, companion_entity_id)``. The live loop finds
    them by name, and the shards and narratives name them by id.
    """
    persona = store.ghost_config
    with store.lock:
        missing = [(name, kind) for name, kind in ((persona.user_name, EntityClass.Human),
                                                   (persona.companion_name, EntityClass.Self))
                   if store.get_entity_id(name) is None]
    if missing:
        batch = store.begin_batch(f"{mark.import_id or 'character'}-entities")
        for name, kind in missing:
            batch.create(Entity(content=name, aliases=[name], entity_class=kind,
                                timestamp_creation=mark.imported_at, timestamp_world_begin=mark.imported_at,
                                timestamp_world_end=mark.imported_at, metadata=mark_metadata(mark)))
        store.commit(batch)
    with store.lock:
        return store.get_entity_id(persona.user_name), store.get_entity_id(persona.companion_name)


class ImportedExchanges(BaseModel):
    """The ids that :func:`import_exchanges` created."""

    ticks: list[int] = Field(default_factory=list)
    feature_ids: list[int] = Field(default_factory=list)


def import_exchanges(store: MemoryStore, exchanges: list[Exchange], *, import_id: str, source_sha256: str,
                     personality: PersonalityProfile, imported_at: datetime) -> ImportedExchanges:
    """Write each exchange in its own batch with a new tick.

    The cycle state of an exchange has the neutral mental state: the
    persona's personality and the default readouts. Its stimulus group holds
    the user messages. The turn record names the exchange records and has no
    summary, experience, or decision, because no cognition ran.
    """
    persona = store.ghost_config
    user_id, _ = ensure_persona_entities(store, ImportMark(origin=ImportOrigin.CharacterInit, import_id=import_id,
                                                           synthetic=True, imported_at=imported_at))
    rows = [row for exchange in exchanges for row in exchange.rows]
    sessions = dict(zip((row.row for row in rows), session_numbers(rows)))
    user_key = ShardKey(kind="entity", ref=str(user_id))
    result = ImportedExchanges()
    for index, exchange in enumerate(exchanges):
        correlation = f"{import_id}-x{index:05d}"
        batch = store.begin_batch(correlation, new_tick=True)

        def feature(row: ChatRow, source: str, kind: SourceKind, metadata: dict[str, JsonValue],
                    addressee: int = 0) -> Feature:
            """Create one imported dialogue feature for a row of this exchange."""
            mark = ImportMark(origin=ImportOrigin.ChatLog, import_id=import_id, source_sha256=source_sha256,
                              source_row=row.row, imported_at=imported_at)
            return batch.create(Feature(
                content=row.message, feature_type=FeatureType.Dialogue, interlocus=InterlocusType.Public,
                causal=True, override_source=source, addressee_entity_id=addressee,
                timestamp_creation=row.timestamp, timestamp_world_begin=row.timestamp,
                timestamp_world_end=row.timestamp,
                provenance=Provenance(source_kind=kind, source_ref=f"{IMPORT_SOURCE_PREFIX}:{import_id}:row{row.row}"),
                metadata={"memory_kind": "utterance", "executed_at": row.timestamp.isoformat(),
                          "session": f"{import_id}-s{sessions[row.row]}", **metadata, **mark_metadata(mark)}))

        users = [feature(row, persona.user_name, SourceKind.User,
                         {"exchange_correlation": correlation, "exchange_status": "resolved"})
                 for row in exchange.user_rows]
        replies = [feature(row, persona.companion_name, SourceKind.Self,
                           {"factual_promotion": False, "reply_correlation": correlation,
                            "publication_status": "published", "responding_to_ids": [item.id for item in users]},
                           addressee=user_id)
                   for row in exchange.companion_rows]
        if replies:
            for item in users:
                item.metadata["public_reply_id"] = replies[0].id
                batch.revise(item)
        ids = [item.id for item in [*users, *replies]]
        moment = exchange.rows[-1].timestamp
        workspace = GlobalWorkspace()
        workspace.ensure_shard(user_key).tail_ids = list(ids)
        state = CycleState(
            tick_id=batch.tick_id, timestamp=moment,
            mental_state=FullMentalState(state_personality=personality.model_copy()),
            workspace=workspace,
            stimulus_groups=[StimulusGroup(key=user_key, stimulus_ids=[item.id for item in users],
                                           correlation_id=correlation)] if users else [])
        turn = GwtTurnRecord(tick_id=batch.tick_id, slot_id="main", stimulus_ids=[item.id for item in users],
                             knoxel_ids=ids, timestamp=moment)
        batch.set_gwt(turn, state)
        store.commit(batch)
        result.ticks.append(state.tick_id)
        result.feature_ids += ids
    return result


def add_transfer_records(store: MemoryStore, *, import_id: str, imported_at: datetime) -> list[int]:
    """Add the synthetic records that tell the companion about the move.

    1. A ``WorldEvent`` feature at the import time: the transfer happened.
    2. A ``SemanticRecord`` about the companion (the self model).
    3. When the companion has a ``SelfImage`` narrative, a new version of
       its newest version with one more line.

    All three carry a synthetic ``ImportMark`` and neutral wording.
    Returns the new record ids.
    """
    persona = store.ghost_config
    values = {"companion_name": persona.companion_name, "user_name": persona.user_name,
              "date": imported_at.strftime("%Y-%m-%d")}
    mark = mark_metadata(ImportMark(origin=ImportOrigin.Transfer, import_id=import_id, synthetic=True,
                                    imported_at=imported_at))
    provenance = Provenance(source_kind=SourceKind.System, source_ref=f"{IMPORT_SOURCE_PREFIX}:{import_id}")
    with store.lock:
        companion_id = store.get_entity_id(persona.companion_name)
        images = [item for item in store.all_narratives
                  if item.narrative_type == NarrativeTypes.SelfImage and item.entity_id == companion_id]
    batch = store.begin_batch(f"{import_id}-transfer")
    event = batch.create(Feature(
        content=TRANSFER_EVENT_TEXT.format(**values), feature_type=FeatureType.WorldEvent, causal=True,
        override_source=TRANSFER_EVENT_SOURCE, timestamp_creation=imported_at, timestamp_world_begin=imported_at,
        timestamp_world_end=imported_at, provenance=provenance,
        metadata={"memory_kind": "executed_event", "executed_at": imported_at.isoformat(), **mark}))
    text = TRANSFER_SELF_MODEL_TEXT.format(**values)
    first = text.split(". ")[0] + "."
    record = batch.create(SemanticRecord(
        content=text, subject_entity_id=companion_id, record_type=RecordType.Fact, object_kind="history",
        object_value="memory transfer into pm", epistemic_status=EpistemicStatus.Observed, confidence=1.0,
        topic_card="memory_transfer", essence=first, record_id=f"mem_{imported_at.strftime('%Y%m%dT%H')}_transfer",
        subject_label=persona.companion_name, importance=0.8, source_knoxel_ids=[event.id],
        source_method="memory_transfer", timestamp_creation=imported_at, timestamp_world_begin=imported_at,
        timestamp_world_end=imported_at, provenance=provenance,
        metadata={"dedup_key": MemoryTaxonomy.dedup_key(companion_id, first), **mark}))
    record.semantic_record_id = record.id
    batch.revise(record)
    if images:
        newest = max(images, key=lambda item: (item.last_refined_with_tick or -1, item.timestamp_world_begin, item.id))
        sources = newest.lineage.source_event_ids if newest.lineage is not None else []
        batch.create(Narrative(
            content=f"{newest.content.rstrip()}\n\n{TRANSFER_NARRATIVE_LINE.format(**values)}",
            narrative_type=NarrativeTypes.SelfImage, entity_id=companion_id,
            last_refined_with_tick=newest.last_refined_with_tick,
            timestamp_world_begin=newest.timestamp_world_begin, timestamp_world_end=newest.timestamp_world_end,
            provenance=provenance, metadata=dict(mark),
            lineage=RecordLineage(episode_id=f"{IMPORT_SOURCE_PREFIX}:{import_id}", operation="memory_transfer",
                                  source_event_ids=[*sources, event.id], parent_record_ids=[newest.id])))
    return list(store.commit(batch).added_ids)


__all__ = ["COLUMNS", "ChatRow", "Exchange", "ImportedExchanges", "add_transfer_records", "ensure_persona_entities",
           "file_sha256", "group_exchanges", "import_exchanges", "parse_timestamp", "read_chat_csv",
           "session_numbers", "synthesize_times"]
