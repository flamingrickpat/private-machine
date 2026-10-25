"""Read-only index over one loaded memory checkpoint.

The inspector answers questions about what a companion actually stored. This
module reads the loaded ``MemoryStore`` and builds the small derived views
that the browser asks for: record counts per kind, record search, the records
that point at a record, and the shards that held a record in each cycle.

Nothing here writes. The store comes from ``MemoryStore.from_checkpoint`` and
is never saved. No model, no embedder, and no bus runs. Embeddings stay
unread: a plain checkpoint load has none, and these views do not need them.

Record views come from ``pm.context_render.story.record_view``, so a rendered
line in the inspector is the line that the cycle of that checkpoint showed.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from pm.constants import INSPECTOR_PAGE_SIZE
from pm.context_render.story import RecordView, record_view
from pm.memory.store import MemoryStore
from pm.records.base import RecordBase, persisted_type_name
from pm.records.cycle_records import StimulusGroup
from pm.records.cycle_state import CycleState
from pm.records.global_workspace import GwtTurnRecord
from pm.records.memory_records import MemoryClusterRecord
from pm.records.workspace import ShardKey, WorkspaceShard

#: Fields that hold a number that is not a record id. ``id`` is the record's
#: own id, and the two tick fields name a cycle, not a record.
NOT_RECORD_IDS = {"id", "tick_id", "sub_tick_id"}

#: Fields that hold record ids as one comma-separated string.
CSV_ID_FIELDS = {"included_event_ids", "included_cluster_ids"}


class RecordLabel(BaseModel):
    """One record, small enough to list. ``preview`` is the start of its content."""

    record_id: int
    kind: str = ""
    record_class: str = ""
    subtype: str = ""
    preview: str = ""
    content_chars: int = 0
    tick_id: int = 0
    timestamp: datetime | None = None
    in_checkpoint: bool = True


class Count(BaseModel):
    """A name and how often it occurs."""

    name: str
    count: int


class Relation(BaseModel):
    """One reference from one record to another. ``field`` names the field it came from."""

    field: str
    record_id: int
    target: RecordLabel


class ShardHit(BaseModel):
    """One cycle where a record sat in a workspace shard."""

    tick_id: int
    shard: ShardKey
    where: str = "item"


class TickRow(BaseModel):
    """One accepted cycle, with the size of the content it carried."""

    tick_id: int
    timestamp: datetime | None = None
    slot_id: str = ""
    summary: str = ""
    csm_items: int = 0
    shard_items: int = 0
    tail_items: int = 0
    shards: int = 0
    stimulus_ids: list[int] = Field(default_factory=list)
    token_estimate: int = 0
    llm_calls: int = 0


class RecordDetail(BaseModel):
    """Everything the index knows about one record."""

    label: RecordLabel
    content: str = ""
    fields: dict[str, Any] = Field(default_factory=dict)
    outgoing: list[Relation] = Field(default_factory=list)
    incoming: list[Relation] = Field(default_factory=list)
    shard_hits: list[ShardHit] = Field(default_factory=list)


def _preview(content: str, limit: int = 160) -> str:
    """Return one line of at most ``limit`` characters of ``content``."""
    text = " ".join(str(content or "").split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _csv_ids(value: str) -> list[int]:
    """Return the record ids of one comma-separated id string."""
    return [int(part) for part in (value or "").replace(" ", "").split(",") if part.lstrip("-").isdigit()]


def _hidden_field(name: str, value: Any) -> bool:
    """Return True for a field that the inspector must not show.

    Embeddings are vectors of hundreds of floats. They are noise in a text
    view, and the owner asked to keep them out of the reading surface.
    """
    if name.startswith("_") or name in {"embedding", "emotion_embedding"}:
        return True
    return isinstance(value, list) and len(value) > 32 and all(isinstance(item, float) for item in value)


def relation_edges(record: RecordBase) -> list[tuple[str, int]]:
    """Return the ``(field, record_id)`` pairs that one record points at.

    The record classes name their references in many different fields, and no
    shared interface declares them. This scanner reads the fields of the
    loaded model instead: an integer field that ends in ``_id``, a list of
    integers that ends in ``ids``, and the cluster id strings. The relation
    view then stays complete for record classes that nobody has listed here.
    Fields in ``NOT_RECORD_IDS`` hold numbers that name something else.
    """
    edges: list[tuple[str, int]] = []
    for name, value in record.__dict__.items():
        if name.startswith("_") or name in NOT_RECORD_IDS or isinstance(value, bool):
            continue
        if isinstance(value, int) and name.endswith("_id"):
            if value:
                edges.append((name, value))
        elif isinstance(value, (list, tuple)) and name.endswith("ids") and value:
            if all(isinstance(item, int) and not isinstance(item, bool) for item in value):
                edges += [(name, item) for item in value if item]
        elif isinstance(value, str) and name in CSV_ID_FIELDS:
            edges += [(name, item) for item in _csv_ids(value)]
    return edges


class MemoryIndex:
    """The derived views of one loaded checkpoint. Build it once per run.

    A checkpoint of this size (a few thousand records) scans in well under a
    second, so the index builds the relation maps and the shard membership
    map at startup and answers every request from memory.

    Record views and record labels build lazily and stay cached. A view holds
    a reference to the record's embedding list, so building every view up
    front would touch the whole checkpoint for records that nobody opens.
    """

    def __init__(self, store: MemoryStore) -> None:
        self.store = store
        self.companion_name = store.get_companion_name()
        self.user_name = store.get_user_name()
        self.entity_names = {entity.id: entity.content for entity in store.all_entities}
        self.user_entity_id = store.get_entity_id(self.user_name) or 0
        self.records: dict[int, RecordBase] = store.all_knoxels
        self.states: dict[int, CycleState] = {state.tick_id: state for state in store.states}
        self.turns: dict[int, GwtTurnRecord] = dict(store.gwt_turns)
        self._views: dict[int, RecordView] = {}
        self._labels: dict[int, RecordLabel] = {}
        self._outgoing: dict[int, list[tuple[str, int]]] = {}
        self._incoming: dict[int, list[tuple[str, int]]] = {}
        self._shards: dict[int, list[ShardHit]] = {}
        self._scan_relations()
        self._scan_shards()

    # ---------------------------------------------------------------- records

    def _scan_relations(self) -> None:
        """Fill the outgoing and incoming reference maps of the whole checkpoint."""
        for record_id, record in self.records.items():
            self._outgoing[record_id] = relation_edges(record)
            for field, target in self._outgoing[record_id]:
                self._incoming.setdefault(target, []).append((field, record_id))

    def _scan_shards(self) -> None:
        """Fill the map from a record id to the shards that held it."""
        for tick_id, state in self.states.items():
            for shard in self.shards(state):
                for item in shard.items:
                    self._shards.setdefault(item.record_id, []).append(
                        ShardHit(tick_id=tick_id, shard=shard.key, where="item"))
                for record_id in shard.tail_ids:
                    self._shards.setdefault(record_id, []).append(
                        ShardHit(tick_id=tick_id, shard=shard.key, where="tail"))

    @staticmethod
    def shards(state: CycleState) -> list[WorkspaceShard]:
        """Return the shards of one cycle in story order: self, entities, workflows."""
        workspace = state.workspace
        return [workspace.self_shard, *workspace.entity_shards.values(), *workspace.workflow_shards.values()]

    def label(self, record_id: int) -> RecordLabel:
        """Return the list row of one record. A missing record is labeled, not hidden."""
        cached = self._labels.get(record_id)
        if cached is not None:
            return cached
        record = self.records.get(record_id)
        if record is None:
            row = RecordLabel(record_id=record_id, kind="(absent)", content_chars=0,
                              preview="(no such record in this checkpoint)", in_checkpoint=False)
        else:
            subtype = getattr(record, "get_sub_type", None)
            row = RecordLabel(
                record_id=record_id,
                kind=persisted_type_name(type(record)),
                record_class=type(record).__name__,
                subtype=subtype().value if callable(subtype) else "",
                preview=_preview(record.content),
                content_chars=len(record.content or ""),
                tick_id=record.tick_id,
                timestamp=record.timestamp_world_begin,
            )
        self._labels[record_id] = row
        return row

    def labels(self, record_ids) -> list[RecordLabel]:
        """Return the list rows of many records, in the given order."""
        return [self.label(record_id) for record_id in record_ids]

    def view(self, record_id: int) -> RecordView:
        """Return the renderable view of one record, or a labeled stub when it is absent.

        The stub keeps the renderers intact. A shard of an old checkpoint can
        name a record id that the checkpoint does not hold, and the inspector
        then shows that gap inside the rendered text instead of failing to
        render the shard at all.
        """
        cached = self._views.get(record_id)
        if cached is not None:
            return cached
        record = self.records.get(record_id)
        view = (record_view(record) if record is not None else
                RecordView(record_id=record_id, kind="(absent)", content="(no such record in this checkpoint)"))
        self._views[record_id] = view
        return view

    def views(self, record_ids) -> dict[int, RecordView]:
        """Return the ``{record_id: view}`` map that the story renderers take."""
        return {record_id: self.view(record_id) for record_id in dict.fromkeys(record_ids)}

    def detail(self, record_id: int, *, limit: int = INSPECTOR_PAGE_SIZE) -> RecordDetail:
        """Return one record with its own fields, its relations, and its shard history."""
        record = self.records.get(record_id)
        fields: dict[str, Any] = {}
        if record is not None:
            for name, value in record.__dict__.items():
                if _hidden_field(name, value):
                    continue
                fields[name] = (value if isinstance(value, (int, float, str, bool, type(None), datetime))
                                else str(value))
        return RecordDetail(
            label=self.label(record_id),
            content=record.content if record is not None else "",
            fields=fields,
            outgoing=self.relations_out(record_id, limit),
            incoming=self.relations_in(record_id, limit),
            shard_hits=self.shard_hits(record_id),
        )

    def relations_out(self, record_id: int, limit: int = INSPECTOR_PAGE_SIZE) -> list[Relation]:
        """Return the records that this record points at."""
        return [Relation(field=field, record_id=target, target=self.label(target))
                for field, target in self._outgoing.get(record_id, [])][:limit]

    def relations_in(self, record_id: int, limit: int = INSPECTOR_PAGE_SIZE) -> list[Relation]:
        """Return the records that point at this record."""
        return [Relation(field=field, record_id=source, target=self.label(source))
                for field, source in self._incoming.get(record_id, [])][:limit]

    def relation_census(self) -> list[Count]:
        """Return how many edges each reference field carries in this checkpoint."""
        counts: Counter[str] = Counter()
        for edges in self._outgoing.values():
            for field, _ in edges:
                counts[field] += 1
        return [Count(name=name, count=count) for name, count in counts.most_common()]

    def shard_hits(self, record_id: int) -> list[ShardHit]:
        """Return the cycles and shards that held this record, oldest cycle first."""
        return sorted(self._shards.get(record_id, []), key=lambda hit: hit.tick_id)

    def matching(self, *, text: str = "", kind: str = "", subtype: str = "") -> list[int]:
        """Return the record ids that pass the filters, newest id first.

        ``text`` matches the content, the speaker, or the type name, without
        case. The scan is a plain pass over the checkpoint: at this size it
        costs a few tens of milliseconds, so there is no search index to keep
        in step with the data.
        """
        needle = text.strip().casefold()
        found = []
        for record_id in sorted(self.records, reverse=True):
            row = self.label(record_id)
            if kind and row.kind != kind:
                continue
            if subtype and row.subtype != subtype:
                continue
            if needle and needle not in self._searchable(record_id, row):
                continue
            found.append(record_id)
        return found

    def _searchable(self, record_id: int, row: RecordLabel) -> str:
        """Return the one lowercase text that a search may match for this record."""
        record = self.records[record_id]
        view = self.view(record_id)
        return " ".join(part for part in (row.kind, row.subtype, view.source, record.content) if part).casefold()

    def records_page(self, *, text: str = "", kind: str = "", subtype: str = "",
                     limit: int = INSPECTOR_PAGE_SIZE, offset: int = 0) -> list[RecordLabel]:
        """Return one page of the records that pass the filters."""
        return self.labels(self.matching(text=text, kind=kind, subtype=subtype)[offset:offset + limit])

    # ------------------------------------------------------------- statistics

    def counts(self) -> tuple[list[Count], list[Count]]:
        """Return the record counts per kind and per ``(kind, subtype)`` pair."""
        kinds: Counter[str] = Counter()
        pairs: Counter[tuple[str, str]] = Counter()
        for record_id in self.records:
            row = self.label(record_id)
            kinds[row.kind] += 1
            pairs[(row.kind, row.subtype)] += 1
        return ([Count(name=name, count=count) for name, count in kinds.most_common()],
                [Count(name=f"{kind} / {subtype or '-'}", count=count)
                 for (kind, subtype), count in pairs.most_common()])

    def tree(self) -> list[dict]:
        """Return the kind and subtype hierarchy with counts.

        The hierarchy uses the persisted type name and the record's own
        subtype. Both come from the records, so the inspector invents no
        classification of its own.
        """
        kinds: dict[str, Counter[str]] = {}
        for record_id in self.records:
            row = self.label(record_id)
            kinds.setdefault(row.kind, Counter())[row.subtype] += 1
        return [{"kind": kind, "count": sum(subtypes.values()),
                 "subtypes": [{"subtype": name or "-", "count": count}
                              for name, count in subtypes.most_common()]}
                for kind, subtypes in sorted(kinds.items(), key=lambda item: -sum(item[1].values()))]

    def clusters(self) -> list[dict]:
        """Return the episodic clusters with their hierarchy links.

        ``MemoryClusterRecord`` keeps its members as comma-separated id
        strings, so the inspector parses them here.
        """
        rows = []
        for record in self.store.all_episodic_memories:
            if not isinstance(record, MemoryClusterRecord):
                continue
            rows.append({
                "record_id": record.id,
                "cluster_type": record.cluster_type.value,
                "level": record.level,
                "temporal_key": record.temporal_key or "",
                "token": record.token,
                "content_chars": len(record.content or ""),
                "preview": _preview(record.content, 220),
                "events": _csv_ids(record.included_event_ids or ""),
                "clusters": _csv_ids(record.included_cluster_ids or ""),
                "timestamp": record.timestamp_world_begin,
            })
        return sorted(rows, key=lambda row: (-row["level"], row["record_id"]))

    # --------------------------------------------------------------- timeline

    def tick_ids(self) -> list[int]:
        """Return every tick that has a cycle state or a GWT turn, ascending."""
        return sorted(set(self.states) | set(self.turns))

    def tick_row(self, tick_id: int) -> TickRow:
        """Return the timeline row of one cycle."""
        state = self.states.get(tick_id)
        turn = self.turns.get(tick_id)
        shards = self.shards(state) if state is not None else []
        trace = turn.trace if turn is not None else None
        stamp = state.timestamp if state is not None else (turn.timestamp if turn is not None else None)
        return TickRow(
            tick_id=tick_id,
            timestamp=stamp,
            slot_id=turn.slot_id if turn is not None else "(no turn)",
            summary=_preview(turn.summary, 120) if turn is not None else "",
            csm_items=len(state.csm.items) if state is not None else 0,
            shard_items=sum(len(shard.items) for shard in shards),
            tail_items=sum(len(shard.tail_ids) for shard in shards),
            shards=len(shards),
            stimulus_ids=[record.id for record in self.stimulus_of(tick_id)],
            token_estimate=(trace.input_tokens_estimate + trace.output_tokens_estimate) if trace else 0,
            llm_calls=trace.llm_calls if trace else 0,
        )

    def timeline(self, *, limit: int = INSPECTOR_PAGE_SIZE, offset: int = 0,
                 slot: str = "") -> list[TickRow]:
        """Return one page of cycles, newest first."""
        rows = [self.tick_row(tick_id) for tick_id in reversed(self.tick_ids())]
        if slot:
            rows = [row for row in rows if row.slot_id == slot]
        return rows[offset:offset + limit]

    def stimulus_of(self, tick_id: int) -> list[RecordBase]:
        """Return the stimulus records that this cycle claimed, in group order.

        The cycle state stores only the ids, so the records come from the map.
        The renderer needs a view for each of them.
        """
        state = self.states.get(tick_id)
        if state is None:
            return []
        return [self.records[record_id] for record_id in state.claimed_ids() if record_id in self.records]

    def groups_of(self, tick_id: int) -> list[StimulusGroup]:
        """Return the stimulus groups of one cycle (the shard keys that cycle read)."""
        state = self.states.get(tick_id)
        return list(state.stimulus_groups) if state is not None else []

    def summary(self) -> dict:
        """Return the overview block: names, sizes, and the tick range."""
        kinds, pairs = self.counts()
        ticks = self.tick_ids()
        return {
            "companion_name": self.companion_name,
            "user_name": self.user_name,
            "checkpoint": str(self.store.checkpoint or ""),
            "records": len(self.records),
            "entities": len(self.entity_names),
            "cycles": len(ticks),
            "first_tick": ticks[0] if ticks else 0,
            "last_tick": ticks[-1] if ticks else 0,
            "kinds": [row.model_dump() for row in kinds],
            "subtypes": [row.model_dump() for row in pairs[:40]],
            "relations": [row.model_dump() for row in self.relation_census()],
            "character_card_chars": len(self.store.ghost_config.character_card_story or ""),
        }


__all__ = ["CSV_ID_FIELDS", "Count", "MemoryIndex", "NOT_RECORD_IDS", "RecordDetail", "RecordLabel", "Relation",
           "ShardHit", "TickRow", "relation_edges"]
