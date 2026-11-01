"""Present captured records, conversation rows, and GWT cycles for review.

The simulation captures already serialize every record, its revisions, and the
GWT turn of each cycle. This module reads those packages and calls the shared
formatter in ``pm.introspection.record_format``. It does not load the runtime
store, reconstruct missing records, or score an output.

Owns: the record, conversation, and cycle indexes for one participant attempt.
Does not own: record capture or record presentation rules.
See the package overview in ``pm.tools.timeline``.
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from pm.introspection.record_format import RecordLink, present_cycle, record_card
from pm.tools.timeline.evidence import EvidenceError, normalize_time, read_json_object


class RecordCard(BaseModel):
    """One captured record as the shared review formatter presents it."""

    model_config = ConfigDict(extra="ignore")

    id: int
    tick: int = 0
    kind: str = ""
    role: str = ""
    text: str = ""
    source: str = ""
    operation: str = ""
    codelet: str = ""
    selected: JsonValue = None
    status: str = ""
    fixture: bool = False
    details: dict[str, JsonValue] = Field(default_factory=dict[str, JsonValue])
    route: str = ""
    links: list[RecordLink] = Field(default_factory=list[RecordLink])
    record: JsonValue = None


class RecordSummary(BaseModel):
    """One record row for a list or a cycle card, without the full payload."""

    id: int
    tick: int = 0
    kind: str = ""
    role: str = ""
    text: str = ""
    source: str = ""
    operation: str = ""
    codelet: str = ""
    status: str = ""
    fixture: bool = False


class ConversationRow(BaseModel):
    """One public dialogue record in the order the capture recorded it."""

    record_id: int
    participant: str
    speaker: str
    text: str
    at: str = ""
    wall_at: str = ""
    delivery_id: str = ""
    feature_type: str = ""


class CycleSummary(BaseModel):
    """One captured GWT turn, small enough to list."""

    participant: str
    tick: int
    at: str = ""
    wall_at: str = ""
    summary: str = ""
    slot_id: str = ""
    member_count: int = 0
    stimulus_count: int = 0
    codelet_runs: int = 0


class CycleDetail(BaseModel):
    """The full cycle layout plus the cards its rows refer to."""

    participant: str
    tick: int
    presentation: dict[str, JsonValue]
    cards: dict[int, RecordSummary] = Field(default_factory=dict[int, RecordSummary])


class CapturedDialogue(BaseModel):
    """Public dialogue fields read from the serialized evidence boundary."""

    content: str
    provenance: dict[str, JsonValue] = Field(default_factory=dict[str, JsonValue])
    metadata: dict[str, JsonValue] = Field(default_factory=dict[str, JsonValue])
    override_source: str | None = None
    timestamp_world_begin: JsonValue = None
    timestamp_creation: JsonValue = None
    feature_type: str


class CapturedTrace(BaseModel):
    """Captured codelet runs whose count appears in a cycle row."""

    codelet_runs: list[JsonValue] = Field(default_factory=list[JsonValue])


class CapturedGwt(BaseModel):
    """Recorded GWT fields needed by the cycle index."""

    tick_id: int
    timestamp: JsonValue = None
    summary: str = ""
    slot_id: JsonValue = ""
    trace: CapturedTrace = Field(default_factory=CapturedTrace)
    stimulus_ids: list[int] = Field(default_factory=list[int])
    knoxel_ids: list[int] = Field(default_factory=list[int])


class CapturedCycleState(BaseModel):
    """World timestamp from a captured cognitive cycle."""

    timestamp: JsonValue = None


class CapturedCyclePackage(BaseModel):
    """Typed cycle projection; the original package remains available intact."""

    gwt: CapturedGwt
    cycle: CapturedCycleState
    record_ids: list[int]


def _package_records(package: dict[str, JsonValue]) -> dict[int, dict[str, JsonValue]]:
    """Index the serialized records of one package by knoxel ID."""
    rows: dict[int, dict[str, JsonValue]] = {}
    records = package.get("records")
    if not isinstance(records, list):
        return rows
    for wrapped in records:
        if not isinstance(wrapped, dict):
            continue
        data = wrapped.get("data")
        if not isinstance(data, dict):
            raise EvidenceError("A captured record has no data object.")
        record_id = data["id"]
        if not isinstance(record_id, int):
            raise EvidenceError("A captured record ID must be an integer.")
        rows[record_id] = wrapped
    return rows


def _merge_records(target: dict[int, dict[str, JsonValue]], source: dict[int, dict[str, JsonValue]]) -> None:
    """Merge one package into the record map and reject conflicting copies."""
    for key, wrapped in source.items():
        previous = target.get(key)
        if previous is not None and previous != wrapped:
            raise EvidenceError(f"Two different captured versions of knoxel {key} exist in this attempt.")
        target[key] = wrapped


class RecordIndex:
    """Captured records, public conversation, and cycles for one participant."""

    def __init__(self, participant: str, evidence_root: Path) -> None:
        self.participant: str = participant
        self.evidence_root: Path = evidence_root
        self._wrapped: dict[int, dict[str, JsonValue]] = {}
        self._cards: dict[int, RecordCard] = {}
        self.chat_ids: list[int] = []
        self.activity_ids: list[int] = []
        self.conversation: list[ConversationRow] = []
        self._cycles: dict[int, dict[str, JsonValue]] = {}
        self._cycle_order: list[int] = []
        self._load()

    def _load(self) -> None:
        chat_path = self.evidence_root / "chat.json"
        activity_path = self.evidence_root / "activity.json"
        raw_activity_path = self.evidence_root / "raw-activity.json"
        if chat_path.is_file():
            chat = read_json_object(chat_path)
            _merge_records(self._wrapped, _package_records(chat))
            self.chat_ids = TypeAdapter(list[int]).validate_python(chat["record_ids"])
        if activity_path.is_file():
            activity = read_json_object(activity_path)
            _merge_records(self._wrapped, _package_records(activity))
            self.activity_ids = TypeAdapter(list[int]).validate_python(activity["record_ids"])
        if raw_activity_path.is_file():
            raw_activity = read_json_object(raw_activity_path)
            _merge_records(self._wrapped, _package_records(raw_activity))
        self._cards = {key: RecordCard.model_validate(record_card(wrapped)) for key, wrapped in self._wrapped.items()}
        self._load_cycles()
        self.conversation = [self._conversation_row(key) for key in self.chat_ids if key in self._wrapped]

    def _load_cycles(self) -> None:
        root = self.evidence_root / "cycles"
        if not root.is_dir():
            return
        for path in sorted(root.glob("*.json")):
            package = read_json_object(path)
            captured = CapturedCyclePackage.model_validate(package)
            tick = captured.gwt.tick_id
            self._cycles[tick] = package
            self._cycle_order.append(tick)
            _merge_records(self._wrapped, _package_records(package))
        self._cards = {key: RecordCard.model_validate(record_card(wrapped)) for key, wrapped in self._wrapped.items()}
        self._cycle_order.sort()

    def _conversation_row(self, key: int) -> ConversationRow:
        """Project one public dialogue record to a conversation row."""
        data = CapturedDialogue.model_validate(self._wrapped[key]["data"])
        speaker = data.override_source or data.provenance.get("source_ref") or "Unknown"
        return ConversationRow(
            record_id=key,
            participant=self.participant,
            speaker=str(speaker),
            text=data.content,
            at=normalize_time(data.timestamp_world_begin),
            wall_at=normalize_time(data.timestamp_creation),
            delivery_id=str(data.metadata.get("delivery_id", "")),
            feature_type=data.feature_type,
        )

    def cards(self) -> list[RecordCard]:
        """Return every captured record card in record order."""
        return [self._cards[key] for key in sorted(self._cards)]

    def summaries(self) -> list[RecordSummary]:
        """Return every record as a list row without the full payload."""
        return [record_summary(self._cards[key]) for key in sorted(self._cards)]

    def card(self, record_id: int) -> RecordCard:
        """Return one record card or raise ``FileNotFoundError``."""
        card = self._cards.get(record_id)
        if card is None:
            raise FileNotFoundError(f"Record {record_id} is not captured for participant {self.participant}.")
        return card

    def raw(self, record_id: int) -> dict[str, JsonValue]:
        """Return the captured serialized wrapper for one record."""
        wrapped = self._wrapped.get(record_id)
        if wrapped is None:
            raise FileNotFoundError(f"Record {record_id} is not captured for participant {self.participant}.")
        return wrapped

    def cycles(self) -> list[CycleSummary]:
        """Return every captured cycle in tick order."""
        rows: list[CycleSummary] = []
        for tick in self._cycle_order:
            package = self._cycles[tick]
            captured = CapturedCyclePackage.model_validate(package)
            gwt = captured.gwt
            rows.append(CycleSummary(
                participant=self.participant,
                tick=tick,
                at=normalize_time(captured.cycle.timestamp),
                wall_at=normalize_time(gwt.timestamp),
                summary=gwt.summary,
                slot_id=str(gwt.slot_id),
                member_count=len(gwt.knoxel_ids),
                stimulus_count=len(gwt.stimulus_ids),
                codelet_runs=len(gwt.trace.codelet_runs),
            ))
        return rows

    def cycle(self, tick: int) -> CycleDetail:
        """Return one cycle laid out by the shared formatter."""
        package = self._cycles.get(tick)
        if package is None:
            raise FileNotFoundError(f"Cycle {tick} is not captured for participant {self.participant}.")
        cards = {key: card.model_dump() for key, card in self._cards.items()}
        captured = CapturedCyclePackage.model_validate(package)
        used = {key: record_summary(self._cards[key]) for key in captured.record_ids if key in self._cards}
        return CycleDetail(
            participant=self.participant,
            tick=tick,
            presentation=present_cycle(package, cards),
            cards=used,
        )

    def search(self, query: str, limit: int = 200) -> list[RecordCard]:
        """Return cards whose text, kind, role, or ID contains ``query``."""
        needle = query.strip().lower()
        if not needle:
            return self.cards()[:limit]
        rows = [card for card in self.cards()
                if needle in card.text.lower() or needle in card.kind.lower() or needle in card.role.lower()
                or needle in str(card.id)]
        return rows[:limit]

    def search_summaries(self, query: str, limit: int = 200) -> list[RecordSummary]:
        """Return search hits without the full record payload."""
        return [record_summary(card) for card in self.search(query, limit)]


def record_summary(card: RecordCard) -> RecordSummary:
    """Drop the full captured payload from a card for list transport."""
    data = card.record if isinstance(card.record, dict) else {}
    feature_type = data.get("feature_type")
    role = "Dialogue" if feature_type == "Dialogue" else card.role
    kind = "Dialogue" if feature_type == "Dialogue" else card.kind
    return RecordSummary(id=card.id, tick=card.tick, kind=kind, role=role, text=card.text,
                         source=card.source, operation=card.operation, codelet=card.codelet,
                         status=card.status, fixture=card.fixture)


def index_records(participant: str, evidence_root: Path) -> RecordIndex:
    """Build the record and conversation index for one participant attempt."""
    return RecordIndex(participant, evidence_root)


def read_agent_receipts(attempt_root: Path) -> list[dict[str, JsonValue]]:
    """Read harness actor and director receipts, if the attempt has them.

    These are not companion model calls. They are the simulation's own draft,
    review, and stage-judgment calls, so the timeline labels them separately.
    """
    root = attempt_root / "agents"
    if not root.is_dir():
        return []
    rows: list[dict[str, JsonValue]] = []
    for path in sorted(root.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            rows.append({**value, "_path": path.relative_to(attempt_root).as_posix(),
                         "_mtime": path.stat().st_mtime})
    return rows


__all__ = [
    "ConversationRow",
    "CycleDetail",
    "CycleSummary",
    "RecordCard",
    "RecordIndex",
    "RecordSummary",
    "record_summary",
    "index_records",
    "read_agent_receipts",
]
