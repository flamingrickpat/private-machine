"""The serialized message value that crosses the bus, and its JSON form.

An ``Envelope`` is one independently delivered message. It reaches a
subscriber only as JSON text (see ``pm.bus.bus``), so each subscriber owns a
detached copy of the payload and cannot change another subscriber's view.

Owns: the envelope fields, the recreated-envelope contract, and the JSON shape
of a serialized envelope. Does not own: routing, mailboxes, retry, or durable
replay; those live in ``pm.bus.bus`` and in the modules that publish.

Invariant: every payload value must be JSON serializable. Publication fails
loudly with ``TypeError`` on a live object, because a message that cannot be
recreated for the next subscriber must never look deliverable.
"""
from __future__ import annotations

import json
import uuid
from typing import Literal, NotRequired, TypedDict, TypeAlias

#: A value that survives a JSON round trip. Payloads are built by workflows and
#: modules, so the concrete keys differ per topic; the value domain does not.
JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

#: One envelope payload: JSON values keyed by a topic-specific field name.
JsonObject: TypeAlias = dict[str, JsonValue]

EnvelopeKind: TypeAlias = Literal["data", "telemetry", "control"]


class EnvelopeJson(TypedDict):
    """Serialized envelope fields, with constructor defaults allowed on input.

    ``to_json`` always writes both optional fields. A reader also accepts their
    omission, as the constructor does.
    """

    id: str
    source: str
    kind: EnvelopeKind
    topic: str
    correlation_id: str
    payload: JsonObject
    target: NotRequired[str | None]
    causation_id: NotRequired[str | None]


def new_id() -> str:
    """Return a fresh envelope identifier."""
    return uuid.uuid4().hex


class Envelope:
    """One independently delivered bus message."""

    __slots__ = ("id", "source", "kind", "topic", "correlation_id", "payload", "target", "causation_id")

    id: str
    source: str
    kind: EnvelopeKind
    topic: str
    correlation_id: str
    payload: JsonObject
    target: str | None
    causation_id: str | None

    def __init__(
        self,
        id: str,
        source: str,
        kind: EnvelopeKind,
        topic: str,
        correlation_id: str,
        payload: JsonObject,
        target: str | None = None,
        causation_id: str | None = None,
    ) -> None:
        self.id = id
        self.source = source
        self.kind = kind
        self.topic = topic
        self.correlation_id = correlation_id
        self.payload = payload
        self.target = target
        self.causation_id = causation_id

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Envelope) and other.id == self.id

    def __repr__(self) -> str:
        return f"Envelope({self.topic!r}, {self.id[:8]})"

    def to_json(self) -> str:
        """Serialize this envelope. A non-JSON payload value raises ``TypeError``.

        The key order is fixed here and in ``from_json``, so a serialized
        envelope reads the same in traces and in a debugger.
        """
        data: EnvelopeJson = {
            "id": self.id,
            "source": self.source,
            "kind": self.kind,
            "topic": self.topic,
            "correlation_id": self.correlation_id,
            "payload": self.payload,
            "target": self.target,
            "causation_id": self.causation_id,
        }
        return json.dumps(data, allow_nan=False)

    @classmethod
    def from_json(cls, text: str) -> "Envelope":
        """Decode an envelope through its constructor contract.

        Omitted optional fields keep their constructor defaults. Missing required
        fields and unknown fields raise ``TypeError`` instead of losing data.
        """
        data: EnvelopeJson = json.loads(text)
        return cls(**data)


def new_envelope(
    source: str,
    kind: EnvelopeKind,
    topic: str,
    correlation_id: str,
    payload: JsonObject,
    target: str | None = None,
    causation_id: str | None = None,
) -> Envelope:
    """Build an envelope with a fresh id for a module that publishes once."""
    return Envelope(
        id=new_id(), source=source, kind=kind, topic=topic,
        correlation_id=correlation_id, payload=payload, target=target, causation_id=causation_id,
    )
