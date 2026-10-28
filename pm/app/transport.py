"""Asynchronous stimulus delivery and public speech over a WebSocket.

HTTP acknowledges durable stimulus delivery. Speech arrives independently.
Output IDs are knoxel IDs, so clients can replay and deduplicate messages.

The stream is a cursor over commit order, not record IDs: a record ID can be
reserved before another transaction commits. A reconnect sends its last cursor
and receives the later messages exactly once. A closed browser does not cancel
cognition, because the reply is already committed when it becomes visible.
"""
from __future__ import annotations

import json
import socket
from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol
from urllib.parse import parse_qs, urlparse

from pydantic import BaseModel, Field, model_validator
from websockets.datastructures import Headers
from websockets.http11 import Request
from websockets.protocol import State
from websockets.server import ServerProtocol

from pm.memory.store import MemoryStore

if TYPE_CHECKING:
    from pm.app.web_chat import WebChatServer


class EventStreamHandler(Protocol):
    """The HTTP handler surface that one event-stream upgrade needs.

    The web chat supplies its own request handler, so the stream names only the
    members it uses: the request target and headers, the socket, and the two
    helpers that answer a rejected upgrade.
    """

    path: str
    headers: Mapping[str, str]
    connection: socket.socket
    close_connection: bool

    def _json_error(self, error: str, status: int) -> None: ...

    def _same_origin(self) -> bool: ...


class CompletionMessage(BaseModel):
    content: str


class CompletionStimulus(BaseModel):
    """Read only the last content field. PM owns all conversation history."""

    messages: list[dict] = Field(min_length=1)
    user: str | None = None
    delivery_id: str = ""
    entity_ref: str = ""

    @model_validator(mode="after")
    def last_content(self):
        """Reject a missing or empty final text before delivering a stimulus."""
        message = CompletionMessage.model_validate(self.messages[-1])
        if not message.content.strip():
            raise ValueError("The last message content must contain text.")
        return self


class StimulusAccepted(BaseModel):
    status: str = "accepted"
    stimulus_id: str


def serve_events(handler: EventStreamHandler, chat: "WebChatServer") -> None:
    """Upgrade this HTTP connection and replay public companion messages.

    The protocol library handles masking, control frames, and close frames.
    Server shutdown ends the stream. Disconnects do not cancel companion work.
    """
    query = parse_qs(urlparse(handler.path).query)
    cursor = query.get("after", ["0"])[0]
    try:
        parse_cursor(cursor)
    except ValueError as error:
        handler._json_error(str(error), 400)
        return
    if not handler._same_origin():
        handler._json_error("Cross-origin requests are rejected.", 403)
        return
    protocol = ServerProtocol(max_size=None)
    request = Request(handler.path, Headers(list(handler.headers.items())))
    protocol.receive_data(request.serialize())
    protocol.events_received()
    response = protocol.accept(request)
    protocol.send_response(response)
    for data in protocol.data_to_send():
        handler.connection.sendall(data)
    if response.status_code != 101:
        return
    handler.close_connection = True
    handler.connection.settimeout(.1)
    after = cursor

    def flush() -> None:
        """Send every frame the protocol queued, and nothing when it is empty."""
        for data in protocol.data_to_send():
            if data:
                handler.connection.sendall(data)

    while not chat.stopping.is_set() and protocol.state is State.OPEN:
        events, after = public_events(chat.store, after)
        for event in events:
            protocol.send_text(json.dumps(event, ensure_ascii=False).encode("utf-8"))
            flush()
        try:
            incoming = handler.connection.recv(65536)
        except socket.timeout:
            continue
        if not incoming:
            protocol.receive_eof()
            break
        protocol.receive_data(incoming)
        protocol.events_received()
        flush()
    if protocol.state is State.OPEN:
        protocol.send_close(1001, "Runtime stopping")
        flush()


def parse_cursor(value: str) -> tuple[int, int]:
    """Read a commit sequence and ordinal. Zero starts at the imported baseline."""
    parts = value.split(":") if value != "0" else ["0", "0"]
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ValueError("The event cursor must be zero or sequence:ordinal.")
    return int(parts[0]), int(parts[1])


def public_events(store: MemoryStore, after: str = "0") -> tuple[list[dict], str]:
    """Read committed speech after a durable cursor, including late commits.

    Record IDs can be reserved before another transaction commits. Cursors
    therefore use commit order. Lifecycle receipts never repeat an utterance.
    """
    from pm.records.enums import FeatureType, InterlocusType
    from pm.records.memory_records import Feature
    cursor = parse_cursor(after)
    result = []
    with store.lock:
        store._ensure_history()
        history = store.causal_history
        groups = [(0, history.baseline_ids)]
        groups.extend((item.sequence, item.created_ids) for item in history.transactions.values()
                      if item.commit_id in store._live_transactions)
        last = cursor
        for sequence, ids in groups:
            if sequence < cursor[0]:
                continue
            start = cursor[1] if sequence == cursor[0] else 0
            for ordinal, record_id in enumerate(ids[start:], start + 1):
                position = sequence, ordinal
                if position <= cursor:
                    continue
                last = max(last, position)
                record = store._all_knoxels.get(record_id)
                if (not isinstance(record, Feature) or record.feature_type != FeatureType.Dialogue
                        or record.interlocus != InterlocusType.Public
                        or record.override_source != store.get_companion_name()):
                    continue
                result.append({"type": "message", "output_id": record.id, "cursor": f"{sequence}:{ordinal}",
                               "content": record.content, "speaker": record.override_source,
                               "timestamp": record.timestamp_world_begin.isoformat(),
                               "correlation_id": record.metadata.get("reply_correlation", ""),
                               "addressee_entity_id": record.addressee_entity_id})
    return result, f"{last[0]}:{last[1]}"


def event_cursor(store: MemoryStore) -> str:
    """Return the latest committed position without rendering conversation."""
    with store.lock:
        store._ensure_history()
        position = (0, len(store.causal_history.baseline_ids))
        for item in store.causal_history.transactions.values():
            if item.commit_id in store._live_transactions and item.created_ids:
                position = max(position, (item.sequence, len(item.created_ids)))
        return f"{position[0]}:{position[1]}"
