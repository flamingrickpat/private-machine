"""A loopback web chat for one running companion.

The server runs in its own thread beside the application. It reads public
dialogue from the live ``MemoryStore`` and hands new messages to a submit
callable. It owns no memory: every message that the page shows is an
accepted record, so a restart shows the same history.

The page is one static HTML file with a script and a style sheet under
``pm/app/static/web_chat/``. It needs no framework and no build step.

API:

- ``GET /api/session`` returns :class:`ChatSession`.
- ``GET /api/messages?after=<id>`` returns :class:`MessagePage`.
- ``POST /api/messages`` with :class:`PostMessage` JSON calls the submit
  callable and returns :class:`PostResult`.

The server binds to ``127.0.0.1`` and has no authentication. The login name
only selects the speaker. Anyone who can open the URL can speak as any name.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from pm.constants import WEB_CHAT_HOST, WEB_CHAT_MAX_BODY_BYTES, WEB_CHAT_MAX_NAME_CHARS, WEB_CHAT_POLL_MS
from pm.memory.store import MemoryStore
from pm.records.enums import FeatureType, InterlocusType

STATIC_ROOT = Path(__file__).resolve().parent / "static" / "web_chat"

#: URL path -> (file name, content type) of the static page files.
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}

#: The script and the style come only from this server. Message text never
#: becomes markup, and this policy blocks inline script as a second barrier.
CONTENT_SECURITY_POLICY = ("default-src 'self'; style-src 'self'; script-src 'self'; "
                           "connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'")

#: Submit callable: ``submit(text, speaker=name)`` commits one message and
#: returns its exchange correlation ID.
Submit = Callable[..., str]


class ChatMessage(BaseModel):
    """One public dialogue line, as the page shows it.

    ``timestamp`` is the world time of the record in UTC. ``exchange_status``
    is ``pending`` or ``resolved`` on a human message that waits for or got a
    reply, and empty on other messages.
    """

    id: int
    speaker: str
    text: str
    timestamp: datetime
    from_companion: bool
    exchange_status: str = ""


class MessagePage(BaseModel):
    """The answer of ``GET /api/messages``.

    ``messages`` holds the lines with an ID above ``after``, in time order.
    ``last_id`` is the highest message ID that the store holds; the page
    sends it as the next ``after``. ``thinking`` is true while a human
    message after the newest companion line still waits for its reply.
    """

    messages: list[ChatMessage]
    last_id: int
    thinking: bool


class ChatSession(BaseModel):
    """The names and the poll interval that the page needs before login."""

    companion_name: str
    user_name: str
    poll_ms: int


class PostMessage(BaseModel):
    """The body of ``POST /api/messages``. Names and text are stripped."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=WEB_CHAT_MAX_NAME_CHARS)
    text: str = Field(min_length=1)


class PostResult(BaseModel):
    """The answer of ``POST /api/messages``."""

    correlation_id: str


class ErrorResult(BaseModel):
    """The answer of a rejected request."""

    error: str


class _DialogueMetadata(BaseModel):
    """The metadata key of a dialogue record that the page reads. Other keys are ignored."""

    model_config = ConfigDict(extra="ignore")

    exchange_status: str = ""


def _utc(value: datetime) -> datetime:
    """Return ``value`` as an aware UTC time. Records store naive UTC times."""
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def read_messages(store: MemoryStore, after: int = 0) -> MessagePage:
    """Read the public dialogue of ``store`` after the message ID ``after``.

    The read holds the store lock. A commit holds the same lock while it
    validates, writes the checkpoint, and installs the new records, so this
    read sees the state before or after a commit, never a part of one. The
    read only copies field values; it changes no record.

    Imported history counts like live dialogue. The order is world time,
    then ID. New messages always get a higher ID and a later time, so a page
    that appends the lines after its last ID keeps the time order.
    """
    companion = store.get_companion_name()
    with store.lock:
        lines = []
        for feature in store.all_features:
            if feature.feature_type != FeatureType.Dialogue or feature.interlocus != InterlocusType.Public:
                continue
            speaker = feature.source
            lines.append(ChatMessage(
                id=feature.id, speaker=speaker, text=feature.content,
                timestamp=_utc(feature.timestamp_world_begin),
                from_companion=speaker == companion,
                exchange_status=_DialogueMetadata.model_validate(feature.metadata).exchange_status,
            ))
    lines.sort(key=lambda line: (line.timestamp, line.id))
    last_companion = max((index for index, line in enumerate(lines) if line.from_companion), default=-1)
    thinking = any(line.exchange_status == "pending" for line in lines[last_companion + 1:])
    return MessagePage(
        messages=[line for line in lines if line.id > after],
        last_id=max([after, *(line.id for line in lines)]),
        thinking=thinking,
    )


class WebChatServer:
    """Serve the chat page and its API on loopback in one background thread.

    ``submit`` must commit the message before it returns, so the next poll
    shows it. ``port`` 0 takes a free port; ``url`` names the real one.
    The constructor binds the socket and raises ``OSError`` when the port
    is in use. ``stop`` closes the socket and joins the thread; a request
    that runs at that moment finishes first.

    An exception inside a request is a bug in the runtime, not a bad
    request. The handler passes it to ``on_error`` (the application's crash
    path) and answers 500, so the app stops instead of hiding the error.
    """

    def __init__(self, store: MemoryStore, submit: Submit, *, port: int,
                 on_error: Callable[[BaseException], None], host: str = WEB_CHAT_HOST) -> None:
        self.store = store
        self.submit = submit
        self.on_error = on_error
        # Request threads are not daemon threads, so ``server_close`` joins
        # them: a message that is being committed finishes before shutdown.
        self._httpd = ThreadingHTTPServer((host, port), _handler(self))
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="web_chat", daemon=True)

    @property
    def url(self) -> str:
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}/"

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """Stop serving. New messages cannot arrive after this returns."""
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join()

    def session(self) -> ChatSession:
        return ChatSession(companion_name=self.store.get_companion_name(),
                           user_name=self.store.get_user_name(), poll_ms=WEB_CHAT_POLL_MS)

    def names_the_companion(self, name: str) -> bool:
        """Return True when ``name`` is the companion's name or an alias of its entity.

        Such a login would make a human line read as the companion's speech.
        """
        with self.store.lock:
            companion = self.store.get_companion_name()
            companion_id = self.store.get_entity_id(companion)
            return name == companion or (companion_id is not None and self.store.get_entity_id(name) == companion_id)

    def post(self, message: PostMessage) -> PostResult:
        """Hand one checked message to the submit callable."""
        return PostResult(correlation_id=self.submit(message.text, speaker=message.name))


def _handler(chat: WebChatServer) -> type[BaseHTTPRequestHandler]:
    """Build the request handler class that serves ``chat``."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", CONTENT_SECURITY_POLICY)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, model: BaseModel, status: int = 200) -> None:
            self._send(status, model.model_dump_json().encode("utf-8"), "application/json; charset=utf-8")

        def _same_origin(self) -> bool:
            # A page of another site can send a POST to loopback. The browser
            # names that site in Origin, so only this server's own origin passes.
            host = self.headers["Host"]
            origin = self.headers["Origin"]
            return origin is None or origin == f"http://{host}"

        def do_GET(self) -> None:
            self._guarded(self._get)

        def do_POST(self) -> None:
            self._guarded(self._post)

        def _guarded(self, handle: Callable[[], None]) -> None:
            try:
                handle()
            except ConnectionError:
                # The browser closed the connection, for example when the tab
                # closes during a poll. This is not a runtime error.
                pass
            except BaseException as exc:  # a runtime bug must reach the application
                chat.on_error(exc)
                self._json(ErrorResult(error=f"Internal error: {type(exc).__name__}."), 500)

        def _get(self) -> None:
            path = urlparse(self.path)
            if path.path == "/api/session":
                return self._json(chat.session())
            if path.path == "/api/messages":
                query = parse_qs(path.query)
                after = query["after"][0] if "after" in query else "0"
                if not after.isdigit():
                    return self._json(ErrorResult(error="after must be a non-negative integer."), 400)
                return self._json(read_messages(chat.store, int(after)))
            if path.path in STATIC_FILES:
                name, content_type = STATIC_FILES[path.path]
                return self._send(200, (STATIC_ROOT / name).read_bytes(), content_type)
            return self._json(ErrorResult(error="Unknown path."), 404)

        def _post(self) -> None:
            if urlparse(self.path).path != "/api/messages":
                return self._json(ErrorResult(error="Unknown path."), 404)
            if not self._same_origin():
                return self._json(ErrorResult(error="Cross-origin requests are rejected."), 403)
            length = self.headers["Content-Length"]
            if length is None or not length.isdigit() or not 0 < int(length) <= WEB_CHAT_MAX_BODY_BYTES:
                return self._json(ErrorResult(error=f"The body must have 1 to {WEB_CHAT_MAX_BODY_BYTES} bytes."), 413)
            try:
                message = PostMessage.model_validate_json(self.rfile.read(int(length)))
            except ValidationError as exc:
                return self._json(ErrorResult(error=str(exc)), 400)
            if chat.names_the_companion(message.name):
                return self._json(ErrorResult(error="This name belongs to the companion. Choose another name."), 400)
            return self._json(chat.post(message))

    return Handler
