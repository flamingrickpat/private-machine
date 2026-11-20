"""Read private-machine prompt logs as one model-call ledger.

Each call is written by ``pm.llm.prompt_log``. The JSON file is the exact
request and provider response. The Markdown file is the readable log. The
``usage-*.jsonl`` files carry the start and outcome events with elapsed time and
token usage. This module joins those three receipts without changing them.

Owns: the model-call index for one participant attempt.
Does not own: the model request, the provider response, or timeline ordering.
See the package overview in ``pm.tools.timeline``.
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, JsonValue

from pm.tools.timeline.evidence import normalize_time


class ChatMessage(BaseModel):
    """One message as the model received it."""

    role: str
    content: str


class UsageEvent(BaseModel):
    """One start or outcome event from a prompt-log usage file."""

    event: str
    time: str = ""
    elapsed_seconds: float | None = None
    usage: dict[str, JsonValue] = Field(default_factory=dict[str, JsonValue])
    finish_reason: str | None = None
    ttft_seconds: float | None = None
    tokens_per_second: float | None = None
    error: str | None = None
    world_time: str = ""
    module: str = ""
    invocation_id: str = ""


class CallSummary(BaseModel):
    """One ledger row, small enough to list and filter."""

    call_id: str
    participant: str
    route: str
    model: str = ""
    module: str = ""
    invocation_id: str = ""
    process_id: int | None = None
    thread_id: int | None = None
    prompt_hash: str = ""
    owner_thread: str = ""
    started_at: str = ""
    world_time: str = ""
    elapsed_seconds: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    finish_reason: str = ""
    error: str | None = None
    message_count: int = 0
    has_response: bool = False
    json_log: str = ""
    markdown_log: str = ""
    preview: str = ""


class CallRecord(CallSummary):
    """The full request and response of one model call."""

    endpoint: str = ""
    request_count: int = 0
    messages: list[ChatMessage] = Field(default_factory=list[ChatMessage])
    input_text: str = ""
    response_text: str = ""
    reasoning_text: str = ""
    response: JsonValue = None
    payload: JsonValue = None
    usage_events: list[UsageEvent] = Field(default_factory=list[UsageEvent])


    def summary(self) -> CallSummary:
        """Retain list and timing fields without keeping model inputs or responses alive."""
        return CallSummary(
            participant=self.participant, call_id=self.call_id, route=self.route, model=self.model,
            module=self.module, invocation_id=self.invocation_id, process_id=self.process_id,
            thread_id=self.thread_id, owner_thread=self.owner_thread, prompt_hash=self.prompt_hash,
            started_at=self.started_at, world_time=self.world_time, elapsed_seconds=self.elapsed_seconds,
            prompt_tokens=self.prompt_tokens, completion_tokens=self.completion_tokens, total_tokens=self.total_tokens,
            finish_reason=self.finish_reason, error=self.error, message_count=self.message_count,
            has_response=self.has_response, preview=self.preview, json_log=self.json_log, markdown_log=self.markdown_log,
        )


def message_text(value: JsonValue) -> str:
    """Flatten one message content value to readable text.

    Chat content is normally a string. A multimodal message can be a list of
    typed parts. The raw value remains in the call detail, so this function only
    chooses the text that the review list can read.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("text")
                parts.append(text if isinstance(text, str) else json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False)


def usage_int(usage: dict[str, JsonValue], key: str) -> int | None:
    """Return one integer provider usage value, or ``None`` when absent."""
    value = usage.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _usage_events(prompt_root: Path, process_id: int | None) -> dict[str, list[UsageEvent]]:
    """Index every usage event by call ID."""
    rows: dict[str, list[UsageEvent]] = {}
    if not prompt_root.is_dir():
        return rows
    pattern = "usage-*.jsonl" if process_id is None else f"usage-{process_id}.jsonl"
    for path in sorted(prompt_root.glob(pattern)):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                raw: JsonValue = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(raw, dict):
                continue
            call_id = raw.get("call_id")
            if not isinstance(call_id, str):
                continue
            event = UsageEvent.model_validate(raw)
            rows.setdefault(call_id, []).append(event)
    return rows


def _response_text(response: JsonValue) -> tuple[str, str, str]:
    """Return ``(text, reasoning, finish_reason)`` from a chat response."""
    if not isinstance(response, dict):
        return "", "", ""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return "", "", ""
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        return "", "", str(choice.get("finish_reason", ""))
    text = message_text(message.get("content"))
    reasoning = message_text(message.get("reasoning_content") or message.get("reasoning"))
    return text, reasoning, str(choice.get("finish_reason", ""))


def _messages(payload: JsonValue) -> list[ChatMessage]:
    """Return the request messages as named rows."""
    if not isinstance(payload, dict):
        return []
    raw = payload.get("messages")
    if not isinstance(raw, list):
        return []
    result: list[ChatMessage] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        result.append(ChatMessage(role=str(role) if role is not None else "", content=message_text(item.get("content"))))
    return result


def _input_text(payload: JsonValue) -> str:
    """Return the embedding input, which is not a chat message list."""
    if not isinstance(payload, dict):
        return ""
    value = payload.get("input")
    return message_text(value)


class CallIndex:
    """Every prompt-log call for one participant attempt, keyed by call ID."""

    def __init__(self, participant: str, prompt_root: Path, evidence_root: Path, process_id: int | None = None) -> None:
        self.participant: str = participant
        self.prompt_root: Path = prompt_root
        self.evidence_root: Path = evidence_root
        self._usage: dict[str, list[UsageEvent]] = _usage_events(prompt_root, process_id)
        self.process_id: int | None = process_id
        self._calls: dict[str, CallSummary] = {}
        self._load()

    def _load(self) -> None:
        if not self.prompt_root.is_dir():
            return
        paths = (self.prompt_root / f"{call_id}.json" for call_id in self._usage) if self.process_id is not None else self.prompt_root.glob("*.json")
        for path in sorted(paths):
            if not path.is_file():
                continue
            if path.name.startswith("usage-"):
                continue
            record = self._read_call(path)
            self._calls[record.call_id] = record.summary()

    def _read_call(self, path: Path) -> CallRecord:
        """Read one exact wire receipt. Only its summary survives an index scan."""
        call_id = path.stem
        try:
            raw: JsonValue = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            return CallRecord(
                call_id=call_id, participant=self.participant, route="unreadable",
                error=f"{type(error).__name__}: {error}", json_log=self._relative(path),
            )
        if not isinstance(raw, dict):
            return CallRecord(
                call_id=call_id, participant=self.participant, route="unreadable",
                error="The prompt log is not a JSON object.", json_log=self._relative(path),
            )
        payload = raw.get("payload")
        response = raw.get("response")
        response_text, reasoning_text, finish_reason = _response_text(response)
        events = self._usage.get(call_id, [])
        ended = next((item for item in reversed(events) if item.event == "end"), None)
        started = next((item for item in events if item.event == "start"), None)
        usage: dict[str, JsonValue] = {}
        if ended is not None and ended.usage:
            usage = ended.usage
        elif isinstance(response, dict):
            response_usage = response.get("usage")
            if isinstance(response_usage, dict):
                usage = response_usage
        prompt_tokens = usage_int(usage, "prompt_tokens")
        completion_tokens = usage_int(usage, "completion_tokens")
        total_tokens = usage_int(usage, "total_tokens")
        error_value = raw.get("error")
        error = None
        if isinstance(error_value, str):
            error = error_value
        elif error_value is not None:
            error = json.dumps(error_value, ensure_ascii=False)
        if ended is not None and ended.error:
            error = ended.error
        module = str(raw.get("module") or (started.module if started is not None else ""))
        invocation_id = str(raw.get("invocation_id") or (started.invocation_id if started is not None else ""))
        world_time = str(raw.get("world_time") or (started.world_time if started is not None else ""))
        markdown = path.with_suffix(".md")
        request_count = raw.get("request_count", 0)
        record = CallRecord(
            call_id=call_id,
            participant=self.participant,
            route=str(raw.get("route", "")),
            model=str(raw.get("model", "")),
            module=module,
            invocation_id=invocation_id,
            prompt_hash=str(raw.get("prompt_sha256", "")),
            process_id=raw.get("pid"), thread_id=raw.get("thread_id"),
            owner_thread=str(raw.get("owner_thread", "")),
            started_at=str(raw.get("started_at", "")),
            world_time=world_time,
            elapsed_seconds=ended.elapsed_seconds if ended is not None else None,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            finish_reason=finish_reason or (ended.finish_reason if ended is not None and ended.finish_reason else ""),
            error=error,
            message_count=len(_messages(payload)),
            has_response=bool(response_text or reasoning_text),
            preview=(response_text or reasoning_text or _input_text(payload) or error or "")[:240],
            endpoint=str(raw.get("endpoint", "")),
            request_count=request_count if isinstance(request_count, int) else 0,
            messages=_messages(payload),
            input_text=_input_text(payload),
            response_text=response_text,
            reasoning_text=reasoning_text,
            response=response,
            payload=payload,
            usage_events=events,
            json_log=self._relative(path),
            markdown_log=self._relative(markdown) if markdown.is_file() else "",
        )
        return record

    def _relative(self, path: Path) -> str:
        """Return a path relative to the participant evidence root."""
        try:
            return path.relative_to(self.evidence_root).as_posix()
        except ValueError:
            return ""

    def all(self) -> list[CallSummary]:
        """Return every call in stable call-ID order."""
        return [self._calls[key] for key in sorted(self._calls)]

    def summaries(self) -> list[CallSummary]:
        """Return every call as a list row."""
        return self.all()

    def get(self, call_id: str) -> CallRecord:
        """Return one call or raise ``FileNotFoundError``."""
        record = self._calls.get(call_id)
        if record is None:
            raise FileNotFoundError(f"No prompt log for call {call_id}.")
        return self._read_call(self.prompt_root / f"{call_id}.json")

    def for_invocation(self, invocation_id: str) -> list[CallSummary]:
        """Return the calls that share one workflow invocation."""
        return [call for call in self.all() if call.invocation_id == invocation_id]

    def world_time_for_invocation(self, invocation_id: str) -> str:
        """Return the virtual time shared by an invocation's calls, when recorded."""
        for call in self.for_invocation(invocation_id):
            stamp = normalize_time(call.world_time)
            if stamp:
                return stamp
        return ""


def index_calls(participant: str, prompt_root: Path, evidence_root: Path, process_id: int | None = None) -> CallIndex:
    """Build the call index for one participant attempt.

    A missing prompt folder is allowed because a failed attempt can stop before
    the runtime writes a call. An unreadable call remains a visible error row.
    """
    return CallIndex(participant, prompt_root, evidence_root, process_id)


__all__ = [
    "CallIndex",
    "CallRecord",
    "CallSummary",
    "ChatMessage",
    "UsageEvent",
    "index_calls",
    "message_text",
    "usage_int",
]
