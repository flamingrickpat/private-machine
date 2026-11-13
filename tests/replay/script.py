"""The recorded model call list for one scenario sample.

A recording is a JSONL file of :class:`RecordedCall` rows in the order the
architecture asked for them. Each row keeps the exact request text that
produced the response, so a replay that asks a different question stops at the
first difference instead of silently borrowing an unrelated answer.

The digest is a convenience index. The comparison that matters is on the
request text, because it can be printed and diffed when a prompt changes.

Reading map: :func:`request_text` renders a request canonically,
:class:`Recording` appends live calls, :class:`RecordingReader` serves them
back, and :func:`open_recording` picks the writer or reader for one run.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from collections.abc import Iterator, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pm.llm.prompt_encoding import TurnInput, normalize_turns


#: The two request kinds the harness scripts. Both reach a paid endpoint when
#: recorded, so both must be served from the recording when replayed.
CallKind = Literal["completion", "measurement"]


class RecordedCall(BaseModel):
    """One model request and the exact answer it received.

    ``request`` is the canonical rendering of the turns and generation options,
    including the whole prompt. ``response`` is the completion text, or the
    serialized context measurement. ``elapsed_seconds`` is measurement noise
    from the recording and is retained only for cost and latency inspection.
    """

    model_config = ConfigDict(extra="forbid")

    index: int = Field(description="Zero-based position in this sample's call order.")
    kind: CallKind = Field(description="Completion text or context measurement.")
    digest: str = Field(description="SHA-256 of request; a fast mismatch detector, not the authority.")
    request: str = Field(description="Canonical request text, retained so a prompt change can be diffed.")
    response: str = Field(description="Recorded completion text or serialized measurement.")
    elapsed_seconds: float = Field(description="Real seconds the recorded request took.")


class ReplayDivergence(RuntimeError):
    """The replayed architecture asked a different question than the recording.

    This is a genuine failure, not a missing cache entry. It means the change
    under test altered the prompt, the call order, or the number of calls, so
    the recording cannot speak for the new code.
    """

    def __init__(self, index: int, kind: CallKind, reason: str, *, expected: str = "", actual: str = "") -> None:
        self.index = index
        self.kind = kind
        self.reason = reason
        self.expected = expected
        self.actual = actual
        super().__init__(f"replay diverged at call {index} ({kind}): {reason}{self._diff()}")

    def _diff(self) -> str:
        """Return a bounded excerpt around the first difference.

        The full texts are long rendered prompts, so the message shows one
        window and the caller writes both texts beside the recording for a
        closer look. A unified diff of whole prompts is unreadable.
        """
        if not self.expected or not self.actual:
            return ""
        if self.expected == self.actual:
            return "\nThe request text is identical; only its recorded position differs."
        index = next((position for position, (left, right) in
                      enumerate(zip(self.expected, self.actual)) if left != right),
                     min(len(self.expected), len(self.actual)))
        start = max(0, index - 160)
        return (f"\nFirst difference at character {index}.\n"
                f"recorded: {self.expected[start:index + 200]!r}\n"
                f"replayed: {self.actual[start:index + 200]!r}")


def turn_text(turn: TurnInput) -> str:
    """Render one turn as canonical JSON, keeping tool calls and provider state."""
    normalized = normalize_turns([turn])[0]
    return json.dumps(dataclasses.asdict(normalized), sort_keys=True, ensure_ascii=False, default=str)


#: Prompt fields that report how long a run happened to take. They measure the
#: recording, not the situation, so they cannot be part of a replay's identity.
#: Prompt fields that report how long a run happened to take.
_ELAPSED_FIELDS = re.compile(r"active_elapsed_seconds|elapsed_seconds")
#: A field name can sit behind one, two, or three levels of JSON string
#: escaping, so the closing quote keeps whatever backslashes precede it.
_ELAPSED_VALUE = re.compile(r'((?:\\*)"):\s*-?[0-9][0-9.eE+-]*')


def normalize_timing(text: str) -> str:
    """Replace measured elapsed durations with a stable placeholder.

    The architecture records accumulated wall-clock time for active private
    computation. That number differs on every run by microseconds and reaches
    the prompt through the inquiry state. It is a receipt of how long the run
    took, never evidence the model reasons about, so a replay normalizes it the
    same way the completion cache normalizes rendered timestamps.
    """
    pieces: list[str] = []
    consumed = 0
    for field in _ELAPSED_FIELDS.finditer(text):
        value = _ELAPSED_VALUE.match(text, field.end())
        if value is None:
            continue  # A mention of the name that is not a number stays verbatim.
        # ``value.group(1)`` is the closing quote plus its escaping, for example
        # ``\"``. The placeholder keeps the same quoting so the surrounding
        # JSON string stays valid at whatever nesting depth it sits.
        quote = value.group(1)
        pieces.append(text[consumed:value.start()])
        pieces.append(f'{quote}:{quote}<elapsed>{quote}')
        consumed = value.end()
    pieces.append(text[consumed:])
    return "".join(pieces)


def request_text(kind: CallKind, messages: Sequence[TurnInput], generation: dict[str, object]) -> str:
    """Render one request canonically. Cancellation is scheduling, not content.

    A ``cancel`` event differs on every run and changes no model input, so it is
    excluded. Everything else the caller passed is retained, including tools and
    the reasoning budget, because a change there changes the request. Measured
    elapsed durations are normalized by :func:`normalize_timing`.
    """
    options = {key: value for key, value in generation.items() if key != "cancel"}
    rendered = {
        "kind": kind,
        "turns": [turn_text(turn) for turn in messages],
        "generation": json.loads(json.dumps(options, sort_keys=True, ensure_ascii=False, default=str)),
    }
    text = normalize_timing(json.dumps(rendered, sort_keys=True, ensure_ascii=False, indent=2))
    # A normalization that breaks the JSON would silently produce an unusable
    # recording, so the boundary proves its own output instead of trusting it.
    json.loads(text)
    return text


def digest_of(text: str) -> str:
    """Return the short index digest for one canonical request text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


class RecordingWriter:
    """Append live model calls to one sample's ``calls.jsonl``.

    Each row is written and flushed as it happens, so a run that fails late
    still leaves the calls that did succeed for inspection.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = path.open("w", encoding="utf-8")
        self._index = 0

    def append(self, kind: CallKind, request: str, response: str, elapsed_seconds: float) -> RecordedCall:
        """Retain one answered call and return its row."""
        call = RecordedCall(index=self._index, kind=kind, digest=digest_of(request),
                            request=request, response=response, elapsed_seconds=elapsed_seconds)
        self._stream.write(call.model_dump_json() + "\n")
        self._stream.flush()
        self._index += 1
        return call

    @property
    def count(self) -> int:
        """Return how many calls this recording has retained."""
        return self._index

    def close(self) -> None:
        """Close the append stream."""
        self._stream.close()

    def __enter__(self) -> "RecordingWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class RecordingReader:
    """Serve one sample's recorded calls back in order.

    Reading is eager and strict: a malformed row, a gap in the indices, or a
    trailing request with no recorded answer fails at load time rather than
    halfway through a scenario.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        rows = [RecordedCall.model_validate_json(line)
                for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if [row.index for row in rows] != list(range(len(rows))):
            raise ValueError(f"Recording {path} has a gap or duplicate in its call order.")
        # Normalize on load as well as on write, so a recording made before a
        # normalization existed still compares against the normalized request.
        self._rows = [row.model_copy(update={"request": normalize_timing(row.request),
                                             "digest": digest_of(normalize_timing(row.request))})
                      for row in rows]
        self._position = 0

    def __len__(self) -> int:
        """Return the number of recorded calls."""
        return len(self._rows)

    @property
    def position(self) -> int:
        """Return how many calls the replay has consumed."""
        return self._position

    @property
    def remaining(self) -> int:
        """Return how many recorded calls the replay has not asked for yet."""
        return len(self._rows) - self._position

    def next_call(self, kind: CallKind, request: str) -> RecordedCall:
        """Return the next recorded call, or fail loudly on any mismatch."""
        if self._position >= len(self._rows):
            raise ReplayDivergence(self._position, kind,
                "the replayed architecture made more calls than the recording holds.")
        row = self._rows[self._position]
        if row.kind != kind:
            raise ReplayDivergence(self._position, kind,
                f"the recording holds a {row.kind} at this position.")
        if row.digest != digest_of(request):
            raise ReplayDivergence(self._position, kind,
                "the request text changed since the recording.", expected=row.request, actual=request)
        self._position += 1
        return row

    def unasked(self) -> tuple[RecordedCall, ...]:
        """Return recorded calls the replay never requested, for a completeness check."""
        return tuple(self._rows[self._position:])

    def __iter__(self) -> Iterator[RecordedCall]:
        """Iterate the retained rows for inspection and reporting."""
        return iter(self._rows)


def open_recording(path: Path, mode: Literal["record", "replay"]) -> RecordingWriter | RecordingReader:
    """Return the writer or reader for one sample directory."""
    target = path / "calls.jsonl"
    if mode == "record":
        return RecordingWriter(target)
    if not target.exists():
        raise FileNotFoundError(f"Missing recorded calls for replay: {target}")
    return RecordingReader(target)


class RecordingManifest(BaseModel):
    """What produced one sample, retained beside its calls and checkpoint.

    The manifest is deliberately narrow: enough to know which situation, which
    model, and which revision the sample describes. It never contains
    credentials, and it is not a substitute for the live gate's evidence.
    """

    model_config = ConfigDict(extra="forbid")

    scenario: str = Field(description="Scenario name from tests.replay.scenarios.")
    sample: int = Field(ge=1, description="One-based independent continuation of the same situation.")
    recorded_at_utc: datetime = Field(description="When the recording finished.")
    revision: str = Field(description="Git revision of the code that made the recording.")
    model: str = Field(description="Chat model that produced the recorded completions.")
    endpoint: str = Field(description="Chat endpoint host and path, without credentials.")
    clock_start_utc: datetime = Field(description="Virtual world time the situation started at.")
    call_count: int = Field(ge=0, description="Number of recorded model calls.")
    request_count: int = Field(ge=0, description="Real provider requests the recording cost.")
    source_hashes: dict[str, str] = Field(description="SHA-256 of the modules the recording exercised.")
    notes: tuple[str, ...] = Field(default=(), description="Scenario-specific remarks a reviewer needs.")
    replay_blockers: tuple[str, ...] = Field(default=(),
        description="Reasons this sample cannot be replayed; empty when the call trace is complete.")

    @staticmethod
    def now() -> datetime:
        """Return the current UTC time for a fresh manifest."""
        return datetime.now(timezone.utc)
