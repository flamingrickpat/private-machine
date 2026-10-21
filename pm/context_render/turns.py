"""Provider-neutral turns with ordered text and media content."""
from __future__ import annotations

import base64
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from string import Formatter
import math
from typing import Literal, TypeAlias


class Role(StrEnum):
    """The author of one complete conversation turn."""

    System = "system"
    Developer = "developer"
    User = "user"
    Assistant = "assistant"
    Tool = "tool"


class MediaKind(StrEnum):
    """The modality of an attachment, independent of provider field names."""

    Image = "image"
    Audio = "audio"
    Video = "video"
    File = "file"


@dataclass(frozen=True)
class Text:
    """One text segment. Its contents remain verbatim."""

    text: str


@dataclass(frozen=True)
class Media:
    """One immutable attachment with bytes or an explicit remote URL.

    The MIME type describes the bytes. URL sources remain references; this
    object never downloads them. Provider adapters decide supported formats.
    Local files enter through from_file, which captures their bytes once.
    """

    kind: MediaKind
    mime_type: str
    data: bytes | None = None
    url: str | None = None
    name: str = ""
    detail: Literal["auto", "low", "high", "original"] = "auto"

    def __post_init__(self) -> None:
        if (self.data is None) == (self.url is None):
            raise ValueError("Media needs exactly one byte payload or URL.")
        if not self.mime_type or not isinstance(self.kind, MediaKind):
            raise ValueError("Media needs an explicit kind and MIME type.")

    @classmethod
    def from_file(cls, kind: MediaKind, path: str | Path, mime_type: str) -> Media:
        """Capture file bytes now; file errors propagate without a model call."""
        source = Path(path)
        return cls(kind, mime_type, data=source.read_bytes(), name=source.name)

    @classmethod
    def from_base64(cls, kind: MediaKind, data: str, mime_type: str, *, name: str = "") -> Media:
        """Decode an explicit base64 payload; reject invalid encoded bytes."""
        return cls(kind, mime_type, data=base64.b64decode(data, validate=True), name=name)


@dataclass(frozen=True)
class VideoFrame:
    """An image sampled at an explicit offset in seconds from a video."""

    seconds: float
    image: Media

    def __post_init__(self) -> None:
        if not math.isfinite(self.seconds) or self.seconds < 0 or self.image.kind is not MediaKind.Image:
            raise ValueError("A video frame needs a nonnegative offset and image media.")


@dataclass(frozen=True)
class VideoFrames:
    """Ordered sampled frames. Sampling remains the media owner's operation."""

    frames: tuple[VideoFrame, ...]

    def __post_init__(self) -> None:
        if not self.frames or any(a.seconds > b.seconds for a, b in zip(self.frames, self.frames[1:])):
            raise ValueError("Video frames need a nonempty chronological sequence.")


ContentPart: TypeAlias = Text | Media | VideoFrames


@dataclass(frozen=True, init=False)
class Content:
    """Composable content without a role or provider encoding.

    Addition preserves attachment order and merges adjacent text segments.
    Text access raises TypeError if attachments would otherwise disappear.
    """

    parts: tuple[ContentPart, ...]
    notes: tuple[str, ...]

    def __init__(self, value: str | ContentPart | tuple[ContentPart, ...] = "", *, notes: tuple[str, ...] = ()) -> None:
        parts = (Text(value),) if isinstance(value, str) else value if isinstance(value, tuple) else (value,)
        if not all(isinstance(part, (Text, Media, VideoFrames)) for part in parts):
            raise TypeError("Content needs text or typed media parts.")
        object.__setattr__(self, "parts", parts)
        object.__setattr__(self, "notes", notes)

    def __add__(self, other: Content | str) -> Content:
        right = other if isinstance(other, Content) else Content(other)
        parts = list(self.parts)
        for part in right.parts:
            if parts and isinstance(parts[-1], Text) and isinstance(part, Text):
                parts[-1] = Text(parts[-1].text + part.text)
            else:
                parts.append(part)
        return Content(tuple(parts), notes=tuple(dict.fromkeys((*self.notes, *right.notes))))

    def __radd__(self, other: str) -> Content:
        return Content(other) + self

    @property
    def text(self) -> str:
        """Return text only when no media part would be lost."""
        if any(not isinstance(part, Text) for part in self.parts):
            raise TypeError("Multimodal content cannot become a plain string.")
        prefix = "\n".join(self.notes) + "\n" if self.notes else ""
        return prefix + "".join(part.text for part in self.parts)

    def __str__(self) -> str:
        return self.text

    def strip(self) -> Content:
        """Remove outer text whitespace without removing attachments."""
        parts = list(self.parts)
        if parts and isinstance(parts[0], Text):
            parts[0] = Text(parts[0].text.lstrip())
        if parts and isinstance(parts[-1], Text):
            parts[-1] = Text(parts[-1].text.rstrip())
        return Content(tuple(parts), notes=self.notes)


@dataclass(frozen=True)
class ToolCall:
    """A requested function with its original call ID and argument text.

    Argument parsing belongs to the granted tool owner. Invalid JSON remains
    available for an explicit rejection instead of becoming guessed arguments.
    """

    id: str
    name: str
    arguments: str


@dataclass(frozen=True, init=False)
class Turn:
    """One role-bound content fragment with optional tool protocol fields.

    Tool calls stay on assistant turns. Tool results retain their call ID.
    Creating a turn does not encode media or perform network operations.
    """

    role: Role
    content: Content
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str = ""
    name: str = ""
    provider_state: dict | None = None

    def __init__(self, role: Role | str, content: Content | str = "", tool_calls: tuple[ToolCall, ...] = (),
                 tool_call_id: str = "", name: str = "", provider_state: dict | None = None) -> None:
        object.__setattr__(self, "role", Role(role))
        object.__setattr__(self, "content", Content(content) if isinstance(content, str) else content)
        object.__setattr__(self, "tool_calls", tool_calls)
        object.__setattr__(self, "tool_call_id", tool_call_id)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "provider_state", provider_state)
        if not isinstance(self.content, Content) or not all(isinstance(call, ToolCall) for call in tool_calls):
            raise TypeError("Turn content needs a Content object or text.")
        if self.tool_calls and self.role is not Role.Assistant:
            raise ValueError("Only assistant turns can request tools.")
        if self.role is Role.Tool and not self.tool_call_id:
            raise ValueError("A tool result needs its original call ID.")


def format_content(template: str, **values: Content | str) -> Content:
    """Fill prompt slots without converting media content into text.

    Plain string slots support ordinary format specifications. Content slots
    preserve their ordered parts. Missing keys raise KeyError.
    """
    result = Content(())
    for literal, field, specification, conversion in Formatter().parse(template):
        result += literal
        if field is None:
            continue
        value = values[field]
        if isinstance(value, Content) and not specification and not conversion:
            result += value
        else:
            text = value.text if isinstance(value, Content) else value
            text = Formatter().convert_field(text, conversion) if conversion else text
            result += format(text, specification)
    return result
