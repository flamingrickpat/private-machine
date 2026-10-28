"""Encode neutral turns at the OpenAI-compatible provider boundary.

This module is the only place where a typed
:class:`~pm.context_render.turns.Turn` becomes provider JSON, and the only
place where attachment bytes become base64. :class:`OpenAIChatEncoder` builds
the ``messages`` list of one chat-completions request body:

- A text-only turn keeps a plain string content, so a text prompt keeps its
  exact bytes and its provider cache entry.
- A turn with attachments becomes an ordered list of content parts. An image
  travels as a data URL or as an explicit URL, audio as WAV or MP3 bytes,
  another captured file as a ``file`` part, and sampled video as image parts
  with their explicit time offsets.
- Unsupported modality-role pairs and unsupported audio formats raise
  ``ValueError`` before network I/O. No request leaves with lost content.

Provider state fields (for example the signed reasoning details of an
assistant turn) pass through the message mapping unchanged, so a message stays
an open mapping at the outbound boundary. The :class:`TypedDict` shapes below
therefore describe the fields this module owns, not the provider extras.

:func:`normalize_turns` accepts legacy wire turns at migration boundaries.
"""
from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from io import BytesIO as _BytesIO  # noqa: N813
from typing import Any, Literal, Protocol, TypeAlias, TypedDict

from pm.context_render.turns import (
    Content,
    ContentPart,
    Media,
    MediaKind,
    Role,
    Text,
    ToolCall,
    Turn,
    VideoFrames,
)


TurnInput: TypeAlias = Turn | Mapping[str, Any]
"""One prompt turn: a typed turn, or a legacy wire mapping at a boundary."""

#: Image media types that the chat adapter sends as image bytes.
OPENAI_IMAGE_MIME_TYPES = frozenset({"image/png", "image/jpeg", "image/webp", "image/gif"})

#: Audio media types and the short format name the audio part requires.
OPENAI_AUDIO_FORMATS: Mapping[str, str] = {
    "audio/wav": "wav",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
}


class OpenAITextPart(TypedDict):
    """One literal text segment of a message body."""

    type: Literal["text"]
    text: str


class OpenAIImageUrl(TypedDict):
    """The image reference of one ``image_url`` part.

    ``url`` is either a data URL with base64 bytes or an explicit remote URL.
    ``detail`` is the provider's rendering hint for that image.
    """

    url: str
    detail: str


class OpenAIImagePart(TypedDict):
    """One image attachment part."""

    type: Literal["image_url"]
    image_url: OpenAIImageUrl


class OpenAIAudioPayload(TypedDict):
    """The base64 bytes and short format name of one audio part."""

    data: str
    format: str


class OpenAIInputAudioPart(TypedDict):
    """One audio attachment part."""

    type: Literal["input_audio"]
    input_audio: OpenAIAudioPayload


class OpenAIFilePayload(TypedDict):
    """The filename and data URL of one file attachment part."""

    filename: str
    file_data: str


class OpenAIFilePart(TypedDict):
    """One captured-file attachment part."""

    type: Literal["file"]
    file: OpenAIFilePayload


OpenAIContentPart: TypeAlias = OpenAITextPart | OpenAIImagePart | OpenAIInputAudioPart | OpenAIFilePart
"""One content part of a message body that carries attachments."""


class OpenAIFunctionCall(TypedDict):
    """The requested function name and its unparsed argument text."""

    name: str
    arguments: str


class OpenAIToolCall(TypedDict):
    """One requested tool call on an assistant message."""

    id: str
    type: Literal["function"]
    function: OpenAIFunctionCall


class OpenAIRequest(TypedDict):
    """The chat-completions body that carries the encoded turns.

    ``messages`` holds one mapping per turn. Provider state fields stay in
    those messages, so the mapping itself is not a closed record here.
    """

    messages: list[dict[str, object]]


def _base64_data(part: Media) -> str:
    """Return the base64 text of a part's captured bytes.

    Every caller has already rejected a part that only references a URL, so a
    missing byte payload here is a broken media contract, not a user error.
    """
    if part.data is None:
        raise ValueError("This provider part needs captured bytes, not a remote URL.")
    return base64.b64encode(part.data).decode("ascii")


def _png_data_url(url: str) -> str:
    """Convert a data URL image to PNG when the format is unsupported by llama.cpp.

    Returns the unchanged URL for formats that stb_image can decode natively,
    or a new data URL with PNG-encoded bytes when conversion was needed.
    """
    if not url.startswith("data:"):
        return url
    media_type = url.split(",")[0].split(":", 1)[1].split(";")[0]
    b64_data = url.split(",", 1)[1]
    raw = base64.b64decode(b64_data, validate=True)

    # WebP — not supported by stb_image.
    if raw[:4] == b"RIFF" and len(raw) >= 12 and raw[8:12] == b"WEBP":
        from PIL import Image as _PILImage  # noqa: N813
        img = _PILImage.open(_BytesIO(raw))
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        buf = _BytesIO()
        img.save(buf, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
    # BMP — also unsupported by stb_image.
    if raw[:2] == b"BM":
        from PIL import Image as _PILImage  # noqa: N813
        img = _PILImage.open(_BytesIO(raw))
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        buf = _BytesIO()
        img.save(buf, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
    # GIF — convert to avoid palette issues.
    if raw[:6] in (b"GIF87a", b"GIF89a"):
        from PIL import Image as _PILImage  # noqa: N813
        img = _PILImage.open(_BytesIO(raw))
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        buf = _BytesIO()
        img.save(buf, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
    # PNG and JPEG are natively supported.
    return url


class PromptEncoder(Protocol):
    """A backend owns the conversion from typed turns to its input payload."""

    def encode(self, turns: Sequence[TurnInput]) -> OpenAIRequest:
        """Return the provider request body for these turns."""
        ...


def tool_calls_from_wire(calls: Sequence[Mapping[str, Any]]) -> tuple[ToolCall, ...]:
    """Decode provider function requests without parsing their argument text."""
    return tuple(ToolCall(call["id"], call["function"]["name"], call["function"]["arguments"]) for call in calls)


def normalize_turns(turns: Sequence[TurnInput]) -> list[Turn]:
    """Accept legacy wire turns at migration boundaries without losing fields.

    Unsupported content parts raise ValueError. Typed turns pass unchanged.
    Tool call identities and provider reasoning state remain available.
    """
    result: list[Turn] = []
    for turn in turns:
        if isinstance(turn, Turn):
            result.append(turn)
            continue
        content = turn.get("content") or ""
        if isinstance(content, list):
            parts: list[ContentPart] = []
            for part in content:
                if part["type"] == "text":
                    parts.append(Text(part["text"]))
                elif part["type"] == "image_url":
                    image = part["image_url"]
                    url = _png_data_url(image["url"])
                    parts.append(Media(MediaKind.Image, "image/*", url=url, detail=image.get("detail", "auto")))
                elif part["type"] == "input_audio":
                    audio = part["input_audio"]
                    parts.append(Media.from_base64(MediaKind.Audio, audio["data"], "audio/" + audio["format"]))
                else:
                    raise ValueError(f"Unsupported legacy content part: {part['type']}")
            content = Content(tuple(parts))
        known = {"role", "content", "tool_calls", "tool_call_id", "name"}
        result.append(Turn(Role(turn["role"]), content, tool_calls_from_wire(turn.get("tool_calls") or []),
                           turn.get("tool_call_id") or "", turn.get("name") or "",
                           {key: value for key, value in turn.items() if key not in known} or None))
    return result


class OpenAIChatEncoder:
    """Encode text, images, WAV/MP3 audio, files, and sampled video frames.

    Binary media becomes base64 only here. Unsupported modality-role pairs
    and audio formats raise ValueError before network I/O. Video frames become
    ordered image parts with their explicit time offsets.
    """

    def encode(self, turns: Sequence[TurnInput]) -> OpenAIRequest:
        """Return the chat-completions body for these turns."""
        messages = [self.message(turn) for turn in normalize_turns(turns)]
        return {"messages": messages}

    def message(self, turn: Turn) -> dict[str, object]:
        """Return one provider message mapping for ``turn``.

        ``role`` and ``content`` are always present. ``tool_calls``,
        ``tool_call_id``, and ``name`` appear only when the turn carries them.
        Provider state fields come first, so a provider sees its own fields
        unchanged and in their original order.
        """
        content = turn.content
        value: str | list[OpenAIContentPart]
        if all(isinstance(part, Text) for part in content.parts):
            value = content.text
        else:
            if turn.role is not Role.User:
                raise ValueError("The OpenAI chat adapter accepts input media only on user turns.")
            parts: list[OpenAIContentPart] = []
            if content.notes:
                # Unattached notes (for example image provenance) lead the parts.
                parts.append({"type": "text", "text": "\n".join(content.notes) + "\n"})
            for part in content.parts:
                if isinstance(part, Text):
                    parts.append({"type": "text", "text": part.text})
                elif isinstance(part, VideoFrames):
                    # Sampled video travels as ordered images with their offsets.
                    for frame in part.frames:
                        parts.append({"type": "text", "text": f"Video frame at {frame.seconds:g} seconds:"})
                        parts.append(self.media(frame.image))
                else:
                    parts.append(self.media(part))
            value = parts
        result: dict[str, object] = {**(turn.provider_state or {}), "role": turn.role.value, "content": value}
        if turn.tool_calls:
            result["tool_calls"] = [OpenAIToolCall(id=call.id, type="function",
                                                   function=OpenAIFunctionCall(name=call.name,
                                                                               arguments=call.arguments))
                                    for call in turn.tool_calls]
        if turn.tool_call_id:
            result["tool_call_id"] = turn.tool_call_id
        if turn.name:
            result["name"] = turn.name
        return result

    def media(self, part: Media) -> OpenAIContentPart:
        """Return one provider content part for a part's bytes or URL.

        Raises ValueError when this adapter cannot carry the attachment as it
        is: unsupported image bytes, audio that is not WAV or MP3, native
        video, or a file that only has a remote reference.
        """
        if part.kind is MediaKind.Image:
            if part.data is not None and part.mime_type not in OPENAI_IMAGE_MIME_TYPES:
                raise ValueError("OpenAI chat image bytes need PNG, JPEG, WebP, or GIF media.")
            url = part.url if part.url is not None else f"data:{part.mime_type};base64,{_base64_data(part)}"
            return {"type": "image_url", "image_url": {"url": url, "detail": part.detail}}
        if part.kind is MediaKind.Audio:
            if part.url is not None or part.mime_type not in OPENAI_AUDIO_FORMATS:
                raise ValueError("OpenAI chat audio needs WAV or MP3 bytes.")
            return {"type": "input_audio", "input_audio": {"data": _base64_data(part),
                                                             "format": OPENAI_AUDIO_FORMATS[part.mime_type]}}
        if part.kind is MediaKind.Video:
            raise ValueError("This chat adapter needs explicit sampled VideoFrames. Native video needs another backend adapter.")
        if part.url is not None:
            raise ValueError("OpenAI chat file input needs captured bytes.")
        return {"type": "file", "file": {"filename": part.name or "attachment",
                                          "file_data": f"data:{part.mime_type};base64,{_base64_data(part)}"}}
