"""Encode neutral turns at the OpenAI-compatible provider boundary."""
from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, TypeAlias

from pm.context_render.turns import Content, Media, MediaKind, Role, Text, ToolCall, Turn, VideoFrames


TurnInput: TypeAlias = Turn | Mapping[str, Any]


class PromptEncoder(Protocol):
    """A backend owns the conversion from typed turns to its input payload."""

    def encode(self, turns: Sequence[TurnInput]) -> dict[str, Any]: ...


def tool_calls_from_wire(calls: Sequence[Mapping[str, Any]]) -> tuple[ToolCall, ...]:
    """Decode provider function requests without parsing their argument text."""
    return tuple(ToolCall(call["id"], call["function"]["name"], call["function"]["arguments"]) for call in calls)


def normalize_turns(turns: Sequence[TurnInput]) -> list[Turn]:
    """Accept legacy wire turns at migration boundaries without losing fields.

    Unsupported content parts raise ValueError. Typed turns pass unchanged.
    Tool call identities and provider reasoning state remain available.
    """
    result = []
    for turn in turns:
        if isinstance(turn, Turn):
            result.append(turn)
            continue
        content = turn.get("content") or ""
        if isinstance(content, list):
            parts = []
            for part in content:
                if part["type"] == "text":
                    parts.append(Text(part["text"]))
                elif part["type"] == "image_url":
                    image = part["image_url"]
                    parts.append(Media(MediaKind.Image, "image/*", url=image["url"], detail=image.get("detail", "auto")))
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

    def encode(self, turns: Sequence[TurnInput]) -> dict[str, Any]:
        return {"messages": [self.message(turn) for turn in normalize_turns(turns)]}

    def message(self, turn: Turn) -> dict[str, Any]:
        content = turn.content
        if all(isinstance(part, Text) for part in content.parts):
            value = content.text
        else:
            if turn.role is not Role.User:
                raise ValueError("The OpenAI chat adapter accepts input media only on user turns.")
            value = [{"type": "text", "text": "\n".join(content.notes) + "\n"}] if content.notes else []
            for part in content.parts:
                if isinstance(part, Text):
                    value.append({"type": "text", "text": part.text})
                elif isinstance(part, VideoFrames):
                    for frame in part.frames:
                        value.append({"type": "text", "text": f"Video frame at {frame.seconds:g} seconds:"})
                        value.append(self.media(frame.image))
                else:
                    value.append(self.media(part))
        result = {**(turn.provider_state or {}), "role": turn.role.value, "content": value}
        if turn.tool_calls:
            result["tool_calls"] = [{"id": call.id, "type": "function",
                "function": {"name": call.name, "arguments": call.arguments}} for call in turn.tool_calls]
        if turn.tool_call_id:
            result["tool_call_id"] = turn.tool_call_id
        if turn.name:
            result["name"] = turn.name
        return result

    def media(self, part: Media) -> dict[str, Any]:
        encoded = base64.b64encode(part.data).decode("ascii") if part.data is not None else None
        if part.kind is MediaKind.Image:
            if part.data is not None and part.mime_type not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
                raise ValueError("OpenAI chat image bytes need PNG, JPEG, WebP, or GIF media.")
            url = part.url if part.url is not None else f"data:{part.mime_type};base64,{encoded}"
            return {"type": "image_url", "image_url": {"url": url, "detail": part.detail}}
        if part.kind is MediaKind.Audio:
            formats = {"audio/wav": "wav", "audio/mpeg": "mp3", "audio/mp3": "mp3"}
            if part.url is not None or part.mime_type not in formats:
                raise ValueError("OpenAI chat audio needs WAV or MP3 bytes.")
            return {"type": "input_audio", "input_audio": {"data": encoded, "format": formats[part.mime_type]}}
        if part.kind is MediaKind.Video:
            raise ValueError("This chat adapter needs explicit sampled VideoFrames. Native video needs another backend adapter.")
        if part.url is not None:
            raise ValueError("OpenAI chat file input needs captured bytes.")
        return {"type": "file", "file": {"filename": part.name or "attachment",
                                          "file_data": f"data:{part.mime_type};base64,{encoded}"}}
