"""Render typed context as composable text and media for model turns."""
from __future__ import annotations

from enum import IntFlag, auto
from functools import singledispatch
from pathlib import Path
import copy
import json

from pydantic import BaseModel

from pm.context_render.rendering import render_text
from pm.context_render.turns import Content, Media, MediaKind, Role, Text, ToolCall, Turn, VideoFrame, VideoFrames


class ContextKind(IntFlag):
    """Named context sections. Typed renderers select the relevant sections."""

    Facts = auto()
    Emotion = auto()
    Dialogue = auto()
    Memories = auto()
    Ontology = auto()
    All = Facts | Emotion | Dialogue | Memories | Ontology


DEFAULT_SCHEMA = Path(__file__).with_name("default.json")


@singledispatch
def render_context(value: object, kinds: ContextKind = ContextKind.All, *, schema_path: str | Path | None = None,
                   dedupe: bool = True) -> Content:
    """Render selected context without provider encoding or input mutation.

    Type owners register specialized renderers with render_context.register.
    Generic values use the supplied schema or the runtime's default schema.
    Typed media remains binary content, with numbered references in the text.
    Unsupported objects and schema errors propagate before inference.
    This function reads schema files but does not read or download media.
    """
    if isinstance(value, Content):
        return value
    attachments: list[Media | VideoFrames] = []
    records: dict[int | str, list[dict]] = {}

    def extract(item: object) -> object:
        """Preserve typed attachments before model or dataclass serialization."""
        if isinstance(item, (Media, VideoFrames)):
            attachments.append(item)
            return f"[attachment {len(attachments)}]"
        raw = item.model_dump(mode="python") if isinstance(item, BaseModel) else item
        if dedupe and isinstance(raw, dict):
            record_id = raw.get("knoxel_id") if "text" in raw else raw.get("id") if "content" in raw else None
            if isinstance(record_id, (int, str)):
                versions = records.setdefault(record_id, [])
                # Once an ID has multiple versions, an ID-only reference
                # cannot identify which version it means. Keep full records.
                if len(versions) == 1 and raw == versions[0]:
                    return {"record_ref": record_id}
                if not any(raw == previous for previous in versions):
                    versions.append(copy.deepcopy(raw))
        return item

    schema = json.loads(Path(schema_path or DEFAULT_SCHEMA).read_text(encoding="utf-8"))
    if schema_path is None and not kinds & ContextKind.Emotion:
        # Factual callers can retain meaningful measurement precision. The
        # emotion profile rounds display values without changing stored state.
        schema = {**schema, "float_digits": None,
                  "legend": "Tables: zip table_rows with table_fields. IDs retain original values. record_ref points to an earlier identical record. Attachment references identify the following media parts."}
    legend = schema.get("legend", "")
    result = Content(render_text(value, schema, transform=extract, include_legend=False), notes=(legend,) if legend else ())
    for number, attachment in enumerate(attachments, 1):
        result += f"\nAttachment {number}:\n"
        result += Content(attachment)
    return result


__all__ = ["Content", "ContextKind", "Media", "MediaKind", "Role", "Text", "ToolCall", "Turn", "VideoFrame", "VideoFrames", "render_context"]
