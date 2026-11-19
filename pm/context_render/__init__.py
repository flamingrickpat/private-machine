"""Render typed context as composable text and media for model turns."""
from __future__ import annotations


from enum import IntFlag, auto
from typing import TYPE_CHECKING


from pm.context_render.turns import Content, Media, MediaKind, Role, Text, ToolCall, Turn, VideoFrame, VideoFrames


if TYPE_CHECKING:
    from pm.context_render.records import RecordEvidence, KnoxelCollection
    from pm.records.base import RecordBase


class ContextKind(IntFlag):
    """Named context sections. Typed renderers select the relevant sections."""

    Facts = auto()
    Emotion = auto()
    Dialogue = auto()
    Memories = auto()
    Ontology = auto()
    All = Facts | Emotion | Dialogue | Memories | Ontology





def render_context(value: Content | str | Media | VideoFrames | RecordEvidence | KnoxelCollection | RecordBase, kinds: ContextKind = ContextKind.All, *, dedupe: bool = True) -> Content:
    from pm.context_render.records import RecordEvidence, KnoxelCollection, render_record, render_collection
    if isinstance(value, Content):
        return value
    if isinstance(value, str):
        return Content(value)
    if isinstance(value, (Media, VideoFrames)):
        return Content(value)
    if isinstance(value, RecordEvidence):
        return render_record(value, kinds, dedupe=dedupe)
    if isinstance(value, KnoxelCollection):
        return render_collection(value, kinds, dedupe=dedupe)
    return Content(value.get_story_element())



__all__ = ["Content", "ContextKind", "Media", "MediaKind", "Role", "Text", "ToolCall", "Turn", "VideoFrame", "VideoFrames", "render_context"]
