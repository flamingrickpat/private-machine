"""Compatibility entry point for the original text-only renderer."""
from pathlib import Path

from pm.context_render import render_context


def render_object_for_llm(value: object, schema_path: str | Path) -> str:
    """Return text; reject media instead of silently discarding attachments."""
    return render_context(value, schema_path=schema_path).text
