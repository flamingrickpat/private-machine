"""Render model input with rules from a caller-owned JSON schema file."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
import json
from pathlib import Path
from collections.abc import Callable
from collections.abc import Mapping

from pydantic import BaseModel


def render_text(value: object, schema_path: str | Path | Mapping, *, transform: Callable[[object], object] | None = None,
                include_legend: bool = True) -> str:
    """Render an object without changing its stored values.

    The schema controls omitted fields, numeric precision, and table encoding.
    Inputs can contain JSON values, dataclasses, or Pydantic models. Unknown
    fields remain visible. Unsupported objects raise TypeError. File errors
    and invalid schemas propagate to the caller. This function only reads files.
    Rendering does not validate evidence or change acceptance conditions.
    """
    schema = schema_path if isinstance(schema_path, Mapping) else json.loads(Path(schema_path).read_text(encoding="utf-8"))
    if schema["version"] != 1:
        raise ValueError("Unsupported LLM rendering schema version.")
    omitted = set(schema.get("omit_fields", []))
    empty_fields = set(schema.get("omit_empty_fields", []))
    exact_fields = set(schema.get("exact_fields", []))
    key_lists = set(schema.get("key_list_fields", []))
    tables = set(schema.get("table_fields", []))
    vectors = schema.get("vector_fields", {})
    precision = schema.get("float_digits")
    field_precision = schema.get("field_digits", {})
    selected_fields = schema.get("select_fields", {})
    aliases = schema.get("field_aliases", {})

    def project(item: object, field: str = "", exact: bool = False) -> object:
        """Apply field rules recursively while retaining list order."""
        exact = exact or field in exact_fields
        if transform is not None:
            item = transform(item)
        if isinstance(item, BaseModel):
            item = dict(item)
        elif is_dataclass(item) and not isinstance(item, type):
            item = asdict(item)
        if isinstance(item, dict):
            if field in key_lists:
                return list(item)
            selected = selected_fields.get(field)
            return {aliases.get(key, key): project(child, key, exact) for key, child in item.items()
                    if key not in omitted and (selected is None or key in selected)
                    and not (key in empty_fields and child in (None, {}, [], ""))}
        if isinstance(item, (list, tuple)):
            axes = vectors.get(field)
            if axes and item and all(isinstance(row, dict) and set(row) == set(axes) for row in item):
                return [[project(row[axis], axis, exact) for axis in axes] for row in item]
            rows = [project(row, field, exact) for row in item]
            if (field in tables and len(rows) >= schema.get("table_min_rows", 4)
                    and all(isinstance(row, dict) and row.keys() == rows[0].keys() for row in rows)):
                columns = list(rows[0])
                return {"table_fields": columns, "table_rows": [[row[key] for key in columns] for row in rows]}
            return rows
        if isinstance(item, float):
            digits = field_precision.get(field, precision)
            if not exact and digits is not None:
                rounded = round(item, digits)
                return int(rounded) if rounded.is_integer() else rounded
        if isinstance(item, datetime):
            return item.isoformat(timespec="seconds")
        if item is None or isinstance(item, (str, int, float, bool)):
            return item
        raise TypeError(f"Cannot render {type(item).__name__} for an LLM.")

    rendered = json.dumps(project(value), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    legend = schema.get("legend", "")
    return legend + "\n" + rendered if legend and include_legend else rendered
