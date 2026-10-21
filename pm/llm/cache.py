"""Cache keys and atomic publication for local model responses.

Completion counters belong to this process. Different prompts never advance
each other's counters. Timestamp normalization changes keys only.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from threading import Lock

_COUNTERS: dict[str, int] = {}
_LOCK = Lock()
_ISO_TIME = r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:\d{2})?"
_STORY_TIME = r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday), \d{2} [A-Za-z]+ \d{4}, \d{2}:\d{2}"
_RECORD_TIME = re.compile(rf"(?m)^(?P<prefix>(?:\[id \d+\] )?)\({_ISO_TIME}\) ")
_SCENE_TIME = re.compile(rf"(?m)^(?P<prefix>(?:\[\d+\] )?){_STORY_TIME} ")
_RANGE_TIME = re.compile(rf"(?m)^From {_STORY_TIME} to {_STORY_TIME}:$")
_TIME_FIELDS = {"timestamp", "timestamp_world_begin", "timestamp_world_end", "executed_at",
                "started_at", "finished_at", "imported_at", "cycle_time", "recorded_at"}
_CONTEXT_FIELDS = {"evidence", "stimuli", "records", "candidates", "public_context", "table_fields"}


def cache_root() -> Path:
    """Return the shared cache root. PM_CACHE_DIR overrides it for tests."""
    return Path(os.environ.get("PM_CACHE_DIR", str(Path.home() / ".pm/private-machine/cache"))).expanduser()


def model_directory(model: str, root: Path | None = None) -> Path:
    """Return a portable directory name. The full identity remains in the key."""
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", model).strip(".") or "model"
    return (root if root is not None else cache_root()) / name


def normalize_timestamps(text: str) -> str:
    """Remove rendered record times, but retain times inside quoted speech.

    Structured context can follow a legend line. Its timestamp fields and
    table columns are metadata. Text and content fields remain verbatim.
    """
    text = _RECORD_TIME.sub(r"\g<prefix>(<timestamp>) ", text)
    text = _SCENE_TIME.sub(r"\g<prefix><timestamp> ", text)
    text = _RANGE_TIME.sub("From <timestamp> to <timestamp>:", text)

    def normalize(value: object) -> object:
        if isinstance(value, dict):
            columns = value.get("table_fields", [])
            result = {key: ("<timestamp>" if key in _TIME_FIELDS and isinstance(item, str)
                            and re.fullmatch(_ISO_TIME, item) else normalize(item))
                      for key, item in value.items()}
            if columns and "table_rows" in value:
                result["table_rows"] = [["<timestamp>" if column in _TIME_FIELDS and isinstance(item, str)
                                         and re.fullmatch(_ISO_TIME, item) else item
                                         for column, item in zip(columns, row, strict=True)]
                                        for row in value["table_rows"]]
            return result
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value

    # Only parse a complete JSON suffix. Ordinary conversation stays text.
    for match in re.finditer(r"(?m)^\{", text):
        try:
            value = json.loads(text[match.start():])
        except ValueError:
            continue
        if not isinstance(value, dict) or not (_CONTEXT_FIELDS & value.keys()):
            return text
        return text[:match.start()] + json.dumps(normalize(value), ensure_ascii=False, separators=(",", ":"))
    return text


def completion_path(endpoint: str, payload: dict) -> Path:
    """Reserve the next response for this key. Ignore the sampling seed.

    Call once per logical request. Transport retries reuse the returned path.
    """
    normalized = {key: value for key, value in payload.items() if key != "seed"}
    normalized["messages"] = []
    for message in payload.get("messages", []):
        content = message.get("content")
        if message.get("role") != "assistant":
            if isinstance(content, str):
                content = normalize_timestamps(content)
            elif isinstance(content, list):
                content = [{**part, "text": normalize_timestamps(part["text"])}
                           if part.get("type") == "text" else part for part in content]
        normalized["messages"].append({**message, "content": content})
    digest = hashlib.sha256(json.dumps({"endpoint": endpoint, "payload": normalized},
        sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    directory = model_directory(payload.get("model", "model"))
    with _LOCK:
        key = str(directory / digest)
        count = _COUNTERS[key] = _COUNTERS.get(key, 0) + 1
    return directory / f"cache_comp_{digest}_{count}.json"


def publish(path: Path, content: bytes) -> None:
    """Publish complete bytes once. Keep a concurrent writer's complete file.

    A hard link avoids replacement of open files on Windows. Other filesystem
    errors propagate. The temporary file always stays beside the final file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            pass
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
