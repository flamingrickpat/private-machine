"""Write readable chat and embedding logs before each inference attempt.

The default directory is ``~/.pm/private-machine``. Each attempt has a fresh
``uuid4`` hex string with a ``.md`` suffix. The file holds:

- a metadata header: time, model, endpoint, generation settings, token usage,
  latency, tokens per second, and finish reason,
- the full message list, one section per message,
- the assistant response text and any tool calls.

Long lines wrap at 120 characters, so the file stays readable in a terminal or
an editor. The log is a diagnostic record. It never changes a request and it
never raises into the caller. A write failure becomes one warning on stderr.

Set ``PM_PROMPT_LOG=0`` to disable the log. Set ``PM_PROMPT_LOG_DIR`` to use a
different directory.
"""
from __future__ import annotations

import os
import json
import sys
import textwrap
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, current_thread

MAX_LINE = 120
DEFAULT_DIR = Path.home() / ".pm" / "private-machine"
_EVENT_LOCK = Lock()


def logging_enabled() -> bool:
    """Return False only when the caller explicitly disables the log."""
    return os.environ.get("PM_PROMPT_LOG", "1").strip().lower() not in {"0", "false", "off", "no"}


def log_directory() -> Path:
    """Return the directory for prompt files, honoring ``PM_PROMPT_LOG_DIR``."""
    raw = os.environ.get("PM_PROMPT_LOG_DIR")
    return Path(raw).expanduser() if raw else DEFAULT_DIR


def wrap_text(text: str, *, width: int = MAX_LINE, indent: str = "") -> list[str]:
    """Wrap one text block to ``width`` characters, keeping blank lines.

    The indent counts toward the width, so every returned line is at most
    ``width`` characters long.
    """
    body_width = max(1, width - len(indent))
    lines: list[str] = []
    for raw_line in str(text if text is not None else "").splitlines() or [""]:
        if raw_line.strip() == "":
            lines.append("")
            continue
        for chunk in textwrap.wrap(raw_line, width=body_width,
                                   break_long_words=True, break_on_hyphens=False):
            lines.append(indent + chunk)
    return lines or [""]


def _render_messages(messages: list[dict]) -> list[str]:
    """Render the message list as markdown sections."""
    lines = ["## Messages", ""]
    for index, message in enumerate(messages or [], start=1):
        role = str(message.get("role", "unknown"))
        name = str(message.get("name", "") or "")
        heading = f"### {index}. {role}" + (f" ({name})" if name else "")
        lines.append(heading)
        content = message.get("content")
        if isinstance(content, list):
            content = " ".join(str(part.get("text", part)) if isinstance(part, dict) else str(part)
                               for part in content)
        for line in wrap_text(str(content or ""), indent="  "):
            lines.append(line)
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            lines.append(f"  tool_call {call.get('id', '')}: {function.get('name', '')}")
            for line in wrap_text(str(function.get("arguments", "")), indent="    "):
                lines.append(line)
        if message.get("tool_call_id"):
            lines.append(f"  tool_call_id: {message['tool_call_id']}")
        lines.append("")
    return lines


def _render_response(response: dict | None, error: BaseException | None) -> list[str]:
    """Render the assistant response or the transport error."""
    lines = ["## Response", ""]
    if error is not None:
        lines.append(f"error: {type(error).__name__}: {error}")
        lines.append("")
        return lines
    try:
        message = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        lines.append("(no assistant message)")
        lines.append("")
        return lines
    for line in wrap_text(str(message.get("content") or ""), indent="  "):
        lines.append(line)
    for field in ("reasoning", "reasoning_content", "reasoning_details"):
        if message.get(field):
            lines.extend([f"### {field}", *wrap_text(str(message[field]), indent="  ")])
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        lines.append(f"  tool_call: {function.get('name', '')}")
        for line in wrap_text(str(function.get("arguments", "")), indent="    "):
            lines.append(line)
    lines.append("")
    return lines


def _metadata(connection, generation: dict, response: dict | None, error: BaseException | None,
              elapsed: float, request_count: int | None) -> list[str]:
    """Build the metadata header lines."""
    usage = ((response or {}).get("usage") or {}) if isinstance(response, dict) else {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    tps = ""
    if isinstance(completion_tokens, (int, float)) and elapsed > 0:
        tps = f"{completion_tokens / elapsed:.1f}"
    finish_reason = ""
    try:
        finish_reason = str(response["choices"][0].get("finish_reason", ""))
    except (KeyError, IndexError, TypeError):
        finish_reason = ""
    clean_generation = {key: value for key, value in generation.items() if key != "cancel"}
    raw_lines = [
        f"- time: {datetime.now(timezone.utc).isoformat()}",
        f"- model: {getattr(connection, 'model', '')}",
        f"- endpoint: {getattr(connection, 'base_url', '')}",
        f"- latency_seconds: {elapsed:.2f}",
        f"- tokens_per_second: {tps or 'unknown'}",
        f"- prompt_tokens: {prompt_tokens if prompt_tokens is not None else 'unknown'}",
        f"- completion_tokens: {completion_tokens if completion_tokens is not None else 'unknown'}",
        f"- total_tokens: {total_tokens if total_tokens is not None else 'unknown'}",
        f"- finish_reason: {finish_reason or 'unknown'}",
        f"- generation: {clean_generation}",
    ]
    if request_count is not None:
        raw_lines.append(f"- request_count: {request_count}")
    if error is not None:
        raw_lines.append(f"- status: error ({type(error).__name__})")
    else:
        raw_lines.append("- status: ok")
    lines: list[str] = []
    for raw_line in raw_lines:
        lines.extend(wrap_text(raw_line, indent="  "))
    return lines


def write_prompt_log(*, connection, messages: list[dict], generation: dict | None = None,
                     response: dict | None = None, error: BaseException | None = None,
                     elapsed: float = 0.0, request_count: int | None = None,
                     directory: Path | None = None) -> Path | None:
    """Write one prompt file. Return its path, or None when the log is off.

    The caller must not depend on this function for correctness. A write
    failure is reported on stderr and returns None.
    """
    if not logging_enabled():
        return None
    target_dir = directory or log_directory()
    path = target_dir / f"{uuid.uuid4().hex}.md"
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        body = [
            "# Private-machine prompt log",
            "",
            "## Metadata",
            "",
            *_metadata(connection, generation or {}, response, error, elapsed, request_count),
            "",
            *_render_messages(messages),
            *_render_response(response, error),
        ]
        path.write_text("\n".join(body), encoding="utf-8")
        return path
    except OSError as write_error:  # logging must never break a completion
        print(f"Prompt log write failed: {type(write_error).__name__}: {write_error}", file=sys.stderr)
        return None


__all__ = ["DEFAULT_DIR", "MAX_LINE", "log_directory", "logging_enabled", "wrap_text", "write_prompt_log"]


def _event(path: Path, event: dict) -> None:
    """Append one event to this process's usage file and show its call ID.

    Separate files prevent writes from overlapping across console workers.
    Each process serializes its threads. Provider counts remain unknown until
    the response arrives. The event contains no credentials or prompt text.
    """
    row = {"call_id": path.stem, "time": datetime.now(timezone.utc).isoformat(), **event}
    with _EVENT_LOCK:
        with (path.parent / f"usage-{os.getpid()}.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    usage = row.get("usage") or {}
    print(f"[llm] {row['event']} {path.stem} {row.get('model', '')} "
          f"input={usage.get('prompt_tokens', 'unknown')} "
          f"output={usage.get('completion_tokens', 'unknown')} log={path}", file=sys.stderr)


def start_prompt_log(*, connection, payload: dict | None, route: str,
                     request_count: int, directory: Path | None = None) -> Path | None:
    """Save the exact inference payload before external I/O. Return its log path.

    Chat and embedding attempts share this lifecycle. Retries get distinct IDs.
    Headers and credentials never enter the files. A write error prints a
    warning and returns None. No model request depends on logging success.
    """
    if not logging_enabled() or route not in {"chat/completions", "embeddings"}:
        return None
    path = (directory or log_directory()) / f"{uuid.uuid4().hex}.md"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"endpoint": connection.base_url, "model": connection.model,
                  "route": route, "owner_thread": current_thread().name,
                  "request_count": request_count, "payload": payload}
        path.with_suffix(".json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        generation = {key: value for key, value in (payload or {}).items() if key not in {"messages", "input"}}
        body = ["# Private-machine prompt log", "", "- status: pending",
                f"- owner_thread: {current_thread().name}", f"- route: {route}",
                *[line for line in _metadata(connection, generation, None, None, 0, request_count)
                  if "- status:" not in line], ""]
        if route == "chat/completions":
            body.extend(_render_messages((payload or {}).get("messages", [])))
        else:
            body.extend(["## Embedding input", "", *wrap_text(str((payload or {}).get("input", "")))])
        path.write_text("\n".join(body), encoding="utf-8")
        _event(path, {"event": "start", **{key: value for key, value in record.items() if key != "payload"}})
        return path
    except Exception as error:
        print(f"Prompt log write failed: {type(error).__name__}: {error}", file=sys.stderr)
        return None


def finish_prompt_log(path: Path | None, *, response: dict | None = None,
                      error: BaseException | None = None, elapsed: float = 0.0) -> None:
    """Append provider usage and the response, or record an error or cancellation.

    The raw receipt excludes embedding vectors but retains provider usage.
    An interrupted process leaves a pending prompt. HTTP success does not
    establish a valid workflow result. Write errors cannot mask model errors.
    """
    if path is None:
        return
    try:
        request = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        response = response if isinstance(response, dict) else None
        usage = (response or {}).get("usage") or {}
        output_tokens = usage.get("completion_tokens")
        choices = (response or {}).get("choices") or []
        receipt = {"event": "error" if error is not None else "end", "model": request["model"],
                   "route": request["route"], "owner_thread": request["owner_thread"],
                   "elapsed_seconds": elapsed, "usage": usage,
                   "tokens_per_second": output_tokens / elapsed if isinstance(output_tokens, (int, float)) and elapsed > 0 else None,
                   "finish_reason": choices[0].get("finish_reason") if choices else None,
                   "error": f"{type(error).__name__}: {error}" if error is not None else None}
        raw_response = response if request["route"] == "chat/completions" else {"usage": usage}
        path.with_suffix(".response.json").write_text(
            json.dumps({**receipt, "response": raw_response}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        lines = ["", "## Outcome", "", f"- status: {receipt['event']}",
                 *wrap_text(json.dumps(receipt, ensure_ascii=False, indent=2)), ""]
        if request["route"] == "chat/completions":
            lines.extend(_render_response(response, error))
        with path.open("a", encoding="utf-8") as stream:
            stream.write("\n".join(lines))
        _event(path, receipt)
    except Exception as write_error:
        print(f"Prompt log write failed: {type(write_error).__name__}: {write_error}", file=sys.stderr)


__all__ += ["start_prompt_log", "finish_prompt_log"]
