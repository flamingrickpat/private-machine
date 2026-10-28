"""Resolve operator endpoints and report connection health without activation.

The operator console shows which model and embedding endpoints this machine
would use, and whether they answer, before any session starts. This module
owns that resolution: it merges the project model configuration with the
``.env`` overrides, then probes the HTTP endpoints plus the probes of the
loaded assembly. Nothing here starts a model, writes memory, or exports a
credential into global process state.

Failure stays visible: a probe that cannot answer becomes one status row with
the exception name, and a missing credential raises from the configuration
layer. The assembly supplies its own probes, so ``pm`` holds no domain
endpoint knowledge.
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
import os
from pathlib import Path
import time
from typing import TypedDict

import httpx
from pydantic import JsonValue, SecretStr

from pm.llm.config import ConnectionConfig, LlmPreset, ModelConfig, load_config
from pm.paths import REPO_ROOT

ROOT = REPO_ROOT

#: Assembly-supplied probe: receives the operator values, returns status rows.
Probes = Callable[[Mapping[str, str]], Awaitable[Sequence[Mapping[str, JsonValue]]]]


class ConnectionRow(TypedDict):
    """One probed endpoint as the console shows it.

    ``state`` is ``reachable``, ``HTTP <code>``, or the exception class name of
    the failure. ``seconds`` is the wall time of that one probe.
    """

    name: str
    endpoint: str
    model: str
    state: str
    seconds: float


class ConnectionReport(TypedDict):
    """The connection health of one deployment: config source and probe time.

    ``connections`` holds this module's own rows and the assembly's rows, so
    it is typed as a JSON object mapping rather than one fixed row type.
    """

    source: str
    checked_at: float
    connections: list[Mapping[str, object]]


def environment_values() -> tuple[dict[str, str], str]:
    """Read the selected .env without exporting credentials into global state.

    ``PM_ENV_FILE`` or the repository ``.env`` is used first; the legacy
    ``../pm_next/.env`` is the fallback. Process environment values win.
    """
    from pm.llm.env_file import env_file_path, read_env_file
    candidates = [env_file_path()] if os.environ.get("PM_ENV_FILE") else [env_file_path(), ROOT.parent / "pm_next/.env"]
    path = next((p for p in candidates if p.is_file()), None)
    values = read_env_file(path) if path else {}
    values.update(os.environ)
    return values, str(path) if path else "process environment"


def operator_config(config_path: str | None = None) -> tuple[ModelConfig, dict[str, str], str]:
    """Use project model configuration and explicit embedding .env overrides.

    Returns ``(config, values, source)``. ``values`` are the merged operator
    values and ``source`` names the file they came from. The embedding
    connection takes ``OPENAI_EMBEDDING_*`` overrides, so an embedding endpoint
    can differ from the chat endpoint without editing the config file.
    """
    values, source = environment_values()
    config = load_config(config_path or default_config_path())
    embedding = config.embedding.model_dump()
    for key, field in [("OPENAI_EMBEDDING_API_BASE", "base_url"), ("OPENAI_EMBEDDING_MODEL", "model")]:
        if values.get(key):
            embedding[field] = values[key]
    credential_name = embedding.get("api_key_env")
    if credential_name and os.environ.get(credential_name):
        embedding.update(api_key=SecretStr(os.environ[credential_name]), api_key_env=None)
    elif values.get("OPENAI_EMBEDDING_API_KEY"):
        embedding.update(api_key=SecretStr(values["OPENAI_EMBEDDING_API_KEY"]), api_key_env=None)
    elif embedding.get("api_key_env") and values.get(embedding["api_key_env"]):
        embedding.update(api_key=SecretStr(values[embedding["api_key_env"]]), api_key_env=None)
    if values.get("OPENAI_EMBEDDING_DIMENSIONS"):
        embedding["dimensions"] = int(values["OPENAI_EMBEDDING_DIMENSIONS"])
    config.embedding = type(config.embedding).model_validate(embedding)
    return config, values, source


async def connection_status(config_path: str | None = None, probes: Probes | None = None) -> ConnectionReport:
    """Probe model HTTP endpoints, then the assembly's own probes.

    ``probes`` is an async callable that receives the operator values and
    returns status rows. The runtime knows no domain endpoint itself. Every
    probe answers with a row, so one dead endpoint does not hide the others.
    """
    config, values, source = operator_config(config_path)
    rows: list[Mapping[str, object]] = []

    async def http_probe(name: str, connection: ConnectionConfig) -> ConnectionRow:
        """Ask one endpoint for its model list without activating a model."""
        started = time.perf_counter()
        try:
            resolved = connection.resolve()
            headers = {"Authorization": "Bearer " + resolved.api_key.get_secret_value()} if resolved.api_key.get_secret_value() else {}
            async with httpx.AsyncClient(timeout=4) as client:
                response = await client.get(resolved.base_url + "/models", headers=headers)
            status = "reachable" if response.is_success else f"HTTP {response.status_code}"
        except Exception as exc:
            status = type(exc).__name__
        return {"name": name, "endpoint": connection.base_url, "model": connection.model,
                "state": status, "seconds": round(time.perf_counter() - started, 3)}
    chat_connection = config.chat[LlmPreset.Default]
    rows.extend(await asyncio.gather(http_probe("Chat model", chat_connection),
                                     http_probe("Embeddings", config.embedding)))
    if probes is not None:
        rows.extend(await probes(values))
    return {"source": source, "checked_at": time.time(), "connections": rows}


def default_config_path() -> str:
    """Return the model config file that an unnamed launch would use."""
    if os.environ.get("PM_MODEL_CONFIG"):
        return os.environ["PM_MODEL_CONFIG"]
    if Path("config.local.yaml").is_file():
        return "config.local.yaml"
    return "config.example.yaml"
