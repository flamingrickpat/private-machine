"""Select the local model YAML and report endpoint health without activation."""
from __future__ import annotations

import asyncio
from pathlib import Path
import time
import httpx
from pydantic import BaseModel

from pm.llm.config import ConnectionConfig, LlmPreset, ModelConfig
from pm.paths import REPO_ROOT

class ConnectionRow(BaseModel):
    """One probed endpoint as the console shows it.

    ``state`` is ``reachable``, ``HTTP <code>``, or the exception class name of
    the failure. ``seconds`` is the wall time of that one probe.
    """

    name: str
    endpoint: str
    model: str
    state: str
    seconds: float


class ConnectionReport(BaseModel):
    """The connection health of one deployment: config source and probe time.

    Each model endpoint supplies one connection row.
    """

    source: str
    checked_at: float
    connections: list[ConnectionRow]


async def probe_model_endpoint(name: str, connection: ConnectionConfig) -> ConnectionRow:
    """Ask one configured endpoint for its model list without activation."""
    started = time.perf_counter()
    resolved = connection.resolve()
    key = resolved.api_key.get_secret_value()
    headers = {"Authorization": "Bearer " + key} if key else {}
    try:
        async with httpx.AsyncClient(timeout=4) as client:
            response = await client.get(resolved.base_url + "/models", headers=headers)
        status = "reachable" if response.is_success else f"HTTP {response.status_code}"
    except httpx.HTTPError as error:
        status = type(error).__name__
    return ConnectionRow(name=name, endpoint=connection.base_url, model=connection.model,
                         state=status, seconds=round(time.perf_counter() - started, 3))


async def connection_status(config: ModelConfig, source: Path) -> ConnectionReport:
    """Probe the configured model endpoints and report each result."""
    chat = config.chat[LlmPreset.Default]
    rows = await asyncio.gather(probe_model_endpoint("Chat model", chat),
                                probe_model_endpoint("Embeddings", config.embedding))
    return ConnectionReport(source=str(source), checked_at=time.time(), connections=list(rows))


def default_config_path() -> str:
    """Select the main local configuration when the caller omits --config."""
    for filename in ("config.local.yaml", "config.yaml"):
        path = REPO_ROOT / filename
        if path.is_file():
            return str(path)
    return str(REPO_ROOT / "config.example.yaml")
