"""Resolve operator endpoints and report connection health without activation."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
import socket
import time

import httpx
from pydantic import SecretStr

from pm.llm.config import load_config
from pm.paths import REPO_ROOT

ROOT = REPO_ROOT


def environment_values() -> tuple[dict, str]:
    """Read the selected .env without exporting credentials into global state."""
    candidates = [Path(os.environ["PM_ENV_FILE"])] if os.environ.get("PM_ENV_FILE") else [ROOT / ".env", ROOT.parent / "pm_next/.env"]
    path = next((p for p in candidates if p.is_file()), None)
    values = {}
    if path:
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.removeprefix("export ").split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    values.update(os.environ)
    return values, str(path) if path else "process environment"


def operator_config(config_path: str | None = None):
    """Use project model configuration and explicit embedding .env overrides."""
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


async def connection_status(config_path: str | None = None) -> dict:
    """Probe model HTTP endpoints, MCP tool discovery, and Minecraft TCP only."""
    config, values, source = operator_config(config_path)
    rows = []
    async def http_probe(name, connection):
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
    rows.extend(await asyncio.gather(http_probe("Chat model", config.chat["Default"]), http_probe("Embeddings", config.embedding)))
    mcp = values.get("MINECRAFT_MCP_URL") or f"http://127.0.0.1:{values.get('PM_MC_MCP_PORT', '6767')}/mcp"
    try:
        from fastmcp import Client
        async def discover():
            async with Client(mcp) as client:
                return [t.name for t in await client.list_tools()]
        tools = await asyncio.wait_for(discover(), timeout=5)
        rows.append({"name": "Minecraft MCP", "endpoint": mcp, "state": "ready", "tools": tools})
    except Exception as exc:
        rows.append({"name": "Minecraft MCP", "endpoint": mcp, "state": "not ready", "error_type": type(exc).__name__})
    host, port = values.get("PM_MINECRAFT_HOST", "127.0.0.1"), int(values.get("PM_MINECRAFT_PORT", "12345"))
    def tcp_probe():
        try:
            with socket.create_connection((host, port), timeout=3):
                return "TCP reachable"
        except OSError:
            return "unreachable"
    rows.append({"name": "Minecraft server", "endpoint": f"{host}:{port}", "state": await asyncio.to_thread(tcp_probe)})
    return {"source": source, "checked_at": time.time(), "connections": rows}


def default_config_path() -> str:
    if os.environ.get("PM_MODEL_CONFIG"):
        return os.environ["PM_MODEL_CONFIG"]
    if Path("config.local.yaml").is_file():
        return "config.local.yaml"
    return "config.example.yaml"
