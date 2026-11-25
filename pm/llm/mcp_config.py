"""The typed ``tool_call`` section of the model configuration: MCP servers for the companion.

The section lives in the same YAML file as the chat and embedding
connections (``--config``). Each setup or character home therefore gives the
companion its own MCP servers through its own config file. Example::

    tool_call:
      timeout_seconds: 1800
      mcp_servers:
        - name: image_gen
          url: http://127.0.0.1:9877/mcp
          headers:
            Authorization: Bearer your-local-token
          request_timeout_seconds: 600

Keep literal credential headers in the ignored local YAML file.
"""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pm.constants import (MINECRAFT_REQUEST_SECONDS, TOOL_CALL_REQUEST_SECONDS, TOOL_CALL_SESSION_SECONDS,
                          TOOL_CALL_TURN_LIMIT)

#: One environment reference in a header value. Group 1 is the variable name.
ENV_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class McpConfigError(ValueError):
    """The MCP configuration cannot be used as written, for example a missing header variable."""


class McpServerConfig(BaseModel):
    """One HTTP MCP server that the tool session can use.

    ``name`` is unique and becomes the tool prefix in the native session, so
    tools with the same name on two servers stay distinct. Reports keep the
    name, never the URL or the headers. ``request_timeout_seconds`` limits one
    tool request.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    url: str = Field(min_length=1, repr=False)
    headers: dict[str, str] = Field(default_factory=dict, repr=False)
    request_timeout_seconds: int = Field(default=TOOL_CALL_REQUEST_SECONDS, gt=0, strict=True)

    @field_validator("url")
    @classmethod
    def _http_url(cls, value: str) -> str:
        parts = urlsplit(value)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            raise ValueError("An MCP server needs an HTTP or HTTPS URL.")
        return value

    def resolved_headers(self) -> dict[str, str]:
        """Return literal headers. Reject old environment references before discovery."""
        for value in self.headers.values():
            if ENV_REFERENCE.search(value):
                raise McpConfigError(f"MCP server {self.name!r}: set literal headers in your local model YAML.")
        return self.headers.copy()


class ToolCallConfig(BaseModel):
    """The ``tool_call`` section: every MCP server of one tool session, and the session limits.

    An empty server list means that the companion has no tools: the
    application then does not register the ``tool_call`` workflow, so
    routing never offers it. ``model_connection`` names the chat connection
    of the native session. ``timeout_seconds`` is the session deadline.
    ``gwt_turn_limit`` counts the complete GWT turns that the session reads.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    model_connection: str = Field(default="Default", min_length=1)
    timeout_seconds: int = Field(default=TOOL_CALL_SESSION_SECONDS, gt=0, strict=True)
    gwt_turn_limit: int = Field(default=TOOL_CALL_TURN_LIMIT, gt=0, strict=True)
    mcp_servers: list[McpServerConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_names(self) -> ToolCallConfig:
        names = [server.name for server in self.mcp_servers]
        if len(set(names)) != len(names):
            raise ValueError("Each MCP server needs a unique name.")
        return self

    def validate_headers(self) -> None:
        """Reject any old environment reference before opening a connection."""
        for server in self.mcp_servers:
            server.resolved_headers()


class MinecraftConfig(BaseModel):
    """The ``minecraft`` section: the one Minecraft body endpoint of this character.

    The section is optional. Without it the application starts no Minecraft
    body: no world poll, no body tools in the assistant, and no body route.
    With it the assembly asks the endpoint once at startup and stops when
    nothing answers. A wrong URL then fails before the first cycle instead of
    surfacing much later as an unverified task.

    ``request_timeout_seconds`` limits one body request. A long mining or
    building command needs room. ``probe_timeout_seconds`` limits only the
    startup check, so a dead endpoint fails fast.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    mcp_url: str = Field(min_length=1)
    request_timeout_seconds: int = Field(default=MINECRAFT_REQUEST_SECONDS, gt=0, strict=True)
    probe_timeout_seconds: float = Field(default=5.0, gt=0, le=120)
    lease_directory: Path | None = Field(default=None,
        description="Shared lease registry for clients of this body. Null uses the Windows local application data directory.")

    @field_validator("mcp_url")
    @classmethod
    def _http_url(cls, value: str) -> str:
        parts = urlsplit(value)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            raise ValueError("The minecraft section needs an HTTP or HTTPS mcp_url.")
        return value


__all__ = ["ENV_REFERENCE", "McpConfigError", "McpServerConfig", "MinecraftConfig", "ToolCallConfig"]
