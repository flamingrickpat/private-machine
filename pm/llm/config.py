"""Resolve model config without network calls or process changes.

Invalid external config raises ValueError with redacted input. Code errors
keep their tracebacks. Resolving a credential reads its environment reference
once, then the git-ignored repository ``.env`` (``pm.llm.env_file``). Resolved connections never select a backend through global environment.
"""
from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path
from urllib.parse import urlsplit

import yaml

from pm.utils import yaml_util
from pydantic import BaseModel, ConfigDict, Field, SecretStr, StrictInt, field_validator, model_validator

from pm.llm.mcp_config import MinecraftConfig, ToolCallConfig


class LlmPreset(StrEnum):
    """The chat connections that every config must define."""

    Default = "Default"
    UnitTestEvaluator = "UnitTestEvaluator"


class Connection(BaseModel):
    """One resolved chat connection, including its credential.

    ``auto`` is not a connection: the runtime selects a preset by name, so no
    global environment variable can silently change the target model.
    ``context_input_fraction`` is the share of the serving context that input
    may use; the rest belongs to chat framing, the response, and estimate error.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    base_url: str
    model: str = Field(min_length=1)
    api_key: SecretStr = SecretStr("")
    startup_script: Path | None = None
    # Remote services can omit serving limits. This optional bound applies
    # to workflow classification, native agents, and direct project decisions.
    context_window: StrictInt | None = Field(default=None, gt=0)
    # Reserve the remainder for chat framing, response, and estimation error.
    context_input_fraction: float = Field(default=0.8, gt=0, lt=1, strict=True)
    # Cap the resolved serving context without increasing the model's limit.
    override_context: StrictInt | None = Field(default=None, gt=0)

    @field_validator("base_url")
    @classmethod
    def check_url(cls, value: str) -> str:
        parts = urlsplit(value)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            raise ValueError("Use an HTTP or HTTPS model URL.")
        if parts.username or parts.password or parts.query or parts.fragment:
            raise ValueError("Keep credentials and query parameters out of the model URL.")
        return value.rstrip("/")

    @field_validator("startup_script", mode="before")
    @classmethod
    def empty_script(cls, value: object) -> object:
        """Read an empty YAML value as "no startup script"."""
        return None if value == "" else value


class ConnectionConfig(Connection):
    """One configured chat connection whose credential may be unresolved.

    ``api_key`` holds a literal secret; ``api_key_env`` names the environment
    variable to read at resolution time. Setting both is an error.
    """

    api_key: SecretStr | None = None
    api_key_env: str | None = None

    @model_validator(mode="after")
    def one_credential_source(self) -> "ConnectionConfig":
        if self.api_key is not None and self.api_key_env is not None:
            raise ValueError("Specify api_key or api_key_env, not both.")
        return self

    def resolve(self) -> Connection:
        """Return the connection with its credential resolved.

        Raises ValueError when the named variable has no value; no anonymous
        request replaces the missing credential.
        """
        key = self.api_key or SecretStr("")
        if self.api_key_env is not None:
            # The process environment wins; the git-ignored .env is the fallback.
            from pm.llm.env_file import credential
            value = credential(self.api_key_env)
            if not value:
                raise ValueError(f"Set the credential environment variable {self.api_key_env} or add it to .env.")
            key = SecretStr(value)
        return Connection(base_url=self.base_url, model=self.model, api_key=key, startup_script=self.startup_script,
                          context_window=self.context_window, context_input_fraction=self.context_input_fraction,
                          override_context=self.override_context)


class EmbeddingConfig(ConnectionConfig):
    """The shared embedding connection and the vector size it returns."""

    dimensions: int = Field(gt=0)
    response_model: str | None = None

    @model_validator(mode="before")
    @classmethod
    def legacy_options(cls, value: object) -> object:
        """Accept old files while removing unused embedding cache settings."""
        if isinstance(value, dict):
            return {key: item for key, item in value.items() if key not in {"cache_dir", "legacy_cache_dir"}}
        return value

    @model_validator(mode="after")
    def remote_lifetime(self) -> "EmbeddingConfig":
        if self.startup_script is not None:
            raise ValueError("The shared embedding service does not use a chat startup script.")
        return self


class ModelConfig(BaseModel):
    """The model configuration file: chat connections, embeddings, and the companion's MCP tools.

    ``tool_call`` is optional. Without it the companion has no MCP tools
    (:class:`pm.llm.mcp_config.ToolCallConfig`).

    ``minecraft`` is optional. Without it the companion has no Minecraft body:
    no world poll, no live body read in the assistant, and no startup probe
    (:class:`pm.llm.mcp_config.MinecraftConfig`). With it the assembly asks the
    endpoint once at startup and stops when nothing answers.

    ``cognitive_control`` holds the operator limits for waits, focus, and
    mental sessions. It stays a plain mapping here because the runtime must
    not import ``workflows``. ``workflows.cognitive_control.schemas.ControlConfig``
    validates it when the assembly builds, so an unknown key fails at startup
    rather than being ignored.
    """

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    chat: dict[LlmPreset, ConnectionConfig]
    embedding: EmbeddingConfig
    tool_call: ToolCallConfig = Field(default_factory=ToolCallConfig)
    minecraft: MinecraftConfig | None = None
    cognitive_control: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def required_presets(self) -> "ModelConfig":
        if set(self.chat) != set(LlmPreset):
            raise ValueError("Configure every LlmPreset, including UnitTestEvaluator.")
        return self

    def evaluator_shares_backend(self) -> bool:
        """Return whether the evaluator preset uses the same endpoint and model."""
        default = self.chat[LlmPreset.Default]
        evaluator = self.chat[LlmPreset.UnitTestEvaluator]
        return (default.base_url, default.model) == (evaluator.base_url, evaluator.model)


def load_config(path: str | Path) -> ModelConfig:
    """Read and validate one model config file.

    A relative ``startup_script`` resolves against the config file's directory,
    so a config stays valid from any working directory. Invalid YAML raises
    ValueError with a redacted message; schema errors keep their field paths.
    """
    path = Path(path).resolve()
    try:
        data = yaml_util.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        raise ValueError("The model config contains invalid YAML. Check the local file.") from None
    config = ModelConfig.model_validate(data)
    for preset, connection in config.chat.items():
        if connection.startup_script is not None:
            script = connection.startup_script.expanduser()
            config.chat[preset] = connection.model_copy(update={"startup_script": (path.parent / script).resolve()})
    return config


def resolve_connection(
    config: ModelConfig | None = None, *, preset: LlmPreset | str | None = None,
    base_url: str | None = None, model: str | None = None,
    api_key: str | None = None, startup_script: str | Path | None = None,
) -> Connection:
    """Select a preset, or pass a complete explicit connection without config.

    Omitted selection means Default. Explicit relative scripts use the caller's
    working directory. Config-relative scripts use the config file directory.
    """
    explicit = any(value is not None for value in (base_url, model, api_key, startup_script))
    if explicit:
        if preset is not None:
            raise ValueError("Choose a preset or an explicit connection, not both.")
        if base_url is None or model is None or api_key is None:
            raise ValueError("An explicit connection requires base_url, model, and api_key.")
        script = Path(startup_script).expanduser().resolve() if startup_script else None
        return Connection(base_url=base_url, model=model, api_key=SecretStr(api_key), startup_script=script)
    if config is None:
        raise ValueError("Supply model config for a preset connection.")
    return config.chat[LlmPreset(preset) if preset is not None else LlmPreset.Default].resolve()
