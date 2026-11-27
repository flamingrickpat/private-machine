"""Freeze the actual participant source and installed dependency origins per attempt.

The participant processes run from this copy, so a developer edit during a live
attempt cannot change workflow scripts under the running cognition. The receipt
records the repository revision, the tracked and unignored file hashes, the model
configuration hash, and the installed distribution origins.

Ordering: every file is copied before it is hashed, and the receipt is written
last, so a ``source.json`` that exists describes files that are already present.
Credentials stay in local data and never enter the copy. The package overview is
in ``simulation/__init__.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.metadata import distribution
from pathlib import Path
import shutil

from pydantic import BaseModel, JsonValue, TypeAdapter

from pm.host.durable_files import atomic_json
from pm.paths import REPO_ROOT
from pm.llm.config import LlmPreset, load_config
from pm.llm.mcp_config import MinecraftConfig
from pm.utils import yaml_util
from pm.utils.operating_system import OperatingSystem


@dataclass(frozen=True, slots=True)
class SimulationConfigOverrides:
    """Launch-specific settings selected by the simulator or live harness."""

    profile: bool = False
    model_url: str | None = None
    body_url: str | None = None
    prompt_directory: Path | None = None
    overnight: bool = False


def copy_simulation_config(source: Path, destination: Path,
                           overrides: SimulationConfigOverrides = SimulationConfigOverrides()) -> None:
    """Write the runtime's private config for a simulation launch.

    Normalize paths before moving the document. Expand credentials only at
    the private YAML boundary.
    Every simulation asks the live model instead of replaying a cache entry.
    """
    settings = load_config(source)
    io = settings.inference_io
    prompt_directory = io.prompt_log_directory
    if overrides.prompt_directory is not None:
        prompt_directory = overrides.prompt_directory.resolve()
    settings.inference_io = io.model_copy(update={
        "completion_cache": False, "prompt_log_directory": prompt_directory,
    })
    if overrides.profile:
        settings.profile_cognition = True
    if overrides.overnight:
        settings.ghost_backups = False
        settings.inference_io = settings.inference_io.model_copy(update={"prompt_log": True})  # pm-lint: allow PM005 -- Pydantic settings update
        settings.minecraft = None
        settings.tool_call = settings.tool_call.model_copy(update={"mcp_servers": []})  # pm-lint: allow PM005 -- Pydantic settings update
        for preset, connection in settings.chat.items():
            if connection.base_url != "https://openrouter.ai/api/v1" or connection.startup_script is not None:
                raise ValueError("Scenario 5 requires remote OpenRouter chat routes.")
            settings.chat[preset] = connection.model_copy(update={  # pm-lint: allow PM005 -- Pydantic settings update
                "model": "deepseek/deepseek-v4.1-flash-1017", "override_context": 65536,
                "context_window": None, "provider": None, "max_concurrent_requests": 0, "require_reasoning": False,
            })
        if settings.embedding.base_url != "https://openrouter.ai/api/v1":
            raise ValueError("Scenario 5 requires the configured OpenRouter embedding route.")
        settings.embedding = settings.embedding.model_copy(update={"max_concurrent_requests": 0})  # pm-lint: allow PM005 -- Pydantic settings update
    if overrides.model_url is not None:
        default = settings.chat[LlmPreset.Default]
        settings.chat[LlmPreset.Default] = default.model_copy(update={"base_url": overrides.model_url.rstrip("/")})
    if overrides.body_url is not None:
        body = settings.minecraft
        if body is None:
            settings.minecraft = MinecraftConfig(mcp_url=overrides.body_url)
        else:
            settings.minecraft = body.model_copy(update={"mcp_url": overrides.body_url})
        servers = [
            server.model_copy(update={"url": overrides.body_url}) if server.name == "minecraft" else server
            for server in settings.tool_call.mcp_servers
        ]
        settings.tool_call = settings.tool_call.model_copy(update={"mcp_servers": servers})

    # Only this private YAML writer expands the secret values.
    wire = TypeAdapter(dict[str, JsonValue])
    document = wire.validate_python(settings.model_dump(mode="json"))  # pm-lint: allow PM007 -- private YAML serialization
    chat: dict[str, JsonValue] = {}
    for preset, connection in settings.chat.items():
        entry = wire.validate_python(connection.model_dump(mode="json"))  # pm-lint: allow PM007 -- private YAML serialization
        entry["api_key"] = connection.resolve().api_key.get_secret_value()
        chat[preset.value] = entry
    document["chat"] = chat
    embedding = wire.validate_python(settings.embedding.model_dump(mode="json"))  # pm-lint: allow PM007 -- private YAML serialization
    embedding["api_key"] = settings.embedding.resolve().api_key.get_secret_value()
    document["embedding"] = embedding
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml_util.safe_dump(document), encoding="utf-8")


class FrozenDependency(BaseModel):
    version: str
    direct_url: str | None


class FrozenSource(BaseModel):
    """Provenance and paths for one frozen simulation attempt."""

    revision: str
    status: str
    files: dict[str, str]  # pm-lint: allow PM004 -- content digests keyed by source path
    dependencies: dict[str, FrozenDependency]  # pm-lint: allow PM004 -- origins keyed by distribution name
    source: Path
    config: Path
    config_sha256: str


def freeze_source(destination: Path, config: Path) -> FrozenSource:
    """Copy tracked and unignored source files without changing the repository.

    Child processes run from this copy. Later developer edits cannot change
    workflow scripts during an active attempt. Credentials stay in local data.
    Returns the receipt that was written beside the copy. Raises
    ``FileExistsError`` when the destination already exists.
    """
    paths = OperatingSystem.output_bytes(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=REPO_ROOT
    ).split(b"\0")
    destination.mkdir(parents=True, exist_ok=False)
    hashes: dict[str, str] = {}
    config_source = config.resolve()
    for raw in paths:
        if not raw:
            continue
        relative = Path(raw.decode("utf-8"))
        if relative.is_relative_to(Path("test_data")) and relative.name.endswith((".zip", ".tar.zst")):
            continue  # Baseline review archives are never participant source.
        source = REPO_ROOT / relative
        if not source.is_file() or source.resolve() == config_source:
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        # Hash the copy, not the original: the receipt must describe the bytes the
        # participant will actually import.
        hashes[relative.as_posix()] = hashlib.sha256(target.read_bytes()).hexdigest()
    saved_config = destination.parent / "model-config.yaml"
    copy_simulation_config(config, saved_config)
    dependencies: dict[str, FrozenDependency] = {}
    for name in ("pm-coder", "pm-workflows", "websockets"):
        installed = distribution(name)
        dependencies[name] = FrozenDependency(version=installed.version, direct_url=installed.read_text("direct_url.json"))
    revision = OperatingSystem.output_bytes(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
    status = OperatingSystem.output_bytes(["git", "status", "--porcelain"], cwd=REPO_ROOT).decode()
    receipt = FrozenSource(revision=revision, status=status, files=hashes, dependencies=dependencies,
                           source=destination.resolve(), config=saved_config.resolve(),
                           config_sha256=hashlib.sha256(saved_config.read_bytes()).hexdigest())
    atomic_json(destination.parent / "source.json",
                receipt.model_dump(mode="json"))  # pm-lint: allow PM007 -- provenance receipt serialization
    return receipt
