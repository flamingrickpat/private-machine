"""Capture source identities and local evidence without changing runtime state."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys
import tomllib
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def digest(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.DEVNULL)


def capture(repo: Path, out: Path) -> dict:
    """Store exact nonignored source bytes, including staged and untracked edits."""
    names = sorted(set(git(repo, "ls-files", "-z", "--cached", "--others", "--exclude-standard").decode().split("\0")) - {""})
    files = []
    archive = out / f"{repo.name}-source.zip"
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as source:
        for name in names:
            path = repo / name
            if path.is_file():
                data = path.read_bytes()
                files.append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
                source.writestr(name, data)
    patch = out / f"{repo.name}-dirty.patch"
    patch.write_bytes(git(repo, "diff", "--binary", "HEAD"))
    return {
        "root": str(repo), "revision": git(repo, "rev-parse", "HEAD").decode().strip(),
        "branch": git(repo, "branch", "--show-current").decode().strip(),
        "status": git(repo, "status", "--porcelain=v1").decode(),
        "source_archive": digest(archive), "dirty_patch": digest(patch), "files": files,
        "runtime_health": "not established by source capture",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="A new directory under .runtime.")
    parser.add_argument("--config", type=Path, default=ROOT / "config.alpha.example.yaml")
    parser.add_argument("--brief", type=Path, default=Path("C:/Temp/pm.md"))
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(ROOT / ".runtime"):
        raise ValueError("Keep source archives and local configuration identities under .runtime.")
    out.mkdir(parents=True, exist_ok=False)
    names = ("private-machine", "private-machine-head", "pm_next", "pm_next_v2", "pm-workflows", "pm-coder", "coding-workflow", "pm-minecraft", "pm-agentic-babymode")
    receipt = {"captured_at": datetime.now(timezone.utc).isoformat(), "python": sys.version,
               "repositories": {name: capture(ROOT.parent / name, out) for name in names}}
    receipt["real_server"] = capture(Path("C:/source/minecraft-server-real"), out)
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    receipt["locks"] = [digest(ROOT / "uv.lock"), digest(ROOT / "pyproject.toml")]
    receipt["locked_pm_packages"] = [package for package in lock["package"] if package["name"] in {"pm-coder", "pm-workflows"}]
    receipt["installed_pm_packages"] = {name: {"version": metadata.version(name),
        "direct_url": json.loads(metadata.distribution(name).read_text("direct_url.json"))}
        for name in ("pm-coder", "pm-workflows")}
    from pm.llm.config import load_config
    config = load_config(args.config)
    receipt["configuration"] = {**digest(args.config), "chat": {
        key: {"base_url": value.base_url, "model": value.model, "startup_script": str(value.startup_script),
              "credential_source": "environment" if value.api_key_env else "literal or empty (redacted)"}
        for key, value in config.chat.items()}, "embedding": {
        key: str(getattr(config.embedding, key)) for key in
        ("base_url", "model", "response_model", "dimensions", "cache_dir", "legacy_cache_dir")}}
    receipt["personas"] = [digest(ROOT / "test_data/personas/spamuel/canon.yaml"), digest(ROOT / "test_data/personas/spamuel/config.yaml")]
    receipt["original_brief"] = digest(args.brief)
    (out / "original-brief.md").write_bytes(args.brief.read_bytes())
    receipt["historical_evidence"] = [digest(path) for pattern in
        ("w7/conversations/*/summary.json", "w8/live-minecraft/*/minecraft.json")
        for path in sorted((ROOT / ".runtime").glob(pattern))]
    receipt["services"] = {}
    embedding_catalogue = "embeddings/models" if "openrouter.ai" in config.embedding.base_url else "models"
    services = (("chat", f"{config.chat['Default'].base_url}/models"),
                ("embedding", f"{config.embedding.base_url}/{embedding_catalogue}"),
                ("decision", "http://127.0.0.1:8082/v1/models"))
    for role, url in services:
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                result = {"status": "available", "models": json.load(response)}
        except (OSError, ValueError) as error:
            result = {"status": "unavailable", "error": str(error)}
        receipt["services"][role] = {"url": url, **result}
    (out / "snapshot.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(out / "snapshot.json")


if __name__ == "__main__":
    main()
