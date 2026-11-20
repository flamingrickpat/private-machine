"""Pack synthetic run evidence for offline inspection. Secrets and source snapshots stay local."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

from pydantic import BaseModel


class BundleEntry(BaseModel):
    path: str
    sha256: str
    size: int


class BundleManifest(BaseModel):
    version: int = 1
    entries: list[BundleEntry]


SECRET = re.compile(rb"(?:sk-or-v1-[a-zA-Z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)")


def include_evidence(relative: Path) -> bool:
    """Select scenario state, synthetic memory, plans, receipts and workflow results."""
    parts = relative.parts
    if relative.as_posix() == "minecraft-setup/server.properties.example":
        return True
    if any(part.startswith(".") for part in parts):
        return False
    if "source" in parts or relative.name == "model-config.yaml" or relative.name.startswith("ghost.before-"):
        return False
    if relative.name.startswith(".env") or "credential" in relative.name.lower():
        return False
    if relative.suffix in {".json", ".jsonl", ".md", ".pickle", ".log", ".yaml", ".yml"}:
        return True
    return relative.name == "character.yaml"


def inspected_digest(path: Path) -> str:
    """Inspect bounded chunks and include overlap for a credential split across reads."""
    digest = sha256()
    previous = b""
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            if SECRET.search(previous + chunk):
                raise ValueError(f"Credential detected in {path}. Review this artifact before export.")
            digest.update(chunk)
            previous = chunk[-256:]
    return digest.hexdigest()


def write_bundle(roots: list[Path], destination: Path) -> None:
    """Keep one compressed blob per content hash. Publish only after every file passes inspection."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    entries: list[BundleEntry] = []
    seen: set[str] = set()
    with TemporaryDirectory(dir=destination.parent) as temporary:
        packed = Path(temporary) / "bundle.zip"
        with ZipFile(packed, "x", compression=ZIP_DEFLATED, compresslevel=3) as archive:
            for root in roots:
                for path in sorted(root.rglob("*")):
                    if not path.is_file() or not include_evidence(path.relative_to(root)):
                        continue
                    if not path.resolve().is_relative_to(root.resolve()):
                        raise ValueError(f"Evidence path escapes run: {path}")
                    digest = inspected_digest(path)
                    name = path.relative_to(root.parent).as_posix()
                    entries.append(BundleEntry(path=name, sha256=digest, size=path.stat().st_size))
                    if digest not in seen:
                        archive.write(path, "blobs/" + digest)
                        seen.add(digest)
            manifest = BundleManifest(entries=entries)
            archive.writestr("manifest.json", manifest.model_dump_json())  # pm-lint: allow PM007 -- bundle manifest serialization
        # Windows rename fails if another export won the same destination.
        packed.rename(destination)


def unpack_bundle(source: Path, destination: Path) -> None:
    """Recreate the debugger's run layout after checking every content digest."""
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as temporary:
        staging = Path(temporary) / "runs"
        staging.mkdir()
        with ZipFile(source) as archive:
            manifest = BundleManifest.model_validate_json(archive.read("manifest.json"))
            if manifest.version != 1:
                raise ValueError(f"Unsupported bundle version: {manifest.version}")
            for entry in manifest.entries:
                relative = Path(entry.path)
                if relative.is_absolute() or relative.drive or ".." in relative.parts:
                    raise ValueError(f"Invalid bundle member: {entry.path}")
                target = (staging / relative).resolve()
                if not target.is_relative_to(staging.resolve()):
                    raise ValueError(f"Bundle member escapes destination: {entry.path}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open("blobs/" + entry.sha256) as blob, target.open("xb") as stream:
                    shutil.copyfileobj(blob, stream, length=1024 * 1024)
                if target.stat().st_size != entry.size or inspected_digest(target) != entry.sha256:
                    raise ValueError(f"Bundle digest mismatch: {entry.path}")
        staging.rename(destination)
