"""Selected implementation inspection for the A18 self-model.

The runtime gives the companion a narrow read grant over source, documentation,
configuration, and runtime evidence. A grant is explicit: it names its roots,
its include patterns, its exclude patterns, and its size limits. The inspector
checks the grant on every call. It never turns a successful read into a
standing capability, and it never reads a credential file.

The module defines no model call and writes no memory. Callers own the commit
boundary and the interpretation.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
import subprocess
from threading import Event
from typing import Callable, Iterable

__all__ = [
    "CREDENTIAL_PATTERNS",
    "SourceDenied",
    "SourceGrant",
    "SourceInspector",
    "SourceUnavailable",
    "git_revision",
]


class SourceDenied(PermissionError):
    """The request is outside the explicit source grant."""


class SourceUnavailable(FileNotFoundError):
    """The granted file is missing or unreadable at observation time."""


#: Patterns that never become readable, even inside a granted root. The
#: grant owner may add patterns; these always apply. Matching is casefold.
CREDENTIAL_PATTERNS: tuple[str, ...] = (
    "**/.env",
    "**/.env.*",
    "**/*.pem",
    "**/*.key",
    "**/*.p12",
    "**/*.pfx",
    "**/id_rsa*",
    "**/id_ed25519*",
    "**/*credential*",
    "**/*secret*",
    "**/*password*",
    "**/*token*",
    "**/.git/**",
    "**/__pycache__/**",
    "**/*.pyc",
    "**/.venv/**",
    "**/.runtime/**",
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _glob_regex(pattern: str) -> "re.Pattern[str]":
    """Compile one POSIX glob with ``**`` crossing directory boundaries.

    ``pathlib.PurePosixPath.match`` treats ``**`` as a single segment, so the
    grant builds its own pattern. ``*`` and ``?`` never cross a ``/``.
    """
    parts = pattern.split("/")
    chunks = []
    for index, part in enumerate(parts):
        last = index == len(parts) - 1
        if part == "**":
            chunks.append(".*" if last else "(?:[^/]+/)*")
            continue
        escaped = re.escape(part).replace(r"\*", "[^/]*").replace(r"\?", "[^/]")
        chunks.append(escaped + ("" if last else "/"))
    return re.compile("^" + "".join(chunks) + "$", re.IGNORECASE)


_GLOB_CACHE: dict[str, "re.Pattern[str]"] = {}


def _match(pattern: str, relative: str) -> bool:
    """Match one pattern against a POSIX path without leaving the grant."""
    compiled = _GLOB_CACHE.get(pattern)
    if compiled is None:
        compiled = _glob_regex(pattern)
        _GLOB_CACHE[pattern] = compiled
    return compiled.match(relative) is not None


def git_revision(root: Path, *, timeout: float = 10.0) -> str:
    """Return the working-tree commit for ``root``, or an empty string.

    The revision is evidence for the observed code, not a guarantee that the
    running process imported it. Callers record the installed distribution
    separately when that distinction matters.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(root), capture_output=True,
            text=True, timeout=timeout, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


@dataclass(frozen=True)
class SourceGrant:
    """One explicit, bounded read grant over a set of repository roots.

    ``include`` lists POSIX glob patterns relative to a root. An empty include
    permits every non-excluded file under the roots. ``exclude`` always
    contains the credential patterns, and exclusion wins over inclusion.
    """

    grant_id: str
    roots: tuple[Path, ...]
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = CREDENTIAL_PATTERNS
    max_bytes: int = 200_000
    max_lines: int = 4_000
    label: str = ""
    categories: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.grant_id.strip():
            raise ValueError("A source grant needs an id.")
        if not self.roots:
            raise ValueError("A source grant needs at least one root.")
        if self.max_bytes < 1 or self.max_lines < 1:
            raise ValueError("Source grant limits must be positive.")
        resolved = []
        for root in self.roots:
            path = Path(root).resolve()
            if not path.is_dir():
                raise SourceUnavailable(f"Source grant root is not a directory: {path}")
            resolved.append(path)
        object.__setattr__(self, "roots", tuple(resolved))
        # The credential patterns are not optional.
        object.__setattr__(self, "exclude", tuple(dict.fromkeys((*CREDENTIAL_PATTERNS, *self.exclude))))

    def relative(self, path: Path) -> str | None:
        """Return the POSIX path below a root, or None when outside every root."""
        resolved = Path(path).resolve()
        for root in self.roots:
            try:
                return resolved.relative_to(root).as_posix()
            except ValueError:
                continue
        return None

    def allows(self, path: Path) -> bool:
        """Apply the include and exclude policy to one resolved path."""
        relative = self.relative(path)
        if relative is None:
            return False
        if any(_match(pattern, relative) for pattern in self.exclude):
            return False
        if self.include and not any(_match(pattern, relative) for pattern in self.include):
            return False
        return True

    def resolve(self, reference: str | Path) -> Path:
        """Resolve a reference against the roots, or raise ``SourceDenied``.

        The check uses the resolved path, so ``..`` and symlinks cannot escape
        a root. A non-absolute reference is tried below each root in order.
        """
        candidate = Path(reference)
        options = [candidate] if candidate.is_absolute() else [root / candidate for root in self.roots]
        for option in options:
            resolved = option.resolve()
            if self.allows(resolved):
                return resolved
            if self.relative(resolved) is not None:
                raise SourceDenied(f"The source grant excludes {reference}.")
        raise SourceDenied(f"The source grant does not cover {reference}.")

    def to_dict(self) -> dict:
        """Return a serializable description for a trace or a grant registry."""
        return {
            "grant_id": self.grant_id,
            "roots": [str(root) for root in self.roots],
            "include": list(self.include),
            "exclude": list(self.exclude),
            "max_bytes": self.max_bytes,
            "max_lines": self.max_lines,
            "label": self.label,
            "categories": list(self.categories),
        }

    def narrowed(self, *, include: Iterable[str] | None = None,
                 label: str | None = None) -> "SourceGrant":
        """Return a copy with a smaller include set, for one inspection task."""
        return replace(
            self,
            include=tuple(include) if include is not None else self.include,
            label=label if label is not None else self.label,
        )


@dataclass
class SourceInspector:
    """Read granted files and report digest, revision, and observation time.

    Every call re-reads the file, so a caller that observes a change can mark
    a dependent belief stale. ``cancel`` stops an in-flight read before it
    returns content. ``trace`` receives one compact record per operation.
    """

    grant: SourceGrant
    repo_revision: str = ""
    configuration_revision: str = ""
    cancel: Event | None = None
    trace: Callable[[dict], None] | None = None
    clock: Callable[[], datetime] = _utcnow
    _reads: dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def check_cancelled(self) -> None:
        """Raise ``SourceDenied`` when the owner revoked this inspection."""
        if self.cancel is not None and self.cancel.is_set():
            raise SourceDenied("The source inspection was cancelled.")

    def _observe(self, resolved: Path) -> dict:
        relative = self.grant.relative(resolved)
        assert relative is not None  # ``resolve`` already enforced the grant.
        try:
            raw = resolved.read_bytes()
        except OSError as error:
            raise SourceUnavailable(f"Cannot read the granted file {relative}: {error}") from error
        self.check_cancelled()
        digest = hashlib.sha256(raw).hexdigest()
        # Keep the first digest of the session, so a later change is visible
        # even after the file is read again.
        previous = self._reads.setdefault(relative, digest)
        return {
            "path": relative,
            "absolute_path": str(resolved),
            "bytes": len(raw),
            "lines": raw.count(b"\n") + (1 if raw else 0),
            "sha256": digest,
            "changed": previous != digest,
            "repo_revision": self.repo_revision,
            "configuration_revision": self.configuration_revision,
            "observed_at": self.clock().isoformat(),
            "grant_id": self.grant.grant_id,
        }

    def digest(self, reference: str | Path) -> dict:
        """Return the observation metadata for one granted file."""
        resolved = self.grant.resolve(reference)
        return self._observe(resolved)

    def read(self, reference: str | Path, *, offset: int = 0, length: int | None = None) -> dict:
        """Return a bounded text page of one granted file.

        The result carries the full-file digest, not the page digest, so a
        caller can detect a change outside the returned range. The result
        refuses a file above ``max_bytes`` and truncates to ``max_lines``.
        """
        if offset < 0:
            raise ValueError("The source read offset must not be negative.")
        resolved = self.grant.resolve(reference)
        metadata = self._observe(resolved)
        if metadata["bytes"] > self.grant.max_bytes:
            raise SourceDenied(
                f"{metadata['path']} is {metadata['bytes']} bytes; the grant limit is "
                f"{self.grant.max_bytes} bytes.")
        text = resolved.read_text(encoding="utf-8", errors="replace")
        self.check_cancelled()
        lines = text.splitlines()
        if len(lines) > self.grant.max_lines:
            lines = lines[: self.grant.max_lines]
            metadata["truncated_lines"] = True
        window = lines[offset:]
        if length is not None:
            if length < 0:
                raise ValueError("The source read length must not be negative.")
            window = window[:length]
        numbered = [{"line": offset + index + 1, "text": value} for index, value in enumerate(window)]
        result = {
            **metadata,
            "offset": offset,
            "lines_returned": len(numbered),
            "next_offset": offset + len(numbered) if offset + len(numbered) < len(lines) else None,
            "text": "\n".join(value for _, value in ((item["line"], item["text"]) for item in numbered)),
            "numbered_lines": numbered,
        }
        if self.trace is not None:
            self.trace({"operation": "read", **{key: result[key] for key in
                        ("path", "sha256", "repo_revision", "observed_at", "grant_id")}})
        return result

    def list_files(self, *, limit: int = 200) -> list[dict]:
        """List granted files with digests, bounded by ``limit``.

        Directories are skipped. The listing stays inside the roots and the
        grant policy, so a caller cannot discover a credential file's path.
        """
        if limit < 1:
            raise ValueError("The file listing limit must be positive.")
        found: list[dict] = []
        for root in self.grant.roots:
            for path in sorted(root.rglob("*")):
                self.check_cancelled()
                if not path.is_file() or not self.grant.allows(path):
                    continue
                found.append({"path": self.grant.relative(path), "bytes": path.stat().st_size})
                if len(found) >= limit:
                    return found
        return found

    def changed_files(self) -> list[str]:
        """Return the granted paths whose digest changed during this session."""
        return sorted(path for path, digest in self._reads.items()
                      if digest and self._recompute(path) not in (None, digest))

    def _recompute(self, relative: str) -> str | None:
        try:
            resolved = self.grant.resolve(relative)
            return hashlib.sha256(resolved.read_bytes()).hexdigest()
        except (SourceDenied, SourceUnavailable, OSError):
            return None

    def revision(self, references: Iterable[str | Path] | None = None) -> str:
        """Return one revision over the named files, or over the whole grant.

        The caller records this value beside a capability belief. A later
        mismatch marks the dependent belief stale.
        """
        if references is None:
            items = [(item["path"], item["bytes"]) for item in self.list_files()]
            payload = repr(sorted(items))
        else:
            digests = []
            for reference in references:
                metadata = self.digest(reference)
                digests.append((metadata["path"], metadata["sha256"]))
            payload = repr(sorted(digests))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_default_grant(repo_root: Path, *, allowed: Iterable[str] | None = None,
                        label: str = "default") -> SourceGrant:
    """Build a read grant over the repository for source-aware inspection.

    The default include set covers Python source, documentation, workflow
    manifests, and example configuration. Credentials and runtime directories
    stay excluded by ``CREDENTIAL_PATTERNS``.
    """
    include = tuple(allowed) if allowed is not None else (
        "pm/**/*.py",
        "workflows/**/*.py",
        "workflows/**/*.workflow.md",
        "workflows/**/*.yaml",
        "docs/**/*.md",
        "config.*.yaml",
        "AGENTS.md",
        "pyproject.toml",
    )
    return SourceGrant(
        grant_id="default-source-inspection",
        roots=(Path(repo_root).resolve(),),
        include=include,
        label=label,
        categories=("source", "documentation", "configuration", "runtime"),
    )
