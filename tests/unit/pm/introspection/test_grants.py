"""Source grant and inspector contracts.

No store and no model. The grant is the security boundary: a credential, a
traversal attempt, or an oversized file must fail before content returns.
"""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest

from pm.introspection.grants import (
    CREDENTIAL_PATTERNS,
    SourceDenied,
    SourceGrant,
    SourceInspector,
    build_default_grant,
    git_revision,
)


def _repo(tmp: str) -> Path:
    root = Path(tmp) / "repo"
    (root / "pm").mkdir(parents=True)
    (root / "docs").mkdir(exist_ok=True)
    (root / "pm" / "module.py").write_text("print('hello')\n", encoding="utf-8")
    (root / "docs" / "guide.md").write_text("# Guide\n", encoding="utf-8")
    (root / ".env").write_text("OPENROUTER_API_KEY=secret\n", encoding="utf-8")
    (root / "pm" / "credentials.json").write_text('{"token": "x"}\n', encoding="utf-8")
    (root / "outside.txt").write_text("not granted\n", encoding="utf-8")
    return root


class SourceGrantTests(unittest.TestCase):
    def test_read_returns_digest_revision_and_observation_time(self):
        with TemporaryDirectory() as tmp:
            root = _repo(tmp)
            grant = SourceGrant("test", roots=(root,), include=("pm/**/*.py",))
            inspector = SourceInspector(grant, repo_revision="abc123", configuration_revision="cfg1")
            result = inspector.read("pm/module.py")
            self.assertEqual(result["path"], "pm/module.py")
            self.assertEqual(result["repo_revision"], "abc123")
            self.assertEqual(result["configuration_revision"], "cfg1")
            self.assertIn("print('hello')", result["text"])
            self.assertEqual(len(result["sha256"]), 64)

    def test_credential_files_are_always_denied(self):
        with TemporaryDirectory() as tmp:
            root = _repo(tmp)
            grant = SourceGrant("test", roots=(root,))
            inspector = SourceInspector(grant)
            with self.assertRaises(SourceDenied):
                inspector.read(".env")
            with self.assertRaises(SourceDenied):
                inspector.read("pm/credentials.json")
            # The deny list covers the defaults even without an include set.
            self.assertIn("**/.env", CREDENTIAL_PATTERNS)

    def test_path_traversal_cannot_escape_a_root(self):
        with TemporaryDirectory() as tmp:
            root = _repo(tmp)
            grant = SourceGrant("test", roots=(root,))
            inspector = SourceInspector(grant)
            with self.assertRaises(SourceDenied):
                inspector.read("../outside.txt")
            with self.assertRaises(SourceDenied):
                inspector.read(root.parent / "outside.txt")

    def test_include_pattern_bounds_the_grant(self):
        with TemporaryDirectory() as tmp:
            root = _repo(tmp)
            grant = SourceGrant("test", roots=(root,), include=("docs/**/*.md",))
            inspector = SourceInspector(grant)
            self.assertEqual(inspector.read("docs/guide.md")["path"], "docs/guide.md")
            with self.assertRaises(SourceDenied):
                inspector.read("pm/module.py")

    def test_oversized_file_is_refused(self):
        with TemporaryDirectory() as tmp:
            root = _repo(tmp)
            (root / "pm" / "big.py").write_text("x" * 5000, encoding="utf-8")
            grant = SourceGrant("test", roots=(root,), include=("pm/**/*.py",), max_bytes=1000)
            inspector = SourceInspector(grant)
            with self.assertRaises(SourceDenied):
                inspector.read("pm/big.py")

    def test_changed_digest_is_reported(self):
        with TemporaryDirectory() as tmp:
            root = _repo(tmp)
            grant = SourceGrant("test", roots=(root,))
            inspector = SourceInspector(grant)
            first = inspector.read("pm/module.py")
            (root / "pm" / "module.py").write_text("print('changed')\n", encoding="utf-8")
            second = inspector.read("pm/module.py")
            self.assertFalse(first["changed"])
            self.assertTrue(second["changed"])
            self.assertNotEqual(first["sha256"], second["sha256"])
            self.assertEqual(inspector.changed_files(), ["pm/module.py"])

    def test_cancellation_stops_a_read(self):
        with TemporaryDirectory() as tmp:
            root = _repo(tmp)
            grant = SourceGrant("test", roots=(root,))
            cancel = Event()
            cancel.set()
            inspector = SourceInspector(grant, cancel=cancel)
            with self.assertRaises(SourceDenied):
                inspector.read("pm/module.py")

    def test_default_grant_covers_source_docs_and_configuration(self):
        grant = build_default_grant(Path.cwd())
        self.assertIn("pm/**/*.py", grant.include)
        self.assertIn("docs/**/*.md", grant.include)
        self.assertIn("config.*.yaml", grant.include)


class GitRevisionTests(unittest.TestCase):
    def test_git_revision_returns_empty_for_a_non_repository(self):
        with TemporaryDirectory() as tmp:
            self.assertEqual(git_revision(Path(tmp)), "")


if __name__ == "__main__":
    unittest.main()
