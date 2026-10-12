"""Enforce the import directions of the runtime and workflow packages.

See docs/work_items/refactor/index.md, section "Dependency rules".
The test parses source files only. It imports no project module, so it runs
without model endpoints, Minecraft, or installed dependencies.
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_ROOTS = {"pm", "workflows", "runtime", "pm_core", "main", "modules", "tests", "dev"}
EVERYTHING_IN_PM = {"records", "llm", "memory", "bus", "host", "runs", "app", "console", "paths"}

#: Runtime package -> runtime packages it may import.
ALLOWED_IN_PM: dict[str, set[str]] = {
    "records": {"records", "paths"},
    "llm": {"llm", "records", "paths"},
    "memory": {"memory", "records", "llm", "paths"},
    "bus": {"bus", "paths"},
    "host": {"host", "bus", "memory", "records", "paths"},
    "runs": {"runs", "host", "memory", "llm", "records", "bus", "paths"},
    "app": EVERYTHING_IN_PM,
    "console": EVERYTHING_IN_PM,
}

#: Known violations that later refactor phases remove. Do not add entries.
#: Each entry is (file, imported module).
KNOWN_VIOLATIONS = {
    ("pm/console/worker.py", "workflows.companion"),
    ("pm/runs/supervisor.py", "workflows.public_speech.public_context"),
}


def imports(path: Path):
    """Yield (line, module) for every absolute import in a file, nested ones included."""
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.lineno, node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name


def python_files(folder: str):
    return sorted(p for p in (ROOT / folder).rglob("*.py") if "__pycache__" not in p.parts)


class RuntimeDependencyTests(unittest.TestCase):
    def test_runtime_packages_import_only_allowed_runtime_packages(self):
        problems = []
        for path in python_files("pm"):
            relative = path.relative_to(ROOT).as_posix()
            parts = path.relative_to(ROOT / "pm").parts
            package = parts[0].removesuffix(".py") if len(parts) > 1 else None
            for line, module in imports(path):
                head = module.split(".")
                if head[0] != "pm" or len(head) < 2 or package is None:
                    continue
                if head[1] not in ALLOWED_IN_PM.get(package, EVERYTHING_IN_PM):
                    problems.append(f"{relative}:{line} pm.{package} imports {module}")
        self.assertEqual(problems, [])

    def test_runtime_never_imports_workflows_or_development_code(self):
        problems = []
        for path in python_files("pm"):
            relative = path.relative_to(ROOT).as_posix()
            for line, module in imports(path):
                if module.split(".")[0] in LOCAL_ROOTS - {"pm"} and (relative, module) not in KNOWN_VIOLATIONS:
                    problems.append(f"{relative}:{line} imports {module}")
        self.assertEqual(problems, [])

    def test_known_violations_still_exist(self):
        """Remove an allowlist entry as soon as its import is gone."""
        found = {(path.relative_to(ROOT).as_posix(), module)
                 for path in python_files("pm") for _, module in imports(path)}
        self.assertEqual(KNOWN_VIOLATIONS - found, set())


class WorkflowDependencyTests(unittest.TestCase):
    def test_workflows_never_import_tests_or_development_code(self):
        problems = []
        for path in python_files("workflows"):
            if "tests" in path.relative_to(ROOT / "workflows").parts:
                continue
            for line, module in imports(path):
                if module.split(".")[0] in {"tests", "dev"}:
                    problems.append(f"{path.relative_to(ROOT).as_posix()}:{line} imports {module}")
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
