"""Enforce the import directions of the runtime and workflow packages.

See docs/work_items/refactor/index.md, section "Dependency rules".
The test parses source files only. It imports no project module, so it runs
without model endpoints, Minecraft, or installed dependencies.
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCAL_ROOTS = {"pm", "workflows", "runtime", "pm_core", "main", "modules", "tests", "dev"}
EVERYTHING_IN_PM = {"records", "llm", "context_render", "memory", "bus", "host", "runs", "app", "console", "paths", "introspection"}
#: Leaf modules that every runtime package may import. They import no other
#: runtime package: ``pm.constants`` holds tunable numbers (A19F engineering
#: rules) and ``pm.utils`` holds the YAML helpers.
LEAF_MODULES = {"constants", "utils"}

#: Runtime package -> runtime packages it may import.
ALLOWED_IN_PM: dict[str, set[str]] = {
    "records": {"records", "paths"},
    "llm": {"llm", "context_render", "records", "paths"},
    "context_render": {"context_render", "records", "paths"},
    "memory": {"memory", "records", "llm", "paths"},
    "introspection": {"introspection", "records", "memory", "paths"},
    "bus": {"bus", "paths"},
    "host": {"host", "bus", "memory", "records", "paths"},
    "runs": {"runs", "host", "memory", "llm", "records", "bus", "paths"},
    "app": EVERYTHING_IN_PM,
    "console": EVERYTHING_IN_PM,
}

#: Known violations that later refactor phases remove. Do not add entries.
#: Each entry is (file, imported module).
KNOWN_VIOLATIONS: set[tuple[str, str]] = set()


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
            name = parts[0].removesuffix(".py")
            package = name if len(parts) > 1 or name in LEAF_MODULES else None
            for line, module in imports(path):
                head = module.split(".")
                if head[0] != "pm" or len(head) < 2 or package is None:
                    continue
                if package in LEAF_MODULES:
                    problems.append(f"{relative}:{line} leaf module pm.{package} imports {module}")
                elif head[1] not in ALLOWED_IN_PM.get(package, EVERYTHING_IN_PM) | LEAF_MODULES:
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


#: Workflow libraries have no manifest; several workflows share them.
LIBRARIES = {"public_speech", "minecraft_body"}
#: Libraries may build on accepted cognition state.
LIBRARY_MAY_IMPORT = LIBRARIES | {"main_cognition"}
#: Known library -> task-workflow imports. Do not add entries.
KNOWN_LIBRARY_COUPLINGS = {
    ("workflows/public_speech/reply_recovery.py", "workflows.minecraft_project.acceptance"),
    ("workflows/public_speech/reply_recovery.py", "workflows.minecraft.specialist_session"),
}


class WorkflowDependencyTests(unittest.TestCase):
    def test_libraries_import_no_task_workflow(self):
        problems, found = [], set()
        for library in sorted(LIBRARIES):
            for path in python_files(f"workflows/{library}"):
                relative = path.relative_to(ROOT).as_posix()
                for line, module in imports(path):
                    head = module.split(".")
                    if head[0] != "workflows" or len(head) < 2 or head[1] in LIBRARY_MAY_IMPORT:
                        continue
                    found.add((relative, module))
                    if (relative, module) not in KNOWN_LIBRARY_COUPLINGS:
                        problems.append(f"{relative}:{line} imports {module}")
        self.assertEqual(problems, [])
        self.assertEqual(KNOWN_LIBRARY_COUPLINGS - found, set())

    def test_only_entry_points_import_the_assembly(self):
        problems = []
        for path in python_files("workflows") + python_files("pm"):
            for line, module in imports(path):
                if module == "workflows.companion":
                    problems.append(f"{path.relative_to(ROOT).as_posix()}:{line}")
        self.assertEqual(problems, [])

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
