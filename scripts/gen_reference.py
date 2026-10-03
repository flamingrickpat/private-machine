"""Generate module references without importing application entry points."""
from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()
roots = [Path(name) for name in ("bus", "console", "pm_core", "runtime", "workflows")]
paths = [p for root in roots for p in root.rglob("*.py")]
paths += [Path(name) for name in ("main.py", "modules.py", "workflow_adapter.py", "working_target.py", "ghost_mcp.py")]
paths.append(Path("scripts/alpha_input_gate.py"))
for path in sorted(paths):
    if "tests" in path.parts or any(part.startswith("_") for part in path.parts):
        continue
    parts = path.with_suffix("").parts
    target = Path("reference", *parts).with_suffix(".md")
    nav[parts] = str(Path(*parts).with_suffix(".md"))
    with mkdocs_gen_files.open(target, "w") as stream:
        stream.write(f"# {'.'.join(parts)}\n\n::: {'.'.join(parts)}\n")
    mkdocs_gen_files.set_edit_path(target, path)
# Render the installed dependency so the manual matches the consuming lock.
with mkdocs_gen_files.open("reference/pm_decision.md", "w") as stream:
    stream.write("# pm_decision\n\n::: pm_decision\n    options:\n        inherited_members: [__init__]\n        show_if_no_docstring: true\n")
nav[("pm_decision",)] = "pm_decision.md"
# Read the installed workflow package so API links match the consuming lock.
for module in ("inputs", "protocol", "working_target", "python_role", "kernel", "child", "manifest", "journal"):
    identifier = f"pm_workflows.{module}"
    target = f"reference/pm_workflows/{module}.md"
    with mkdocs_gen_files.open(target, "w") as stream:
        stream.write(f"# {identifier}\n\n::: {identifier}\n")
    nav[("pm_workflows", module)] = f"pm_workflows/{module}.md"
with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as stream:
    stream.writelines(nav.build_literate_nav())
