"""Generate module references without importing application entry points."""
from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()
roots = [Path("pm"), Path("workflows")]
paths = [p for root in roots for p in root.rglob("*.py")]
paths += [Path("main.py")]
paths += [Path("dev/work_items/alpha_input_gate.py"), Path("dev/work_items/alpha_classification_gate.py"), Path("dev/work_items/alpha_pm_classification_gate.py")]
paths += [Path("dev/work_items/alpha_a08_memory_gate.py"), Path("dev/tools/alpha_memory_index.py")]
paths += [Path("dev/work_items/alpha_a09_dialogue_gate.py")]
paths += [Path("dev/work_items/alpha_a09_public_gate.py")]
paths += [Path("dev/tools/compile_minecraft_skill.py"), Path("dev/work_items/alpha_a10_minecraft_gate.py")]
paths += [Path("dev/work_items/alpha_a11_minecraft_gate.py")]
paths += [Path("dev/work_items/alpha_a12_recovery_gate.py")]
paths += [Path("dev/work_items/alpha_a12_publication_gate.py")]
paths += [Path("dev/work_items/alpha_a13_minecraft_gate.py")]
paths += [Path("dev/work_items/alpha_a13_failure_gate.py")]
paths += [Path("dev/work_items/alpha_a13_lifecycle_gate.py")]
paths += [Path("dev/work_items/alpha_a14_cooperation_gate.py")]
paths += [Path("dev/work_items/alpha_a14_draft_gate.py")]
paths += [Path("dev/work_items/alpha_a14b_memory_gate.py")]
paths += [Path("dev/work_items/alpha_a14_dialogue_gate.py")]
paths += list(Path("dev/work_items/a18b").glob("*.py"))
paths += [Path("dev/work_items/alpha_a18c_wait_gate.py")]
paths += [Path("dev/work_items/alpha_a19_reflection_gate.py")]
paths += [Path("dev/work_items/alpha_a19_review.py"), Path("dev/review/cognitive_trace.py")]
paths += [Path("dev/review/record_format.py")]
for path in sorted(paths):
    if "tests" in path.parts or any(part.startswith("_") for part in path.parts):
        continue
    parts = path.with_suffix("").parts
    target = Path("reference", *parts).with_suffix(".md")
    nav[parts] = str(Path(*parts).with_suffix(".md"))
    with mkdocs_gen_files.open(target, "w") as stream:
        stream.write(f"# {'.'.join(parts)}\n\n::: {'.'.join(parts)}\n")
    mkdocs_gen_files.set_edit_path(target, path)
# The context API lives in its package initializer, which the file scan skips.
with mkdocs_gen_files.open("reference/pm/context_render.md", "w") as stream:
    stream.write("# pm.context_render\n\n::: pm.context_render\n")
nav[("pm", "context_render")] = "pm/context_render.md"
# The A18 self-model API lives in its package initializer too.
with mkdocs_gen_files.open("reference/pm/introspection.md", "w") as stream:
    stream.write("# pm.introspection\n\n::: pm.introspection\n")
nav[("pm", "introspection")] = "pm/introspection.md"
# Render the installed dependency so the manual matches the consuming lock.
with mkdocs_gen_files.open("reference/pm_decision.md", "w") as stream:
    stream.write("# pm_decision\n\n::: pm_decision\n    options:\n        inherited_members: [__init__]\n        show_if_no_docstring: true\n")
nav[("pm_decision",)] = "pm_decision.md"
# Read the installed workflow package so API links match the consuming lock.
for module in ("inputs", "protocol", "working_target", "python_role", "kernel", "child", "manifest", "journal", "classification", "runtime", "dryrun"):
    identifier = f"pm_workflows.{module}"
    target = f"reference/pm_workflows/{module}.md"
    with mkdocs_gen_files.open(target, "w") as stream:
        stream.write(f"# {identifier}\n\n::: {identifier}\n")
    nav[("pm_workflows", module)] = f"pm_workflows/{module}.md"
with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as stream:
    stream.writelines(nav.build_literate_nav())
