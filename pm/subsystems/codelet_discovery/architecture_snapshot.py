from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pm.utils.token_utils import get_token_count


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_TOKENS = 4000
DEFAULT_INPUT_BUDGET_TOKENS = 22000
DEFAULT_OUTPUT_MD = ROOT / "architecture_description.md"
DEFAULT_OUTPUT_YAML = ROOT / "architecture_description.yaml"
ALLOWED_EXTENSIONS = {".py", ".md", ".yaml", ".yml"}
IGNORED_DIR_NAMES = {
    ".git",
    ".idea",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "data",
    "logs",
    "test",
    "tests",
}


@dataclass
class FileDigest:
    path: Path
    relative_path: str
    importance: float
    category: str
    token_estimate: int
    summary: str


def _iter_source_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in IGNORED_DIR_NAMES for part in path.parts):
            continue
        if path.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue
        yield path


def _categorize_file(relative_path: str) -> tuple[str, float]:
    normalized = relative_path.replace("\\", "/").lower()

    if normalized.startswith("pm/model/knoxel"):
        return "core data model", 1.00
    if normalized.startswith("pm/model/"):
        return "data model", 0.88
    if normalized.startswith("pm/ghost/"):
        return "cognitive runtime", 0.98
    if normalized.startswith("pm/subsystems/procedures/"):
        return "cognitive procedures", 0.94
    if normalized.startswith("pm/subsystems/memory/"):
        return "memory subsystem", 0.92
    if normalized.startswith("pm/subsystems/codelet/"):
        return "codelet subsystem", 0.90
    if normalized.startswith("pm/subsystems/context/"):
        return "context subsystem", 0.68
    if normalized.startswith("pm/agents/definitions/"):
        return "agent definitions", 0.84
    if normalized.startswith("pm/agents/agent_base.py"):
        return "agent runtime", 0.92
    if normalized.startswith("pm/agents/"):
        return "agent orchestration", 0.80
    if normalized.startswith("pm/system/llm/"):
        return "llm runtime", 0.78
    if normalized.startswith("pm/system/"):
        return "system configuration", 0.62
    if normalized.startswith("pm/persist/"):
        return "persistence", 0.52
    if normalized.startswith("pm/data/prompts/"):
        return "prompt assets", 0.20
    if normalized.startswith("pm/utils/"):
        return "utility layer", 0.00
    if normalized.startswith("main.py"):
        return "entrypoint", 0.72
    if normalized.startswith("prototype_"):
        return "prototype script", 0.40
    if normalized.startswith("training/"):
        return "training helper", 0.10
    if normalized.startswith("debug/"):
        return "debug helper", 0.05
    return "misc", 0.18


def _safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def _summarize_python_file(path: Path, relative_path: str) -> str:
    text = _safe_read_text(path)
    line_count = len(text.splitlines())
    token_estimate = get_token_count(text)

    try:
        module = ast.parse(text)
    except SyntaxError:
        clipped = text[:1000].replace("\n", " ").strip()
        return (
            f"path={relative_path}\n"
            f"type=python\n"
            f"lines={line_count} tokens~={token_estimate}\n"
            f"summary=Could not parse with ast; treat as raw source.\n"
            f"excerpt={clipped}"
        )

    docstring = ast.get_docstring(module) or ""
    imports: list[str] = []
    classes: list[str] = []
    functions: list[str] = []
    assignments: list[str] = []

    for node in module.body:
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names[:5])
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            imports.append(mod)
        elif isinstance(node, ast.ClassDef):
            bases = []
            for base in node.bases[:3]:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    bases.append(base.attr)
            base_text = f"({', '.join(bases)})" if bases else ""
            classes.append(f"{node.name}{base_text}")
        elif isinstance(node, ast.FunctionDef):
            functions.append(node.name)
        elif isinstance(node, ast.AsyncFunctionDef):
            functions.append(f"async {node.name}")
        elif isinstance(node, ast.Assign):
            for target in node.targets[:3]:
                if isinstance(target, ast.Name):
                    assignments.append(target.id)

    sections = [
        f"path={relative_path}",
        "type=python",
        f"lines={line_count} tokens~={token_estimate}",
    ]
    if docstring:
        sections.append(f"docstring={docstring[:300].replace(chr(10), ' ')}")
    if imports:
        sections.append(f"imports={', '.join(imports[:10])}")
    if classes:
        sections.append(f"classes={', '.join(classes[:12])}")
    if functions:
        sections.append(f"functions={', '.join(functions[:16])}")
    if assignments:
        sections.append(f"globals={', '.join(assignments[:12])}")
    return "\n".join(sections)


def _summarize_text_file(path: Path, relative_path: str) -> str:
    text = _safe_read_text(path)
    line_count = len(text.splitlines())
    token_estimate = get_token_count(text)
    clipped = text[:1800].replace("\n", " ").strip()
    return (
        f"path={relative_path}\n"
        f"type=text\n"
        f"lines={line_count} tokens~={token_estimate}\n"
        f"excerpt={clipped}"
    )


def _build_file_digest(path: Path) -> FileDigest:
    relative_path = path.relative_to(ROOT).as_posix()
    category, importance = _categorize_file(relative_path)
    summary = _summarize_python_file(path, relative_path) if path.suffix.lower() == ".py" else _summarize_text_file(path, relative_path)
    return FileDigest(
        path=path,
        relative_path=relative_path,
        importance=importance,
        category=category,
        token_estimate=get_token_count(summary),
        summary=summary,
    )


def _select_digests_for_prompt(digests: list[FileDigest], input_budget_tokens: int) -> list[FileDigest]:
    selected: list[FileDigest] = []
    used_tokens = 0

    for digest in sorted(
        digests,
        key=lambda item: (item.importance, -len(item.relative_path)),
        reverse=True,
    ):
        if digest.importance <= 0:
            continue
        candidate_tokens = digest.token_estimate + 24
        if used_tokens + candidate_tokens > input_budget_tokens and selected:
            continue
        selected.append(digest)
        used_tokens += candidate_tokens

    return selected


def _build_prompt(digests: list[FileDigest], max_output_tokens: int, output_format: str) -> list[tuple[str, str]]:
    lines: list[str] = []
    lines.append("Repository architecture digest, ordered by importance.")
    lines.append("Higher importance files define ontology, cognitive cycle, memory, codelets, agents, or LLM/runtime control.")
    lines.append("Utility files are intentionally omitted or heavily down-weighted.")
    lines.append("")

    for digest in digests:
        lines.append(
            f"## {digest.relative_path} | category={digest.category} | importance={digest.importance:.2f}"
        )
        lines.append(digest.summary)
        lines.append("")

    system_prompt = (
        "You are compiling an architecture description for an AI companion codebase.\n"
        "Write a precise technical overview of the CURRENT implemented architecture.\n"
        "Prefer implemented mechanisms over aspirations.\n"
        "Infer system relationships from the file summaries, but do not invent subsystems that are not supported.\n"
        "Focus most on high-importance files and treat omitted utility files as non-architectural unless clearly central.\n"
        "Mention uncertainty explicitly when a connection is only weakly evidenced.\n"
    )

    if output_format == "yaml":
        format_instructions = (
            f"Return YAML only. Keep it under about {max_output_tokens} tokens.\n"
            "Recommended top-level keys: overview, runtime_flow, core_types, subsystems, agents, llm_layer, open_questions.\n"
            "Use compact but readable prose strings and lists."
        )
    else:
        format_instructions = (
            f"Return Markdown only. Keep it under about {max_output_tokens} tokens.\n"
            "Use short sections. Recommended sections: Overview, Runtime Flow, Core Types, Subsystems, Agent Layer, LLM Layer, Open Questions.\n"
            "Be dense and implementation-focused."
        )

    user_prompt = (
        f"{format_instructions}\n\n"
        "Use the following architecture evidence.\n\n"
        + "\n".join(lines)
    )
    return [("system", system_prompt), ("user", user_prompt)]


def _default_output_path(output_format: str) -> Path:
    return DEFAULT_OUTPUT_YAML if output_format == "yaml" else DEFAULT_OUTPUT_MD


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile a weighted architecture description from the current repository.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--tokens", type=int, default=DEFAULT_OUTPUT_TOKENS, help="Target max output size in approximate tokens")
    parser.add_argument("--input-budget", type=int, default=DEFAULT_INPUT_BUDGET_TOKENS, help="Approximate token budget for repository evidence sent to the model")
    parser.add_argument("--format", choices=["md", "yaml"], default="md", help="Output format")
    parser.add_argument("--output", default=None, help="Output file path")
    parser.add_argument("--include-zero-importance", action="store_true", help="Also include files with 0.0 importance")
    args = parser.parse_args()

    from pm.system.llm.llm_common import CommonCompSettings, LlmPreset
    from pm.system.llm.llm_proxy import start_llm_thread
    from pm.system.load_config import load_config

    digests = [_build_file_digest(path) for path in _iter_source_files(ROOT)]
    if not args.include_zero_importance:
        digests = [digest for digest in digests if digest.importance > 0]

    selected_digests = _select_digests_for_prompt(digests, input_budget_tokens=args.input_budget)
    if not selected_digests:
        raise RuntimeError("No source files were selected for the architecture prompt.")

    output_format = "yaml" if args.format == "yaml" else "md"
    messages = _build_prompt(selected_digests, max_output_tokens=args.tokens, output_format=output_format)

    cfg = load_config(args.config)
    llm = start_llm_thread(cfg)
    result = llm.completion_text(
        LlmPreset.Default,
        messages,
        comp_settings=CommonCompSettings(
            max_tokens=args.tokens,
            temperature=0.2,
            caller_id="architecture_snapshot",
            enable_thinking=False,
        ),
    ).strip()

    output_path = Path(args.output) if args.output else _default_output_path(output_format)
    output_path.write_text(result + "\n", encoding="utf-8")

    selected_count = len(selected_digests)
    total_count = len(digests)
    print(f"Wrote {output_path}")
    print(f"Selected {selected_count}/{total_count} files for prompt")
    print(f"Approx input tokens: {sum(item.token_estimate for item in selected_digests)}")
    print(f"Approx output tokens: {get_token_count(result)}")


if __name__ == "__main__":
    main()
