"""Compile scoped Minecraft resources without changing the upstream skill."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
SECTIONS = {"shared": [*range(1, 10), 36, 37, 39, 40, 41, 42, 49, 50],
            "navigator": [10, 11, 12, 13, 26], "recovery": [14, 15, 16, 17, 18, 33, 34, 35],
            "gatherer": [19, 20, 21, 22, 23, 27, 28, 29], "builder": [24, 25],
            "miner": [30, 31, 32]}


def compile_skill(source: Path, output: Path) -> dict:
    """Copy upstream sections and local guidance; record every source digest."""
    raw = source.read_bytes()
    text = raw.decode("utf-8")
    matches = list(re.finditer(r"^## (\d+)\. .*$", text, flags=re.MULTILINE))
    sections = {int(match.group(1)): text[match.start():matches[index + 1].start() if index + 1 < len(matches) else len(text)]
                for index, match in enumerate(matches)}
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"source": str(source), "source_sha256": hashlib.sha256(raw).hexdigest(), "resources": {}}
    for name, numbers in SECTIONS.items():
        content = f"# Minecraft {name} resource\n\nCompiled from minecraft-survival-coach. Shared and specialist resources preserve the upstream sections.\n\n"
        content += "\n".join(sections[number] for number in numbers)
        guidance = ROOT / "workflows/minecraft/guidance" / f"{name}.md"
        addition = {}
        if guidance.is_file():
            content = content.rstrip("\n") + "\n" + guidance.read_text(encoding="utf-8")
            addition = {"guidance_source": str(guidance.relative_to(ROOT)),
                        "guidance_sha256": hashlib.sha256(guidance.read_bytes()).hexdigest()}
        target = output / f"{name}.md"
        target.write_text(content, encoding="utf-8", newline="\n")
        manifest["resources"][name] = {"sections": numbers, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), **addition}
    (output / "source.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("C:/Users/rick/.agents/skills/minecraft-survival-coach/SKILL.md"))
    parser.add_argument("--output", type=Path, default=ROOT / "workflows/minecraft/resources")
    args = parser.parse_args()
    print(json.dumps(compile_skill(args.source, args.output), indent=2))
