# Agent instructions for private-machine (pm)

## Summary

This repository holds the private-machine (pm) companion runtime and its workflows.
Spamuel is the synthetic test persona. Its history is test data, not product data.
Read this file before any change.
For work-item development, also read the [work-item process](docs/work_items/process.md).
Use the terms of the [glossary](docs/concepts/glossary.md) and follow the [naming rules](docs/concepts/naming.md).
The [structural refactor](docs/work_items/refactor/index.md) is in progress. Follow its relocation map for new files.

## Repository layout

| Path | Content |
| --- | --- |
| `pm/` | The runtime: bus, bus modules, LLM access, records, memory store, managed runs, CLI, console. No companion behavior. |
| `workflows/` | All behavior. Each folder holds a manifest, its scripts and prompts, and `host.py`. `public_speech/` and `minecraft_body/` are shared libraries. `companion.py` is the assembly. |
| `main.py` | Launcher for the terminal session. |
| `tests/` | `unit/pm/<package>/` and `unit/workflows/<workflow>/` mirror the code. `contract/` runs workflows through the real kernel. `live/` needs real services. |
| `test_data/` | Synthetic personas (`personas/`) and captured receipts (`captures/`). |
| `dev/work_items/` | Work-item acceptance gates. They are development tools, not product code. |
| `dev/tools/` | Maintenance tools: skill compiler, reference generator, snapshots. |
| `docs/` | The engineering manual. `docs/work_items/` holds plans, decisions, and verification records. |

Product code must not import `dev/` or `tests/`. `pm/` must not import `workflows/`.
`tests/unit/test_architecture.py` checks these import directions.

## Evidence and dependency discipline

Read the [baseline evidence](docs/work_items/alpha-plan/baseline/receipts.md) and the selected item's recorded gates before changing behavior.
Read the [scenario matrix](docs/work_items/alpha-plan/baseline/scenarios.md) before making a behavioral comparison.
Use the [local commands](docs/work_items/alpha-plan/baseline/commands.md) for reproducible source capture and live operation.
Leave the preserved original user brief unchanged.
Keep source archives, full transcripts, checkpoints, and local credentials under ignored runtime paths.

Record repository revisions and installed package origins before changing a dependency.
Update affected consuming constraints and locks to the tested revisions after the change.
Verify the installed distribution, rather than assuming that a sibling checkout supplies its code.
Do not present source review of an older PM generation as proof that its runtime works.
Keep failed receipts visible until new real evidence establishes a repair.
Separate durable state equality from correct retrieval, useful replies, emotional continuity, and immersion.
Separate a completed workflow receipt from verified execution of the requested task.
Inspect actual file contents or world effects before claiming task success.
Record human review status without inventing ratings or treating a model's answer as a truth label.

## Comments and docstrings for human maintainers

Use the simple-english skill for all new or revised explanatory code prose.
Use short sentences, concrete terms, active voice, and consistent names.
Explain the reason for an ordering constraint or invariant where the reader needs it.
Document state ownership, commit boundaries, cancellation boundaries, retries, and uncertain external outcomes beside the relevant code.
If units, limits, provenance, or recovery behavior affect correctness, document them beside the relevant code.

Add useful docstrings to public APIs and non-obvious helpers.
Describe inputs, results, expected exceptions, side effects, and cancellation behavior.
Keep type annotations accurate.
Explain subtle branches and platform-specific behavior instead of making the reader reconstruct their purpose.
Keep comments current after a behavior change.
Avoid comments that merely repeat the next line of code.
Do not change identifiers, commands, quoted errors, or preserved source documents to satisfy a language rule.

## Required checks and HTML regeneration

Run the root foundation suite and the selected item's relevant real gates.
Keep live evidence under a fresh `.runtime` directory.
Record failures and unavailable endpoints without converting them into passes.

```powershell
uv run --locked python -m unittest discover -s tests/unit -t . -b
uv run --locked python -m unittest discover -s tests/contract -t . -b
```

If the native-exit anomaly recurs, retain its evidence and repeat with the locked interpreter's diagnostics:

```powershell
.venv/Scripts/python.exe -X faulthandler -m unittest discover -s tests/unit -t . -v
```

After updating code comments, docstrings, or manual pages, rebuild the complete HTML site:

```powershell
uv run --locked --with-requirements requirements-docs.txt mkdocs build --strict
```

Open `site/index.html` and inspect the changed guide and its API links.
Verify that local navigation points to HTML files and that the build reports no unresolved links.
The site uses `use_directory_urls: false` for local file access.
Generated HTML stays in ignored `site/` and must be regenerated locally after each delivery.
Fix documentation errors before reporting the item as delivered.
