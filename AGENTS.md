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

## Local development planning

Use the [local Kanban board](docs/workflows/kanban.md) for development planning and current card status.
Before planning work, read `dev/kanban/board.json` and the relevant work-item documents.
Update an existing card when it already covers the requested work.
Add cards for new work that the user discusses or requests.
Keep undeveloped ideas in `backlog`, agreed queued work in `todo`, and current work in `doing`.
Move a card to `done` after the requested work and its relevant checks are complete.
Record blockers, pending owner review, work-item paths, and evidence paths in the card notes.
Keep detailed designs, acceptance gates, and verification records in the work-item documents.
A board status does not replace execution evidence or human review.

Edit the JSON file directly to manage cards without a running server.
Preserve existing card ids and unrelated cards.
Give each new card the current `next_id`, then increase `next_id` by one.
Use `id`, `title`, `description`, and `status` for each card.
Do not infer current card status from historical plans.
Complete direct file edits between browser saves.
After a direct edit, the user can press **Reload** in the browser.
Start the browser board with `dev/kanban/start.ps1`.
The board saves locally and does not create Git commits or push changes.

## Base references

The research PDFs and their converted TXT files are base references for companion architecture and behavior.
Use the relevant texts when designing or changing cognition, memory, appraisal, codelets, introspection, or reflection.
The local text collection is `.runtime/research/a18c-20261017-013806/`.
Its `index.json` identifies the original PDF filenames and page counts.
The original PDFs are under `C:/source/pm/pm_next/documents/pdfs/`.
Use the [curated source guide](docs/work_items/alpha-plan/cognitive-sources.md) to locate relevant mechanisms and passages.
Read the source text when its details affect a design decision; summaries do not replace the base references.
Consult the original PDF when diagrams, page references, or extraction gaps matter.
Current user instructions and accepted work-item requirements determine scope.
References inform design; real execution evidence establishes runtime behavior.

## Preferred owner review

Use runtime record packages and the browser review desk as the preferred path for owner review.
Follow the [record-package guide](docs/subsystems/record-packages.md) and [review desk instructions](dev/review/README.md).
Use `pm.introspection.record_packages` to capture named collections from a frozen snapshot.
Include actual chat from the user's perspective (Jesse in Spamuel tests), relevant GWT ticks, and private thoughts.
Include appraisals, codelet output, and the memory contents supplied to later decisions.
Show recorded links from the inspected knoxels through attention to the later routing choice and public reply.
Preserve record text, chronology, revisions, provenance, uncertainty, and useful metadata.
Keep embeddings hidden. Let the reader show useful metadata on hover and all captured fields on expansion.

Register the completed package files with `dev/review/server.py register-package` and provide the direct browser review URL.
Keep ratings beside the captured content and save feedback separately from immutable evidence.
Never invent human ratings or replace them with model scores.
When the owner asks to check feedback, read it with `uv run --locked python dev/review/server.py feedback`.

Fix engineering failures before asking the owner to review companion quality.
Retain failed receipts as development evidence outside the ready review package.
Do not substitute loosely connected Markdown reports, hashes, timing statistics, or stack traces for the owner's review.
Keep final handoffs concise and lead with the review URL when human review is the next step.

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

## Prompt scope and quality

Quality takes priority over minimizing model calls.
Do not force a reasoning or review task into one prompt.
Use smaller, focused calls when they improve quality.
Preserve required facts, source identifiers, chronology, and uncertainty across those calls.
Use the applicable context limits. Do not silently remove necessary evidence to fit a prompt.
This rule applies to all prompts, not only test agents or reviewers.

## Default behavior and edge cases

Keep expected default behavior as the primary path.
Treat a new "what about this?" situation as an additional edge case unless the owner explicitly changes the default.
Do not replace the main behavior with the edge case.
Exercise edge cases separately and give them lower priority than normal use.

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

Run only the tests that cover the affected code by default.
Use the smallest relevant selection that covers the changes and important related mechanics.
The owner actively tests the application during this phase.
Do not run the full unit suite, full contract suite, or all tests unless the user explicitly requests that run.
Do not treat a request to update tests, commit, or push as a request for a full suite.
Run the selected item's relevant real gates only when the current task requires them.
Keep live evidence under a fresh `.runtime` directory.
Record failures and unavailable endpoints without converting them into passes.

Use these full-suite commands only after an explicit user request:

```powershell
uv run --locked python -m unittest discover -s tests/unit -t . -b
uv run --locked python -m unittest discover -s tests/contract -t . -b
```

If a requested full unit run has a native exit, retain its evidence.
Do not start a full diagnostic repeat unless the user explicitly requests it.
For a requested diagnostic repeat, use the locked interpreter's diagnostics:

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
