# Codex project instructions

## Summary

Develop the companion alpha through its ordered work items.
The user requires real model inference, real Spamuel history, and the real Minecraft server for behavioral acceptance.
Every delivery must include readable code comments, detailed implementation documentation, and regenerated HTML.
This file contains general project instructions.
Session progress belongs in the copyable handoff at the end of each session.

## Start the next session

1. Read this file before implementation.
2. Read the [alpha plan](docs/decisions/alpha-plan/index.md) and the selected work item.
3. Read its linked design and the latest handoff that the user supplies.
4. Read applicable repository instructions and inspect uncommitted changes in every affected repository.
5. Open the `simple-english` skill before writing comments, docstrings, or documentation.
6. Open the `project-documentation` skill before updating the engineering manual.
7. Inspect the current implementation and live configuration before assuming that an older guide still applies.
8. Complete the selected item, its relevant checks, documentation, and handoff before moving to another item.

The installed skills use these locations:

- `C:/Users/rick/.agents/skills/simple-english/SKILL.md`
- `C:/Users/rick/.agents/skills/project-documentation/SKILL.md`

If a skill path changes, locate the installed skill before using a replacement procedure.
Use simple-english in its pragmatic mode unless the user requests strict STE compliance.

## Real acceptance is mandatory

The prototype phase with dummy acceptance is over.
Use the production path, actual services, and real input data to establish behavior.
Do not replace a difficult live gate with a fake world, fixed model response, or dummy conversation.
Do not introduce dummy tests as a substitute for a required live test.

Use `data/spamuel_agentic_canon.yaml` as the primary companion seed.
It contains the real Spamuel/Jesse history and approximately 11,200 records.
Do not replace this history with a tiny invented seed to obtain passing companion results.
Keep the source seed unchanged and use separate runtime checkpoints for experiments.

An existing checkpoint takes precedence over a changed seed.
Use a fresh instance to establish a new baseline from the real seed.
Use the same checkpoint for a continuity or restart scenario.
Preserve transcripts, state changes, task receipts, and observed world effects.

The alpha plan also requires a cheerful comparison persona.
Its existing fixture is `tests/fixtures/alpha/cheerful.yaml`.
Keep that control separate from the primary Spamuel acceptance evidence.
Its small, empty-history seed cannot establish behavior with the real Spamuel history.

Retain the existing foundation tests for state, routing, and protocol contracts.
Passing those tests does not establish model quality, memory recall, or gameplay success.
Use captured real protocol data and real failure conditions for new boundary checks.
Keep generated routing examples visibly distinct from captured application reports.

If a required service is unavailable, diagnose its actual connection and startup configuration.
If the configuration is available, start the service through the documented procedure.
If essential deployment information is missing, ask the user for that information.
Record an unavailable gate explicitly until it runs against the real service.
Never report dummy execution as a replacement pass.

## Known local deployment

The local deployment uses the following locations and endpoints.
Verify the active identities before each live run.

| Component | Real location or endpoint | Identity |
| --- | --- | --- |
| Main repository | `C:/source/pm/pm_next_v2` | Python 3.12, `uv`, Git |
| Local model configuration | `config.alpha.example.yaml` | Local chat, OpenRouter embeddings, environment credential reference |
| Chat and comparison LLM | `http://127.0.0.1:8080/v1` | Alias `qwen` |
| Embeddings | `https://openrouter.ai/api/v1` | `perplexity/pplx-embed-v1-0.6b`, 1024 dimensions, `OPENROUTER_API_KEY` |
| Decisions | `http://127.0.0.1:8082/v1/systemone` | Alias `laya` |
| Decision server binary | `C:/workspace/llama-systemone-b11374/llama-server.exe` | Build 11374, revision `b92761a51` |
| Decision model | `C:/source/ai/Laya-BF16.gguf` | Laya BF16 |
| Real Minecraft server | `C:/source/minecraft-server-real` | Fabric 1.19.4, Babymode 0.5.0 |
| Game socket | `127.0.0.1:12345` | Normal survival world |
| Minecraft MCP | `http://127.0.0.1:6767/mcp` | Body HTTP 6768, viewer 6769 |
| RCON | `127.0.0.1:25575` | Credentials stay in local configuration |
| MCP repository | `C:/source/pm/pm-minecraft` | Real-server environment: `.env_real` |
| Normal console | `http://127.0.0.1:8090` | Separate instance checkpoints |

Both example configurations use OpenRouter embeddings.
Select `config.alpha.example.yaml` explicitly for local chat with OpenRouter embeddings.
Chat currently runs through `C:/source/strata2/serve/server.py` with `C:/source/strata2/strata-orca-iq3_xxs.json`.
It serves Qwen3.8-Flash-Next-Uncensored under alias `qwen`, with 262,144 serving tokens.
The externally owned chat service remains running across Minecraft fixture shutdowns.
Earlier receipts used the `C:/workspace/llama13/` chat binary.
The unused local embedding endpoint on 8081 has a recorded invalid-vector defect.
Use the decision-capable binary listed above for Laya.
Do not launch a second decision server on a port that the console smoke uses.

## Use the real Minecraft world

The real server directory is **`C:/source/minecraft-server-real`**.
The user explicitly selected this server for baseline and acceptance work.
Its world is normal terrain with structures, survival mode, and normal difficulty.
The recorded seed is `23452352345`.

If you cannot find or identify the real Minecraft server, ask the user where it is.
Do not conclude that Minecraft acceptance is impossible after inspecting the superflat TDD template.
Do not substitute `C:/source/minecraft-server-tdd` for this real server.
The generic historical W8 autostart path uses that TDD template.
Use the real-server wrapper below to select the actual server.

The real server provides `wipe-world.ps1` and `start-server.ps1`.
The wipe script removes `world`, `logs`, and `.fabric` while the server is stopped.
Its printed seed is outdated.
Read the actual seed from `server.properties`.
The active Babymode configuration is `config/agentic_babymode.json`.

The user authorized the required real-server world resets for this testing procedure.
Before a reset, verify that the server is stopped.
Verify that every resolved removal target remains inside the named server directory.
Use the provided wrapper for an automated, owned server and MCP lifecycle:

```powershell
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
uv run --locked python dev/work_items/alpha_minecraft_gate.py --config config.alpha.example.yaml --out ".runtime/alpha/$stamp-minecraft" --reset-world
```

The wrapper uses the actual server directory and a separate test character.
It copies `.env_real` into the ignored run directory and enables test control routes there.
Its character directory stays inside that run's `.runtime` folder.
`/test/reset` clears those character files.
It does not reset the game world or the normal companion's character files.
The wrapper stops the game server and MCP processes that it owns and retains their evidence.

For later gameplay fixtures, record spawn, inventory, terrain, available resources, and the complete reset procedure.
Babymode `/commit` and `/revert` restore blocks and block entities only.
They do not restore inventory, player state, entities, time, weather, gamerules, advancements, or scoreboards.
If the fixture requires complete restoration of initial state, use a complete world reset.
Do not supply iron or a pickaxe to a scenario that requires self-mined iron and self-crafted tools.
Do not treat taking village supplies as mining iron.

The wrapper currently runs the existing narrow one-action gate.
It does not establish a finished house, an iron pickaxe, sustained survival, or visual interpretation.
Screenshot files alone do not establish that a planner understood their pixels.

## Evidence and dependency discipline

Read the [baseline evidence](docs/decisions/alpha-plan/baseline/receipts.md) and the selected item's recorded gates before changing behavior.
Read the [scenario matrix](docs/decisions/alpha-plan/baseline/scenarios.md) before making a behavioral comparison.
Use the [local commands](docs/decisions/alpha-plan/baseline/commands.md) for reproducible source capture and live operation.
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

## Decision capability

Use the normal LLM as the default for selection, classification, and interpretation.
Laya was planned for frequent classification, but A02 found poor interpretation quality, including confidently wrong answers.
Retain its technology for later use.
Use Laya only for the simplest classifications after realistic live-endpoint unit tests establish that specific use.
If making Laya work requires expensive effort, use the normal LLM and stop the experiment.
Do not enforce a percentage of Laya usage or require adoption for an alpha item to pass.
Read the [model policy](docs/decisions/alpha-plan/model-policy.md) before implementing classification or selecting a cognitive backend.

Each classifier needs use-specific classes, descriptions, input limits, and an uncertainty policy.
Always provide an INVALID route to a scoped LLM handler for missing, ambiguous, stale, oversized, or unusable evidence.
Do not rely on confidence alone to catch wrong answers.
Write unit tests that call the real endpoint with realistic synthetic inputs and independently defined expected results.
Include captured application inputs when available, and preserve input digests.
Synthetic inputs do not permit fixed model responses or simulated inference.
Simple fact-based test assertions need no separate human-label approval.
Keep human review for ambiguous interpretation and persona immersion.
Preserve the historical A01 corpus, holdout, and their pending review status.
Use untouched cases before enabling a specific optional classifier.

The normal LLM path is a core requirement.
Diagnose and repair direct Python failures or lost required context immediately.
Preserve pm-coder's automatic compaction for accumulated agent execution history.
Direct chat callers must select suitable task context or explicitly compact it while preserving necessary evidence.
Do not treat Laya's decision-head limits as limits on chat or pm-coder.

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

## Detailed documentation is part of every item

Use the project-documentation skill with simple-english for the engineering manual.
Explain the actual implementation in enough detail for a human to modify, operate, and diagnose it.
Brief prototype summaries are insufficient for the real modules.
Keep each page focused and add linked child pages when a mechanism needs a separate explanation.

For every delivered mechanism, document:

- Its purpose, entry points, callers, and owning repository.
- Its exact inputs, output schemas, accepted values, and user-visible results.
- Its real execution path, workflow stages, routing conditions, and fallback path.
- Its state owner, revisions, durable commit boundary, and publication behavior.
- Its external effects, cancellation boundaries, retries, restart behavior, and uncertain outcomes.
- Its configuration, live endpoints, activation procedure, model and tool requirements, and effective limits.
- Its memory provenance, evidence files, trace fields, and steps to inspect a real result.
- Its concrete failure modes, diagnostic steps, and tested recovery procedures.
- Its exact commands, representative real-input examples, measured results, and remaining limits.
- Its API docstrings and links from the conceptual explanation to the exact Python objects.

Separate active behavior, verified results, known defects, proposals, and unverified assumptions.
After a contract, workflow, or procedure changes, update affected guides and navigation.
Describe failure and recovery as thoroughly as the successful path.
Do not copy dummy examples into a guide and label them as real execution.
Use observed data and actual receipts for worked examples.
Link deeper pages back to their parent and related mechanisms.
Keep the API reference generated from source and docstrings.
Do not duplicate API declarations by hand in conceptual pages.

## Required checks and HTML regeneration

Run the root foundation suite and the selected item's relevant real gates.
Keep live evidence under a fresh `.runtime` directory.
Record failures and unavailable endpoints without converting them into passes.

```powershell
$testModules = Get-ChildItem tests/test_*.py | ForEach-Object { 'tests.' + $_.BaseName }
uv run --locked python -m unittest $testModules -b
```

If the native-exit anomaly recurs, retain its evidence and repeat with the locked interpreter's diagnostics:

```powershell
.venv/Scripts/python.exe -X faulthandler -m unittest $testModules -v
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

## Finish each session

The user authorized commits and pushes for related projects under `C:/source/pm`.
Commit and push the tested changes and documentation within the selected item's scope.
Preserve unrelated uncommitted work and local credentials.
Record changed contracts, exact revisions, lock changes, commands, real receipts, and remaining limits.
Retain diagnosed failures until the owning item fixes them with new real evidence.

If the session makes changes, write a self-contained, copyable handoff in the final response.
Include completed work, affected repositories and commits, validation commands and results, known failures, pending work, and the next item.
Include exact evidence paths and deployment changes that the next agent needs.
The user copies that handoff into the next session.
Do not append session history, current progress, next-item status, or handoff notes to this file.
Update this file only when general project instructions or deployment facts change.
