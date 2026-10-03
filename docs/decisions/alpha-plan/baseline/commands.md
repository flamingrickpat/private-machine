# Repeat the A01 baseline

## Summary

The commands reuse the foundation suite, conversation evaluator, console worker, and narrow Minecraft gate.
Each live run stores its own evidence under `.runtime`.
The Minecraft procedure uses the real server and its wipe script.
The Laya endpoint remains independent of the existing chat workflows.

Parent: [A01 baseline](index.md).
Results: [receipts](receipts.md).

## Prepare the environment

1. Open PowerShell in `C:/source/pm/pm_next_v2`.
2. Run `uv sync --locked`.
3. Start chat and embeddings on their configured ports.
4. Use `config.alpha.example.yaml` for the local A01 configuration.

| Service | Endpoint | Observed identity |
| --- | --- | --- |
| Chat and comparison LLM | `http://127.0.0.1:8080/v1` | `qwen`, Q4_K medium, 27,320,697,856 parameters |
| Embeddings | `http://127.0.0.1:8081/v1` | `pplx-embed-v1-0.6b`, F16, 1024 dimensions |
| Decisions | `http://127.0.0.1:8082/v1/systemone` | `laya`, BF16, 421,029,889 parameters |
| Console smoke | `http://127.0.0.1:8093` | Separate deployment and checkpoint |
| Minecraft | `127.0.0.1:12345` | Real Fabric 1.19.4 server, Babymode 0.5.0 |
| Minecraft MCP | `http://127.0.0.1:6767/mcp` | Body HTTP 6768. Viewer 6769 |
| Minecraft RCON | `127.0.0.1:25575` | Credentials remain local |

The default `config.example.yaml` still describes a remote embedding connection.
The A01 configuration selects the user's local embedding service without changing existing defaults.
The decision endpoint follows the [llama.cpp decision-model article](https://huggingface.co/blog/ggml-org/decision-models-in-llamacpp).
Start Laya separately only for its explicit client or candidate tests.
Normal LLM workflows do not require that service.
The [model policy](../model-policy.md) governs optional adoption after A02's quality results.
A02 supplies the public client and actual typed inference checks.
A01 service discovery does not establish Laya classification quality.
The running Laya service uses `C:/workspace/llama-systemone-b11374/llama-server.exe`, build 11374, revision `b92761a51`.
Chat and embeddings use the earlier build 11351 from `C:/workspace/llama13/`.

## Capture a fresh source snapshot

```powershell
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
uv run --locked python scripts/alpha_snapshot.py --config config.alpha.example.yaml --out ".runtime/alpha/$stamp-snapshot"
```

The snapshot records all four PM generations and five supporting PM repositories.
It also records the real server checkout, locks, installed package origins, persona hashes, and historical receipt hashes.
Source archives include nonignored working-tree files and separate binary Git patches.
Their configuration files can contain local credentials, so the archives stay under ignored `.runtime` paths.
The published handoff links a baseline Git commit for implementation reproduction.
Older runtime health remains unverified.

## Foundation and documentation gates

```powershell
$testModules = Get-ChildItem tests/test_*.py | ForEach-Object { 'tests.' + $_.BaseName }
uv run --locked python -m unittest $testModules -b
uv run --locked --with-requirements requirements-docs.txt mkdocs build --strict
uv run --locked python -m compileall -q scripts
```

The root suite excludes nested live gates.
The planning baseline is 257 tests in 363.527 seconds.
Documentation dependencies remain in `requirements-docs.txt`.

## Console exchange, evidence, and restart

```powershell
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
uv run --locked python scripts/alpha_console_smoke.py --config config.alpha.example.yaml --out ".runtime/alpha/$stamp-spamuel"
uv run --locked python scripts/alpha_console_smoke.py --config config.alpha.example.yaml --persona cheerful --out ".runtime/alpha/$stamp-cheerful"
```

The script starts the current console HTTP service and real worker.
It discusses a codeword, captures memory and task receipts, restarts, and asks for recall.
It compares accepted records, GWT, and managed-run receipts across restart.
It retains the deployment, worker trace, checkpoint, transcript, and compact receipt.
Add `--port 8094` if another isolated console run already owns 8093.
Its `passed` result establishes completed exchanges and durable restart equality.
Task dispatch and reply quality remain observed outcomes in those receipts.

For direct operator inspection:

```powershell
uv run --locked python -m console.server --root .runtime/alpha/a01/console-spamuel-dialogue/instances
```

Open `http://127.0.0.1:8090` and start the recorded instance.
Use **Read memory**, the trace, **Managed runs**, and **Worker log** to inspect the same evidence.
The automated smoke covers the HTTP/worker path.
A01 records no new browser usability or human immersion verdict.

For the separate task-request scenario, append `--request-task` to the smoke command.
The first A01 task-request run recorded a completed control worker but produced no public reply within 600 seconds.
That worker's artifacts do not verify the requested note contents.
Its failed receipt remains part of the baseline.

## Current LLM comparison

```powershell
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
uv run --locked python scripts/alpha_routing_comparison.py --config config.alpha.example.yaml --out ".runtime/alpha/$stamp-routing"
```

The script sends only development-case input and class descriptions to the current chat model.
It does not send expected labels or any holdout case.
It stores the unmodified responses and request timing.
The fixture hash fixes the comparison input.
Pending label review prevents a claim of accepted classification accuracy.

The existing evaluator remains available for the broader application smoke:

```powershell
$env:PM_MODEL_CONFIG = 'config.alpha.example.yaml'
$env:PM_W7_SCENARIOS = 'casual_banter,managed_start_and_status'
$env:PM_W7_EXCHANGES = '2'
$env:PM_W7_EVALUATE = '0'
uv run --locked python -m unittest tests.live.test_conversations -v -b
Remove-Item Env:PM_W7_SCENARIOS, Env:PM_W7_EXCHANGES, Env:PM_W7_EVALUATE, Env:PM_MODEL_CONFIG
```

For the full historical gate, omit the three `PM_W7_*` overrides.
Full acceptance requires 200 completed application cycles and at least seven good trajectories.
Generation completes before evaluation.
The evaluator cannot rewrite replies.

## Real Minecraft world and reset

The selected server is `C:/source/minecraft-server-real`.
Its configuration uses survival mode, normal difficulty, normal terrain, structures, and seed `23452352345`.
The server and MCP must be stopped before the managed gate starts.
The runner refuses occupied game, RCON, MCP, body, or viewer ports.

CAUTION: `--reset-world` removes the server's `world`, `logs`, and `.fabric` directories through its existing wipe script.
The runner validates every resolved removal target inside the selected server directory before the script runs.
The wipe script prints an older seed.
`server.properties` supplies the actual seed.

```powershell
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
uv run --locked python scripts/alpha_minecraft_gate.py --config config.alpha.example.yaml --out ".runtime/alpha/$stamp-minecraft" --reset-world
```

The runner starts the actual server directory and a separate test character.
Its copied `.env_real` enables test routes and selects a character directory inside the run's `.runtime` folder.
`/test/reset` clears that character's files, rather than the normal companion's character directory.
The runner checks the existing one-action W8 gate and stops its owned processes afterward.
It retains the real world and service logs after the gate.

The actual mod file is `mods/agentic-babymode-0.5.0.jar`.
Its SHA-256 is `d53cf9b424bc69ae6fd0bfad39e31c7d61089e91f65410952e2d4e864507d453`.
The active configuration file is `config/agentic_babymode.json`.
The README names `config/agentic-babymode.json`, but the current source resolves the mod ID `agentic_babymode` plus `.json`.
The server logged Babymode 0.5.0 startup, and the underscore configuration path matches its source.

A full world wipe resets world and player persistence at the next start.
Babymode `/commit` and `/revert` restore blocks and block entities only.
They do not restore inventory, player state, entities, time, weather, gamerules, advancements, or scoreboards.
Later fixtures must record and reset those separately if they use block rollback.
