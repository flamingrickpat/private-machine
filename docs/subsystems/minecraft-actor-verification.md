# A10 sustained actor verification

## Summary

The A10 gate uses the real Spamuel seed, configured chat endpoint, real Minecraft server, and production managed worker.
It measures five-tree collection and return separately from concurrency, updates, cancellation, and direct inventory dialogue.
Failed attempts remain in separate runtime directories.

Parent: [sustained actor](minecraft-actor.md).
Controls: [body ownership and instructions](minecraft-actor-controls.md).

## Repeat the live gate

Stop the owned Minecraft and MCP services before this command.
Keep the existing chat service available at `http://127.0.0.1:8080/v1`.
Use an unused output directory.

```powershell
uv run --locked python dev/work_items/alpha_a10_minecraft_gate.py --config config.alpha.example.yaml --out .runtime/alpha/a10-wood-repeat
```

[`dev.work_items.alpha_a10_minecraft_gate.run`][] starts `C:/source/minecraft-server-real` and the real MCP checkout.
It copies `.env_real` into ignored evidence and selects an isolated character directory.
It starts the ordinary survival player `tdd_player` in the actual world.
It does not reset the world or supply items.
Its receipt records the world seed and unchanged source-seed digest.
The current world, initial position, inventory, and terrain are part of each run's evidence.

The production supervisor starts the five-tree leaf.
The gate attempts a second actor during ownership and queues a real instruction update after useful harvest.
The update retains the original task and asks the actor to keep its logs.
The gate also runs real LLM action selection for a direct inventory question.
The production Assistant answers from one current observation while the actor remains active.
A separate native actor establishes cancellation without replacing the successful wood leaf.

## Acceptance facts

| Check | Required original evidence |
| --- | --- |
| Five-tree return | Distinct positive harvest bases, retained inventory, travel, final return, and fresh body state |
| Instruction update | Revision 1 reaches the gateway and final report |
| Second actor | Failed ownership receipt before world dispatch |
| Direct inventory | Real action selection and one executed observation receipt |
| Concurrent chat | Accepted public Assistant report while the managed actor remains running |
| Cancellation | Retained journal stops changing and ownership releases after cleanup |

The gate does not establish a house, iron pickaxe, or generic durable task tree.
Those gates belong to A11–A14.
It does not establish screenshot understanding.
Human persona ratings remain separate from factual checks.
The A09 operator ratings do not rate new A10 samples.

## Current evidence

The final real gate passed all seven checks in `.runtime/alpha/20261005-a10-wood-07/`.
The original receipt, journal, observations, native session, and written handoff remain in that directory.
Independent replay appears in `.runtime/alpha/a10/final-live-review.json`.
The session receipt and copyable handoff appear in `.runtime/alpha/a10/`.

The final leaf used 22 calls, harvested five distinct tree columns, and retained 25 additional birch logs: 96 became 121.
Fresh observations measured 22.993 blocks of travel and a 0.305-block return distance.
The gateway applied instruction revision 1 between calls.
The second actor failed before any world dispatch.
A separate actor dispatched a real walk call that returned `no_path` before cancellation.
Its unchanged journal proves that cancellation prevented later dispatches and released ownership.
The original receipt's `body_mutation_completed` flag means a non-read call returned; it does not prove a successful world mutation.

Normal LLM action selection chose the direct Assistant inventory route.
One successful observation reported 97 birch logs and one dirt block while the wood actor was running.
The accepted reply matched those contents and the active-task receipt.
It took 61.671 seconds; this result does not establish fluid conversation latency.
The reply started no new managed task and used no Laya routing.

The host preserved the native handoff's 2,790 characters.
Its final health statement became stale while the player remained in the world.
The host's later observation recorded health 3.93/20 and food 0.
The cancellation actor's last observation recorded health 0.93/20 and food 0.
Use the fresh body state before further survival work.
This gate establishes wood collection and return, rather than broad survival competence.

Earlier attempts remain preserved:

| Run | Original result and repair |
| --- | --- |
| `20261005-a10-wood-01` | Actor startup failed because the installed FastMCP API did not accept a function name keyword. Registration now uses `Tool.from_function`. |
| `20261005-a10-wood-02` | The actor collected wood, but concurrent chat retried a deterministic HTTP 500. Its synthesized tool call lacked `type: function`. The transcript and permanent-error classification are repaired. The operator stopped the owned services without resetting the world. |
| `20261005-a10-wood-03` | Wood return passed with 25 calls, five trees, 15 additional logs, 40.602 blocks of travel, and a 0.182-block return distance. The overall gate failed: the lease check captured an unfinished receipt, and the Assistant manifest omitted its observation grant. Raw tool markup was wrongly accepted as speech. All three boundaries are repaired. |
| `20261005-a10-wood-04` | The wood actor collected 26 additional logs through 47 calls. The direct observation overlapped a live command and returned `busy`; claim review also lacked proof of the active task. Per-command coordination and owner-receipt evidence repair these boundaries. |
| `20261005-a10-wood-05` | Wood return passed with 22 calls and 28 additional logs. The coordinated observation reported 59 birch logs and one dirt block. The candidate matched it, but review invented numeric IDs for tool evidence and correctly rejected those unknown IDs. Separate tool-receipt citations repair the missing integration. Native handoff inspection also found that the host replaced a written document with its path; the host now preserves the document. |
| `20261005-a10-wood-06` | Concurrent inventory dialogue passed in 221.594 seconds. The wood actor retained 12 additional logs through 32 calls, but the fresh final return distance was 3.544 blocks. The strict three-block requirement failed. The final run kept this requirement and did not reset the world. |

The initial focused suite passed 29 checks in 22.349 seconds.
A pre-inventory-repair foundation run passed 336 checks in 612.215 seconds.
These earlier results do not establish final-source acceptance.
The inventory and protocol repair subset passed 16 checks in 11.630 seconds.
It includes actual MCP listener-thread cleanup and captured parser and speech failures.
The command-boundary subset passed 38 checks in 12.651 seconds.
The citation and publication subset passed 29 checks in 0.898 seconds.
The final foundation passed 343 checks in 521.695 seconds.
The unchanged workflow package retains A08's 199-check result in 68.00 seconds; it was not rerun.
The strict documentation build regenerated the complete site.
All 221 HTML pages and their local links were checked; no missing targets, directory links, or broken anchors remained.
Changed guides and generated API pages were inspected in `.runtime/alpha/a10/html-inspection.json`.

## Foundation and HTML

```powershell
$testModules = Get-ChildItem tests/test_*.py | ForEach-Object { 'tests.' + $_.BaseName }
uv run --locked python -m unittest $testModules -b
uv run --locked --with-requirements requirements-docs.txt mkdocs build --strict
```

Regenerate the whole site and inspect the actor guides and their API links.
Local links must target actual HTML files and existing anchors.
Generated HTML remains under ignored `site/`.
