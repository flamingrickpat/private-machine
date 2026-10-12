# Run A13 failure fixtures

## Summary

The real fixture runner separates operator setup, protocol boundaries, and native adaptation.
It retains actual tool calls and checks world effects independently.
Hazard runs, resource adaptation, and project replanning establish different outcomes.

Parent: [A13 verification](index.md).

## Captured boundary evidence

`test_data/captures/alpha/a13-gameplay-results.json` retains original receipts and their source paths.
It includes blocked navigation, an unreachable mining target, missing crafting ingredients, partial construction, and a negative local search.
The partial construction result reports three placed cells while the action remains unsuccessful.
The search retains its original bounds and observation time.
These foundation inputs do not establish new autonomous gameplay acceptance.

`tests/test_a13_gameplay.py` checks outcome distinctions, retained quantities, paused methods, diagnostic freshness, and concrete deferral.
It also rejects an older planning observation after a current body observation.
The original failed A11 and A12 receipts remain preserved.

`a13-water-and-path.json` adds actual A13 blocked-path, wrong-tool, water, and negative ore-search receipts.
`a13-dry-grass-escape.json` retains the native actor's original escape observations and geometry.
That regression proves the projection can accept the captured dry landing after two stable observations.
It does not convert the original failed live receipt into a pass.

`a13-native-m3-contract-repair.json` retains the actual native revision-five contract repair and its preservation audit.
It proves complete scenario operands while retaining accepted work; it does not prove the house or iron chain.
`a13-region-air-count.json` retains verifier 70's original site reads and matching specification.
Its 75 empty cells use a separate aggregate counter; one foundation cell remains unseen.
The boundary checks accept the aggregate only under an explicit all-air-variants criterion projection.
The original failed native receipt remains failed.

## Failure fixture procedure

[`dev.work_items.alpha_a13_failure_gate.run`][] creates observed temporary obstructions and a liquid source through recorded operator commands.
It grants no items.
The default water placement is a contained one-block pit at the observed body.
The `--water-placement head` variant retains the earlier flowing head-space setup.
The default lava placement is an adjacent pit below feet height.
Ordinary survival mining prepares that pit; operator setup places only its temporary liquid source.
The `--lava-placement head` variant reproduces the dangerous immersion setup.
The lava-edge audit requires actual retreat and rejects recorded player deaths.
The optional `--reset-world` flag uses the same checked real-server wipe procedure as M3.
Protocol calls exercise blocked movement, an unreachable block, missing ingredients, wrong harvesting tools, and an empty local ore search.
After two ineffective repetitions, a third unchanged call must be rejected before dispatch.
The actual native recovery actor must choose a diagnosed action and establish safety before cleanup.
Water and lava runs establish their hazard behavior. The resource cases and concrete project replanning have separate evidence.
The `--hazard none` mode starts four sequential native cases after the repeated protocol failures.
These cases cover tool and material prerequisites, blocked navigation, unreachable stone, and changed ore-search coverage.
Each case retains the original failure journal and every earlier case journal.
They use the same full-history companion, real world, and body lease.
There is no reset or item grant between cases.
Each case must finish safely and produce its required real effects.
The server death audit rejects survival claims based on a respawn.
Standalone actor reports remain completion candidates.
Their managed receipts reject public completion without generic leaf verification.
The separate fixture audit checks original effects and final server observations; it does not change those managed receipts.
It audits changed navigation, reachable alternative mining, corrected harvesting tools, crafted sticks, and changed ore-search coverage.
Its audit requires diagnosed actions or surveyed changed methods, plus original material deltas.
Unchanged paused mining and search methods still require their bound diagnosis.
Crafted sticks establish a concrete changed prerequisite method for the failed iron recipe.
Iron remains unfinished when only its stick prerequisite succeeds.

```powershell
uv run --locked python dev/work_items/alpha_a13_failure_gate.py --config config.alpha.example.yaml --out .runtime/alpha/a13-water-new --hazard water --water-placement pit --timeout 900
uv run --locked python dev/work_items/alpha_a13_failure_gate.py --config config.alpha.example.yaml --out .runtime/alpha/a13-lava-new --hazard lava --lava-placement edge --reset-world --timeout 900
uv run --locked python dev/work_items/alpha_a13_failure_gate.py --config config.alpha.example.yaml --out .runtime/alpha/a13-adaptation-new --hazard none --reset-world --timeout 1800
```

Use fresh output directories.
`--restore-fixture` requires original air evidence before it removes retained setup blocks.
The actor must inspect the current state after any operator restoration.

## Retain partial work

Use `--history-fixture` with a prior adaptation directory to retain its actual native journals.
The runner follows the complete recorded fixture chain and saves original journal digests.
It rejects changed seeds, resets, grants, cycles, and journals outside `.runtime`.
Current observations reconcile stock after the older effects. Each new case still needs its own completed report and unchanged lifecycle.
The prerequisite actor reuses suitable existing tools. Operator setup equips an existing block item to reproduce the wrong-tool boundary.

The runner also replays the captured malformed memory operands before native work.
Local `invalid_arguments` must leave the dispatch sequence unchanged and permit a following observation.
Inventory, body identity, and death count must remain unchanged.
`a13-invalid-arguments.json` preserves the original upstream rejection and valid earlier calls.

## Supported water-fixture preparation

The current body refuses digging beneath the player's support columns.
Water preparation therefore walks to a surveyed dry side and centers the body before digging the adjacent soil cell.
It then enters the newly dug dry pit through ordinary walking.
The setup receipt retains the standing side, dig, and entry.
Only the temporary water source uses the permitted operator fixture.
Fresh native escape evidence must follow this revised preparation; the earlier source-26 pass keeps its historical scope.
