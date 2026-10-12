# A11 verification

## Summary

A11 verifies graph contracts with foundation checks and real world projects with separate retained gates.
The live procedure starts the actual Fabric server and MCP.
It uses the unchanged full Spamuel history and the configured normal LLM.
Every failed attempt remains available beside later evidence.

Parent: [durable projects](minecraft-tasks.md).
Evidence contract: [independent observations](minecraft-task-evidence.md).

## Boundary checks

Run:

```powershell
uv run --locked python -m unittest tests.test_a11_minecraft_tasks -b
```

The checks cover dependency order, missing dependencies, cycles, stable revisions, and controlled refinement.
Captured A10 observations establish inventory and exact block-evidence parsing.
Those captured reads do not establish a new Minecraft action or model interpretation.
Actor-only completion claims, missing reads, duplicate reads, and stale verifier identity fail acceptance.
Revision checks use generated contract envelopes around unchanged captured world reads.
They establish graph and evidence-file guards, rather than actor behavior or model quality.
Accepted-reference checks require an exact existing independent certificate.
Captured A11 region reads include incomplete visible coverage after real gathering.
They establish that remaining world logs do not prove an earlier harvest route.
Captured plant inspection retains `is_solid=false` for real tall grass.
Scoped decision inputs preserve that field instead of describing every non-air cell as mineable.
File-commit checks use real Windows file locks.
They verify local retry and explicit failure without replaying a world role.

## Inspect a real plan

Run with a fresh output directory:

```powershell
uv run --locked python dev/work_items/alpha_a11_minecraft_gate.py --config config.alpha.example.yaml --out .runtime/alpha/NEW-a11-plan --plan-only
```

[`dev.work_items.alpha_a11_minecraft_gate.run`][] owns the real stack.
It records fresh vitals before starting project work.
`reviewed-tree.json` contains the plan that passed independent scope review.
Inspect the gathering, crafting, construction, and entry/exit prerequisites before execution.
The plan-only run does not mutate the world.

## Execute the real project

```powershell
uv run --locked python dev/work_items/alpha_a11_minecraft_gate.py --config config.alpha.example.yaml --out .runtime/alpha/NEW-a11-house --source-tree .runtime/alpha/NEW-a11-plan/reviewed-tree.json
```

The optional reset uses the authorized real-server wipe procedure.
The wrapper first requires free server and MCP ports.
It verifies that each removal target stays inside `C:/source/minecraft-server-real`.
The actual seed remains `23452352345`.
No external item grant, creative mode, or actor host command supplies gameplay progress.
The existing typed MCP shape helper uses server-assisted placement and consumes actual inventory.
Its pillar helper also uses server-assisted movement after it consumes a placed block.
These approved helper actions remain visible in original call journals.
Door placement now uses a native Mineflayer action and consumes one item for both halves.

The default request asks for a modest wooden house with explicit floor, wall, roof, doorway, and interior criteria.
Gathering must add logs beyond the starting inventory.
Crafting, construction, entry, and exit use prerequisite-ordered leaves.
Each leaf receives a fresh native actor session.
The final gate requires independent world acceptance and public evidence recomputation.

`--source-tree PATH` uses an inspected proposed tree as the initial specification.
Independent review still runs before execution.
Use a tree whose coordinates and assumptions fit the current fixture.
`--goal TEXT` changes the open-ended goal without changing the workflow.
`--timeout` limits the operator's observation period.
It does not impose a workflow tool-call ceiling.

## Retained evidence

The gate retains the original request, fresh initial observation, managed receipt, task tree, and active evidence index.
It also retains every actor session, original call journal, verifier read, and decision input.
Its source manifest identifies the code that ran.
The wrapper stops only the Minecraft and MCP processes that it owns.
Chat remains independently managed.

## Controlled refinement evidence

The real gate passed in `.runtime/alpha/20261006-a11-refine-04/`.
Its input was an operator-authored oversized request, based on the exact shell cells from a real preliminary plan.
It was input to actual model review and refinement, rather than a fixed model response.
The reviewer selected `house_delivery` for refinement.
The planner converted it to a story with nine dependent leaves.
The runtime preserved every selected acceptance condition and the unrelated goal specification.
A second independent review accepted the refined tree.
The plan-only run dispatched no actor.

`controlled-refinement-review.json` under `.runtime/alpha/a11/` recomputes those scope facts from the immutable snapshots.
The leaves cover gathering, crafting, floor, walls, roof, door, interior clearance, outside access, and final inspection.
The separate house execution trial adds the requested entry and exit outcomes through reviewed revision.

The current-source repeat in `.runtime/alpha/a11-plan-current-02/` performed three
read-only surveys, rejected malformed tool markup, and then retained a planned
revision. Its final state write hit a Windows file-lock permission error; the
raw planner, survey, and error records remain preserved. The earlier accepted
refinement and this run together cover both the positive refinement path and the
strict malformed-output path.

## Preserved preliminary failures

| Attempt | Observed result | Repair |
| --- | --- | --- |
| `20261006-a11-plan-01` | Block conditions attempted to use a volume and count; later refinement changed the selected criteria | Exact position groups; separate revision and refinement outcomes |
| `20261006-a11-plan-02` | Non-root nodes had no matching parent | Explicit hierarchy contract; invalid proposals return to scoped repair |
| `20261006-a11-plan-03` | Review exposed consumed inventory and simultaneous entry/exit criteria | Accepted references to original independent leaf evidence |
| `a11h5` | Three leaves passed. Wall placement filled the room; an aborted dig stopped the actor and cancelled its parent | Ancestor specifications and earlier work summaries reach each child; building guidance protects occupied cells; child stops have a separate linked signal |
| `a11h6` | Review requested a terrain grid before any actor ran | Read-only survey of proposed physical acceptance operands before review; retain original cells and pass them to replanning |
| `a11h7` | Gathering and crafting passed; a mining error stopped one actor, then malformed replanning output exceeded the repair-input budget | Local stop isolation verified live; compact schema-default fields and retain measured inputs before budget rejection |
| `a11h8` | Gathering and crafting passed. A region criterion confused remaining logs with collected logs; preparation repeatedly tried to mine non-solid vegetation | Preserve the original inventory baseline and verifier scope; retain solidity; include the deployed tool's plant constraint in its effective description |
| `a11h9` | Gathering passed with 55 logs. Crafting produced 54 planks and three doors, but its native final JSON omitted `handoff` and failed the parent | Retain the invalid candidate and its world journal; route native output validation failure to parent replanning after cleanup |
| `a11h10` | Review invented wrong plank and door ratios and a requirement to keep consumed logs; planning failed before actor dispatch | Supply original recipe lookups, the initial inventory baseline, and resolved prerequisite edges to review |
| `a11h12` | Planning requested another terrain layer before actor dispatch | Add model-selected read-only survey queries; preserve original criteria and references when a prerequisite becomes a story |
| `a11h13` | Six leaves passed. A missing outside support cell was rejected and repaired. Door installation exposed lower-only server-assisted placement | Use native one-item door placement; test both halves and opening/closing on the real server |
| `a11h14` | The continuation input was unavailable during harness startup; no actor ran | Finish and validate the operator input before the next unused evidence directory |

The third process was interrupted before actor dispatch while its old contract was superseded.
Its original partial receipt and separate operator interruption record remain intact.
All owned service ports were free after that interruption.
This preliminary interruption does not establish safe recovery of world actions.

The fifth trial retained the accepted gathering, crafting, and floor receipts.
Its original journal records the aborted dig as uncertain, without a paired result.
That command was not automatically retried.
The parent cancellation exposed a shared stop-signal defect.
The repair gives each actor a local signal that also observes external parent cancellation.
The parent can then retain a failed attempt and obtain fresh observations after actor cleanup.
This boundary applies within one process. It does not establish A12 restart recovery.

The earlier deployed mining tool rejected grass and tall grass because these blocks have no solid collision box.
Mineflayer's installed digging implementation supports these plants.
The related body repair in `pm-minecraft` removes the collision-box restriction and keeps an explicit changed-to-air result.
Normal digging and its resulting block observation establish vegetation clearance.
Local builder guidance now describes this repaired behavior.
`dev/tools/compile_minecraft_skill.py` appends local specialist guidance after the unchanged upstream resources.
The resource manifest hashes upstream content and local guidance separately.
The eighth process retained a cached actor module during source edits.
The operator stopped it before the next fresh-source trial.
Its original partial receipt still says `running`; a separate interruption record identifies the stop and free owned ports.
That partial receipt does not establish project completion or restart recovery.
The eleventh trial verified gathering and crafting, then exposed the same plant rejection during preparation.
It was stopped for the body rebuild. Its original calls and separate interruption record remain available.
The thirteenth trial retained six independently accepted leaves before the door repair.
Its normal replanning preserved gathering and crafting while changing only the preparation instruction.
One tool-markup proposal failed the JSON contract. A later actual model response repaired it.
The interrupted trial is separate from A12 automatic recovery evidence.

The door primitive passed in `.runtime/alpha/a11-door-01/` without a world reset or item grant.
Thirteen original calls establish one-item placement, both block cells, opening, closing, and ordinary removal.
Captured failed and successful door reads establish the exact two-cell acceptance boundary.
This primitive test does not establish companion behavior or a completed house.
The related Minecraft commits are `d6df968` for vegetation digging and `432d9d5` for native door placement.

## Live house evidence

The real project run in `.runtime/alpha/a11h16/` accepted `gather_logs`,
`craft_materials`, `prepare_site`, `build_floor`, `build_walls`, `build_roof`,
`install_door`, `clear_interior`, and `exit_access`, each with its own actor
journal and independent verifier certificate. The later `enter_house` actor
returned invalid native completion output after the body had reached health 1/20;
the run therefore remains a failed project receipt and does not overclaim entry.
No world reset or item grant occurred.

The fresh continuation in `.runtime/alpha/a11h18/` independently read the exact
house cells. Its actor report claimed completion but set `accepted: false`, so the
verifier rejected that claim. The navigation actor then repeated movement without
opening the door; the operator stopped it before further mutation. The preserved
interruption record is `operator-interruption.json` in that directory.

The independent read-only inspection in `.runtime/alpha/a11-house-inspection-04/`
is the final world check. It made 85 typed calls, inspected 83 cells, and passed
all 54 birch-plank, two birch-door, ten air, 17 support, and alive conditions.
It dispatched no actor and changed no block. This establishes the completed
house structure while keeping the unsuccessful entry attempt visible.

These A11 records establish durable task planning and verification, not a
self-crafted iron pickaxe, sustained survival, visual interpretation, or the
integrated alpha pass owned by later milestones.

## Go deeper

Read [A10 verification](minecraft-actor-verification.md) for the prior sustained wood leaf.
Read [manual milestones](../decisions/alpha-plan/verification.md) for A13's separate house-and-pickaxe gate.
