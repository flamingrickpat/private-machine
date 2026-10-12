# Minecraft tool repairs from A13

## Summary

Earlier real runs exposed recipe-variant, HTTP deadline, walking-budget, and player-air defects.
The protocol probes retained actual outcomes and ordinary native supplies.
They establish tool boundaries separately from native acceptance.

Parent: [A13 verification](../index.md).

These records describe earlier source revisions. Pending statements apply to those named runs.
The parent page records current acceptance.

## Real tool repairs

Run `a13-adaptation-02` also exposed incorrect recipe-search evidence in `pm-minecraft`.
The search reported oak-only ingredients despite available birch planks.
The actual crafting tool later used those birch planks successfully.
The body now examines all recipe variants and chooses an available variant or the smallest ingredient deficit.
It applies the availability filter without reversing the returned boolean.
Three by three recipes also require a table within the crafting tool's four-block range.
The source and compiled body changes remain uncommitted while A13 acceptance continues.

`.runtime/alpha/a13/minecraft-recipe-before/` retains the original body source and compiled file.
`.runtime/alpha/a13-recipe-01/receipt.json` records seven passing real tool checks after the repair.
The check selected birch ingredients, exercised both filters, checked table presence, and crafted the advertised stick and pickaxe recipes.
Its ingredients came from the failed native run's ordinary harvesting and crafting.
It granted no items and did not reset the world.
The saved command was `uv run --locked python .runtime/alpha/a13/recipe_live_check.py`.

Run `a13-adaptation-05` exposed a separate body HTTP timeout during a broad block search.
The repaired block scan allows 30 seconds, while small status reads keep their existing timeout.
`.runtime/alpha/a13-search-01/receipt.json` records passing real wildcard and log searches after that repair.
The wildcard response took 2.422 seconds and retained its actual bounds and match count.
This engineering check used the retained world without model inference, resets, or item grants.
This is tool engineering evidence; it does not establish autonomous resource adaptation or M3.

Run `a13-adaptation-06` exposed separate full budgets for walking phases.
Path search, partial-route search, movement, replanning, and surface approaches now share one deadline.
The actor also receives clearer evidence-sequence errors and a recipe procedure for converting owned logs into available planks.
These changes preserve the guards and require actual fresh reads and inventory effects.
`a13-walk-01` repeats the former target through production MCP and records three passing checks.
The previous run had already moved close to that target, so this quick repeat alone does not prove the timeout repair.
`a13-walk-02` retains a rejected unauthenticated body probe; it establishes no walking result.
`a13-walk-03` uses the approved survival API through MCP with a one-second engineering budget.
It returned a normal timeout in 1.164 seconds and retained fresh player state.
Its aggregate failed because the later pose read occurred during a jump.
`verification-correction.json` compares the captured result with the immediate production state and retains the later airborne pose separately.
These are engineering checks without native model acceptance, resets, or item grants.

Run `a13-adaptation-07` exposed incorrect player air in the installed Mineflayer 4.38.0 plugin.
Its global oxygen field accepts other entities' air metadata.
`air-server-audit-01.json` records full player air, full health, and ground support while MCP reported depleted air.
The body now reads the player's own air metadata and converts it to the documented bubble scale.
`.runtime/alpha/a13/minecraft-air-before/` preserves the original body and installed plugin source.
Fresh water and lava-edge checks must cover this repaired body before delivery.
`a13-air-01` passed air isolation but failed a cleanup check before the client's death animation ended.
`a13-air-02` passed all four real checks, including removal of the owned tagged bat.
The server reported full player air while the bat had low air; MCP retained 20 player bubbles.
Both checks preserved player health and inventory, granted no items, and used no world reset.
Their commands record the temporary NPC fixture and its cleanup.

Run `a13-adaptation-04` exposed the same first-variant defect in failed craft costs.
With two birch planks and four sticks, the failure reported three missing oak planks.
The body now reports the failed variant with the smallest actual ingredient deficit.
`.runtime/alpha/a13-recipe-02/receipt.json` records five passing real checks for that repair.
It reported one missing birch plank, preserved inventory on failure, and crafted after adding planks.
That check also used retained ordinary native supplies without grants or a reset.

The saved stop procedure now restores a detached control snapshot without applying checkpoint restart recovery.
`stop_live_case.py` uses `Ghost._restore(read_checkpoint(path), None)` before the normal supervisor termination API.
The failed run's `operator-stop-correction.json` records the earlier loader mistake.

Run `a13-m3-03` exposed a diagnosis binding error after ordinary wood gathering and crafting.
The client requested `minecraft_actor_minecraft_craft_item`, while dispatch uses `minecraft_craft_item`.
The gateway accepted the diagnosis but rejected the unchanged internal method as paused.
The gateway now removes the client prefix and validates the result against its admitted tool names.
Unknown names and excluded tools cannot obtain a permit.
`a13-prefixed-diagnosis.json` preserves the original journal prefix, diagnosis, and admitted tool schemas.
Its regression establishes the corrected binding without changing the original failed receipt.
`focused-resumed-06.log` records 88 passing selected tests in 4.615 seconds.

## Supported staircase descent

M3-33 called `minecraft_staircase_down` with depth eight from `(29.4,29,1.56)`.
The original helper cleared notches and used timed forward controls without checking the landing floor.
It reported eight steps and 22 dug blocks, but its end position was `(38.7,0,1.5)`.
The next observation recorded the same body's first death and an empty inventory.
The server log recorded a fall death.
This receipt establishes failure; it does not establish eight safe steps or successful material consumption.

The body now checks the loaded solid floor below each proposed notch before digging.
It stops at unknown support, empty support, unstable floor materials, and liquid hazards.
After digging, it checks dry body clearance and keeps the floor intact.
The normal pathfinder targets one exact landing with its existing one-block drop limit and a five-second movement budget.
Only a grounded one-block descent in the same life increases `depth_achieved`.
Cancellation, changed landing, failed movement, or death ends the action and clears its controls.

The MCP result reports partial or hazardous descent with `ok=false` and an explicit reason.
Positive dug-block and descent quantities remain visible as partial effects.
Body session and death counters distinguish a completed step from an action that crossed a death.
Older persisted results can omit those added lifecycle fields.
Captured geometry and projected boundary checks cover unsupported floors, the actual excessive drop, and the actual death-counter change.
Those checks do not establish a new safe live descent.
M3 continuation must use the retained world and original empty-iron baseline.

## Harvesting prerequisites and occupied body space

M3-34 accepted the revised respawn reconciliation through verifier 111.
This fifth certificate does not establish native healing.
The next actor rebuilt ordinary wood stock but repeatedly tried to mine exposed iron with a stick.
Its block search reported false harvestability.
The exact inspector returned empty tools, null harvesting eligibility, empty drops, and null hardness.
Those omitted fields failed the inspector contract.

The body now supplies tool names sorted by actual breaking time, current held-tool harvestability, registry drops, and hardness.
The MCP inspector publishes those typed facts.
After two wrong-tool failures, the gateway requires a fresh true harvestability read for each requested target.
Changing coordinates or writing another diagnosis cannot override false or missing eligibility.
A fresh block search can also provide exact-target eligibility.
Reads must follow the latest wrong-tool failure and belong to the current actor source.
Other prerequisite actions remain available, subject to the existing recovery guards.
Historical tool certificates remain valid evidence of earlier work; they do not prove current ownership after death.

The same actor dug below its body and placed birch planks inside its feet cell.
The server recorded `tdd_player suffocated in a wall`.
Its final observation reported one death and an empty inventory.
The body now refuses mining below every horizontal column touched by the player's 0.6-block width.
The MCP build path rejects an entire request that overlaps the standing 1.8-block body before any placement or inventory consumption.
The check includes a door's upper half.
A floor ending at the feet and construction from a supported side remain permitted.
Crouching uses the conservative standing-height bound.
These guards use fresh production body positions rather than an old journal pose.

`a13-harvesting-body-space.json` retains the original reads, false-tool calls, placement, death observation, and source digests.
Minecraft's `a13-body-space.json` retains the actual pose and affected cells.
Captured and projected boundary tests are engineering checks; they do not establish repaired native behavior.
No inventory, world, or accepted specification was restored to conceal either death.
M3 remains pending and must continue from M3-34 in the same retained world.

## Equipment no-progress after M3-36

M3-36's native planner revised the respawn reconciliation for `(-8,119,10)` and preserved the original four certificates.
Verifier 117 accepted that revised leaf.
The next actor's real inspector published iron harvesting metadata.
The support guard rejected direct digging beneath the player, and ordinary movement enabled adjacent soil gains.
The same actor then repeated successful equipment calls that held the same two dirt items.
The operator stopped the run with those gains, full health, and no recorded death retained.

[`runtime.minecraft_outcomes.classify`][] now identifies unchanged equipment through item identity and durability.
An unchanged successful acknowledgement records `equipment_unchanged`, with no useful progress.
Two such calls pause the exact equipment method.
A fresh diagnosis cannot make an unchanged item into a missing pickaxe.
An actual equipment change records a quantity but cannot prove harvesting or leaf completion.
`unsafe_support` and `target_changed` retain their own cause even when harvesting is false.

The Minecraft equip result now publishes fresh `can_harvest_with_held` after the action.
False eligibility returns `unharvestable`; absent eligibility returns `harvestability_unknown`.
The actual held item remains visible.
Its `best_possible_tool` craft hint comes from registry tool metadata rather than the selected owned dirt item.
The fastest theoretical tool need not be craftable now; current recipes and eligibility govern the chosen prerequisite.

Recent repeated outcomes can reference a method's last result plus a `changes` object.
Apply those exact replacements to reconstruct each original time, sequence, source, reason, and quantity.
The selector preserves the replacements when it remaps method indexes.
Unusual older shapes with missing fields remain explicit.
`a13-equipment-no-progress.json` retains the actual no-op calls, support refusals, adjacent gains, and original journal digest.
All 147 selected checks passed in 5.100 seconds; eleven Minecraft contract checks also passed.
The earlier selected failures were stale assertions about the old representation, while the complete decoded facts still matched.
These checks do not replace M3 acceptance.

## Interrupted smelting after M3-37

M3-37 rebuilt native wood and tools, mined three fresh raw iron, and cleared the specified floor.
Verifiers 119 and 120 accepted those two leaves, bringing the retained total to seven.
Actor 25 placed furnaces and requested three raw-iron smelts with one birch log.
The body waited without a limit for three outputs, although one log cannot finish that batch.
The HTTP body call timed out after 180 seconds; wrapper cleanup then stopped the owned services.
The receipt records uncertain furnace effects. It does not establish smelting acceptance.

Source 53 bounds the furnace wait at 90 seconds and stops earlier after observed fuel exhaustion.
The tool returns partial output and unused input and fuel to inventory.
It closes the furnace window even after a protocol error; this does not undo inserted items.
The next call records `furnace_before` and collects existing output as `recovered_output` before loading a new batch.
The public result keeps new output, recovered output, actual consumption, and net inventory changes separate.
One log or plank smelts 1.5 items. Three raw iron require two such fuel items.

The M3 provenance audit accepts recovered iron only with a matching uncertain native dispatch.
The furnace must have a successful native placement before that dispatch.
Observed remaining raw iron plus recovered ingots must equal the interrupted batch count.
The net ingot gain must equal explicit new and recovered output.
Existing village output and arbitrary container withdrawals cannot satisfy this boundary.
Returned raw input from a matched interrupted batch remains recovery; it does not increase the mined quantity.
`a13-smelt-timeout.json` retains the actual placements and uncertain dispatch with their original journal digest.
Projected furnace states test the boundary; fresh real observations remain necessary to establish recovery.
The saved real-world chunk independently confirms two raw iron and one ingot in the furnace at `(30,102,5)`.
Its burn time is zero, and its recipe counter records one raw-iron smelt.
M3 acceptance and fresh compatibility checks remain pending.

## Uncertain furnace context after M3-38

M3-38 preserved seven certificates and started another native smelting actor.
Its full reconciliation contained the unresolved smelt after 24 older recorded actions.
The native agent read the opening portion, observed no iron in inventory, and searched for replacement ore.
It never inspected the retained furnace window before several unsuccessful mining and navigation methods.
The operator stopped the run with full health and food, no recorded death, and no new iron.
The failed receipt and the retained furnace contents remain separate evidence.

[`runtime.minecraft_actor.uncertain_world_effects`][] selects unresolved mutations across all retained actor journals.
The opening reconciliation section and initial native prompt now publish their exact source, sequence, tool, operands, and time.
Successful earlier furnace placements supply historical candidate positions for fresh inspection.
Those positions do not establish current contents or authorize automatic replay.
The gateway inspects their current blocks; the native actor must inspect the window and choose the remaining batch.
Inventory absence cannot establish lost furnace input.
A later explicit collection reconciles the older uncertainty only when its slots account for the original batch.
Another actor with recorded calls alone cannot erase unresolved earlier effects.
`a13-uncertain-furnace-context.json` retains the actual reconciliation and both journals with their digests.
Its projected collection checks are engineering boundaries, not a live smelting pass.

## Furnace used as storage after M3-39

Source 55's native actor selected the retained furnace rather than replacement ore.
Its partial walk moved within interaction range.
An actual `minecraft_use_block` result opened the furnace and observed two raw iron, no fuel, and one ingot.
Those contents match the independent saved-world read.
Inventory still contained no iron; opening a window did not collect it.
The actor then selected chest withdrawal with the furnace position.
The body attempted an unsupported container open and raised `containerToOpen is neither a block nor an entity`.
The failed receipt records uncertainty, not a withdrawal or smelting pass.

Source 56 checks current storage type and the five-block distance before opening or transfer.
The body returns `not_storage_container`, `out_of_range`, or `target_changed` through the normal action result.
The MCP result retains that cause and directs furnace recovery to `minecraft_smelt_item`.
The tool contract now states its existing explicit-cell requirement instead of promising implicit open-window support.
Ordinary chest, barrel, and colored shulker-box targets remain permitted.
The native prompt explains remaining raw input in slot zero and enough fuel for that batch.
`a13-furnace-storage-target.json` retains the actual reads, window contents, wrong calls, and original journal digest.
Captured and projected checks do not establish repaired live storage refusal or completed M3.

## Live furnace inventory after M3-41

M3-41 selected `minecraft_smelt_item` with two raw iron and two birch logs.
The body collected the retained two raw iron and one ingot before its availability check.
That check read `bot.inventory.items()` while the furnace window remained open.
Mineflayer copies player slots back to its main inventory only when that window closes.
The stale count therefore returned `missing` despite the real recovery.
The result reports recovered output and zero new input consumption; later observations confirm the inventory gains.

Source 57 counts input and fuel in the live window's player inventory range.
It excludes furnace slots and checks availability after collection.
The original missing reply remains partial progress, with one recovered ingot and two recovered raw iron.
It cannot establish a new smelt or completion.
`a13-window-inventory.json` retains the real reply, subsequent observations, and the installed Mineflayer inventory implementation digest.
Nine body tests and 152 selected Python checks passed; fresh native M3 completion remains pending.
