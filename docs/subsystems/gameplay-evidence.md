# Gameplay history and cognition

## Summary

Original actor journals own gameplay facts.
A derived projection supplies failed methods, local search coverage, partial quantities, and active recovery.
Planning shares repeated preconditions through `method_states` and each method row's `state_index`.
Every method and original evidence reference remains; the actor's full projection stays on disk.
The project shares that history across specialist attempts.
One attempt boundary publishes meaningful progress to memory and cognition.
Individual tool calls do not trigger full cognitive cycles.

Parent: [expected failures](gameplay-recovery.md).
Related: [independent acceptance](minecraft-task-evidence.md), [consolidation](memory-consolidation.md), and [publication recovery](recovery-publication.md).

## Files and identity

| File | Owner and meaning |
| --- | --- |
| `scratch/actor-journal.jsonl` | The actor gateway. Original dispatches, results, diagnosis records, rejections, and uncertain external outcomes. |
| `scratch/gameplay-evidence.json` | An atomic derived projection with schema `pm.gameplay-evidence.v1`. The gateway rebuilds it from retained journals. |
| `inbox/reconciliation.json` | Fresh observation and retained mutation target checks after an interruption. |
| `planning/state.json` | The project. Stable tree, active certificates, attempts, and the required concrete leaf change. |
| `planning/assumptions/*.json` | Original assumption reads, checks, and the explicit changed-assumption outcome. |
| `outbox/world.yaml` | One meaningful body episode with its gameplay evidence. Consolidation reads this artifact. |

A journal sequence identifies a call within one actor attempt.
Its source path distinguishes identical sequences in separate attempts.
The gateway writes each original result before it updates the projection or exposes the result.
A crash during projection replacement retains the authoritative journal.
A replacement actor rebuilds cumulative history and obtains fresh state before mutation.

Earlier journals supply history even when the new actor executes another leaf.
Target reconciliation remains scoped to earlier attempts of the interrupted leaf.
It does not inspect every historical construction cell before each new leaf.
An older planning observation cannot replace a later body observation.

## Search memory

Each `minecraft_find_block` result retains these fields:

| Field | Meaning |
| --- | --- |
| `source`, `seq` | Original actor journal and result identity. |
| `observed_at` | Unix timestamp in seconds from the original receipt. |
| `world` | Configured connection reference; actual world identity remains unknown without deployment evidence. |
| `dimension` | The latest observed game dimension, or `unknown`. |
| `method` | `surface_exposed_loaded_cells`, matching the live tool's visibility rule. |
| `pattern`, `total_matches` | Actual query and reported match count. Missing counts remain unknown. |
| `coverage` | Original center, exact bounds, loaded chunks, scanned columns, and note. |
| `available`, `uncertainty` | Read availability and the unexamined buried or unloaded area. |

The live search reports exposed matches rather than all blocks in the world.
A local `not_found` result cannot establish universal ore absence.
A connection reference cannot prove world continuity after an operator reset.
Keep deployment seed, reset records, and task journals together for real comparisons.

The projection retains every method and search entry.
`life` retains the current public body session and observed death count.
`failed_recoveries` retains deaths with their original observation references and failed episodes.
These records stay separate from progress and completion evidence.
Its latest twenty outcomes provide immediate context with original journal references.
Full original results remain in their journals.
Repeated tool replies include compact paused methods instead of their complete inventory and outcome history.
Successful diagnosis replies identify the exact permitted action and the projection path.
The disk projection retains the complete evidence for inspection and context recovery.
Each meaningful result records `made_progress` separately from `unsuccessful` and `completion_verified`.
Positive quantities and inventory gains establish output; consumed stock alone does not.
Walking retains measured displacement as `quantities.moved_blocks`.
The `navigation` field pools ineffective walks across destination and walker changes.
Its position comes from the latest timestamped body or movement receipt.
This position evidence does not establish arrival, dry support, or goal acceptance.
Routine tool replies supply no current outcome when their journal sequence has no meaningful result.
Direct decisions use a percentage of the actual serving context, measured in tokens.
There is no character limit. See [direct project budgets](llm-context.md#direct-project-decisions-use-token-percentages).
[`workflows.minecraft_tasks.steps.compact_plan_context`][] references identical review text and tool contracts once per decision.
Different feedback and historical contracts retain their original wording.
Native actor execution retains pm-coder context discovery and automatic compaction.

The compact history uses schema `pm.gameplay-planning.v4`.
Earlier captured inputs can use `pm.gameplay-planning.v2` or `v3`.
`method_fields` names the columns in each `methods` row.
`outcome_fields` names the columns in that row's `last` outcome.
Resolve `state_index` through `method_states`, `source_index` through `source_paths`, and `kind_index` through `outcome_kinds`.
In v4, each state's `inventory_indexes` selects complete item facts from `inventory_items`.
Its `held_item_index` selects the held item from that table; null means no held item.
Identical stock is shared across different positions and tool states.
Changed counts, slots, and durability retain separate rows.
This sharing preserves all preconditions; original journals retain display labels and empty fields.
The `tool_index` column resolves through `tool_names`.
Earlier captured inputs can have a direct `tool` column; use the supplied column names.
The method's tool and arguments identify its last outcome unless `additional_fields` overrides them.
Merge those additional fields to reconstruct changed presentation limits or other original operands.
Search entries inherit shared `search_defaults`; their explicit values override those defaults.
In version three, `recent_outcomes` can contain an object with `method_index`.
That index cites the exact last outcome in the corresponding `methods` row.
Reconstruct it with the same tool, argument, kind, source, and additional-field tables.
Only complete field equality permits this reference.
An earlier outcome that differs from the method's latest result remains explicit.
Original journals and disk projections keep their full records.
Other provenance objects also use `source_index`; callable Minecraft `source` arguments keep their original names and values.
These tables retain every method, counter, quantity, inventory delta, search, diagnosis, and original identity.

The decision selector can also supply `method_state_fields` for ordered method-state rows.
Resolve those names before the inventory indexes.
Each search inherits `search_defaults`; explicit row values take precedence.
Its `coverage_index` selects a row in `search_coverages`.
Each coverage row contains a field-set index and ordered values.
Resolve that index through `search_coverage_field_sets`; a null index preserves the raw value.
When `search_coverage_vector_fields` exists, its axes name the center and minimum/maximum bound coordinates.
These coordinate arrays preserve exact numeric values and types.
They do not change coverage, availability, or observation time.
They remove repeated field names and repeated strings from the working context.

## Current server rules

The planner reads `minecraft_info` beside its fresh body observation.
`planning/server-rules/` retains the actual server response before review or another actor.
The planner and reviewer receive its `server_rules` text.
Each actor also reads this response into `inbox/server-rules.json` through its journaled gateway.

`pm-minecraft` obtains these rules through the read-only `babymode help` command.
The command exposes current regeneration and nutrition rules without changing player or world state.
An RCON error remains an external failure; it cannot supply assumed vanilla rules.
Older persisted info records can omit the added text field.

The tested deployment disables natural health regeneration and well-fed buffs.
Full food and sleep therefore cannot establish health recovery from fall damage.
The actor must obtain an ordinary healing effect and prove improved health through fresh observations.
The low-health threshold and recovery guards remain unchanged.
M3-30 independently rejected an `alive` condition described as health above 10.
The condition schema now supports `health` with an inclusive `minimum` in observed health points.
Health above 10 therefore requires `minimum=11`; `alive` still checks only health greater than zero.
Missing, nonnumeric, boolean, out-of-range, and wrong-dimension observations fail this predicate.
The original low-health observation and rejected native criterion remain in the captured fixture.

M3-32 used the concise correction successfully for a supported cave move and a horizon scan.
Its native diagnoses still assumed ordinary food could heal with natural regeneration disabled.
The operator stopped its repeated probes before the next source change; health remained five.
The initial native prompt now includes the actual current server rules and recovery facts directly.
Those facts remain available when execution history compacts.

For a current `natural regen off` report, low-health guidance distinguishes ordinary food, saturation, and nutrition from healing.
It names golden-apple regeneration as an ordinary survival option, with actual recipes and prerequisites still required.
The [official golden-apple guide](https://www.minecraft.net/en-us/article/golden-apple) describes its regeneration effect.
Fresh observed health must prove recovery; absorption cannot satisfy a health predicate.
The guidance cannot grant supplies, change server rules, or accept an uncompleted leaf.

## Bounded decision history

[`workflows.minecraft_tasks.steps.scoped_decision_context`][] selects evidence when a direct decision exceeds 110,000 characters.
The final input limit remains 120,000 characters.
The decision stores the full original payload in `unscoped-context.json`, with its digest and path in `context_archive`.
Actor requests receive the same selected history.
Their `inbox/gameplay-history.json` archive retains the complete cumulative history.
Original journals retain every call and its full result.

The selected history retains every paused method, the latest 32 methods by result time, and all twenty recent outcomes.
It preserves their exact operands, preconditions, counters, quantities, and journal references.
It retains all search coverage, recovery state, body identity, navigation state, and failed recovery records.
Only referenced inventory and method-state rows enter the selected tables.

Older methods supply counts, failure reasons, quantities, inventory gains and removals, and source-index-to-sequence lists.
Those totals sum each method's last result; they do not count every earlier action or establish mined-resource provenance.
The latest four diagnoses remain complete.
Older diagnoses retain source, time, and evidence sequences; their complete explanations remain in the archive.

An independently accepted verification retains its identity, checks, review, and original receipt.
Its old read list remains in that receipt, because it cannot establish current geometry.
An exactly unchanged specification can reference its node in `previous_tree`.
Failed verification keeps its complete selected reads.
These projections cannot modify specifications or issue acceptance certificates.

M3-29 exposed this boundary after 776 native calls.
Its planner input had 274,405 characters and failed before inference.
The captured regression reduces that actual input to approximately 112,400 characters and preserves both mined raw iron items.
This regression establishes context selection; the resumed real run must establish gameplay acceptance.

## Concise actor correction

M3-31 retained 41 rejected calls without sending those actions to the world.
Its refusals repeated all paused methods and enlarged the native context.
The native actor eventually supplied a diagnosis, then tried to eat an inedible dandelion.
The actual `no_food` result remains retained; nutrition did not heal the player.

A refusal now returns only the paused method that matches its exact tool and arguments.
Its `diagnosis_call` supplies existing fresh read sequences and the denied action's operands.
The actor must add its own explanation of the cause and safe method change.
This suggestion does not grant permission or establish that the proposed action is safe.
The diagnosis validator still rejects stale reads, unchanged paused preconditions, and unsupported recovery destinations.

Each journal append also updates `scratch/decision-evidence.json` with current body, safety, inventory, and paused-method facts.
The full projection remains in `scratch/gameplay-evidence.json`.
Successful tool replies identify both paths and return only matching paused-method metadata.
These files preserve the commit order: original journal, complete projection, then current decision projection.
Recovery thresholds, fresh geometry checks, and independent acceptance remain unchanged.

## Death and missing inventory

M3-33 understood the current healing rules, then died during a staircase action.
The tool reported eight steps despite a change from Y29 to Y0.
The following body observation reported a death-counter increase and an empty inventory.
The native actor returned `replan`; respawn did not establish healing or successful recovery.
Its original text incorrectly described all 252 missing items as blocks consumed.

Death records now list `inventory_missing_after_death` and explain this evidence limit.
[`runtime.minecraft_outcomes.missing_inventory`][] compares observed item counts without assigning a consumption cause.
Missing quantities do not prove consumption, placement, destruction, a drop location, or successful retrieval.
Older method totals use `removed` for negative inventory deltas.
The original action delta and rejected native interpretation remain intact.

The staircase result now includes the body session and death counters before and after the action.
An observed action death immediately creates failed recovery and blocks further mutations before the next observation.
[`runtime.minecraft_outcomes.action_death`][] requires a body identity and an increasing integer death counter.
Fresh respawn observations still cannot authorize another mutation in that actor epoch.
The next native plan must reconcile current tools, supplies, lost items, and preserved world work.
The [tool repair history](../work_items/verification/a13/history/tool-repairs.md) describes the new descent checks.

## Meaningful progress publication

After a completed actor attempt, the project writes its world episode before it calls the private `managed_milestone` operation.
[`pm.runs.supervisor.ManagedRunSupervisor.milestone`][] accepts a worker epoch, attempt key, summary, and original journal reference.
It rejects a stale worker and reuses an existing milestone key.
The durable managed receipt owns the event before publication.
The callback runs outside the supervisor lock.

The event uses `gameplay_progress`, the original event ID, and the attempt ID as `milestone_key`.
Its `detail` describes observed progress or failure.
Its `evidence` identifies the original journal.
The [existing bus path](../workflows/exchange.md) supplies this event to cognition.
No per-call appraisal or emotional update runs inside the gateway.

[`runtime.consolidation.Consolidator`][] retains gameplay evidence inside `BodyEpisodeRecord.evidence`.
The episode text names the latest failure and any active recovery causes.
Artifact hashes and original paths preserve its provenance.
Repeated consolidation keeps the existing memory keys.
Gameplay summaries remain attributed workflow evidence; they do not replace independent acceptance.

## Inspection procedure

1. Open the managed run's artifact directory.
2. Read `planning/state.json` and select the actor attempt.
3. Read that attempt's original journal and gameplay projection.
4. Compare the failed operands, diagnosis read sequences, and permitted next action.
5. Read each negative search's exact coverage and observation time.
6. Inspect recovery vitals and the feet, head, and floor reads.
7. Read the managed receipt's `gameplay_progress` event and original event ID.
8. Inspect the consolidated body episode and its artifact hashes.
9. Read independent verifier receipts before interpreting a completion claim.

## Traps

A partial quantity preserves work but cannot prove the whole goal.
A diagnosis records the model's interpretation; original observations establish external facts.
A recovery permit cannot prove that the selected action actually improves safety.
An empty inventory query and an unsuccessful ore search describe different evidence.
Deferred work remains unfinished.
An uncertain external mutation remains uncertain after a service restart.

## Go deeper

Read [verification](../work_items/verification/a13/index.md) for captured failures and live acceptance status.

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

## Historical survey selection after M3-35

M3-35 stopped before a fresh actor attempt when planner 113 reached 124,188 input characters.
Its requested world survey still belonged to tree revision zero; the retained tree was revision 24.
The scope selector now keeps that older survey's exact queries, recipe patterns, identity, original path, digest, and result counts.
Its old output bodies remain in the original receipt.
They cannot establish current landing geometry or current ingredient availability.
A requested survey at the current tree revision remains fully explicit.
Failed independent verification remains fully explicit too.

Homogeneous search records now use `search_fields` and ordered rows.
This representation retains every source, sequence, time, pattern, dimension, coverage, match count, and availability value.
Unusual older record shapes remain unchanged.
The captured overflow fits below 115,000 characters after selection.
The hard 120,000-character input limit remains unchanged.
`a13-historical-survey-budget.json` retains both original inputs and their source digests.
All 145 selected checks passed in 5.135 seconds.
This proves context selection and boundary compatibility; it does not prove M3 gameplay.

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
