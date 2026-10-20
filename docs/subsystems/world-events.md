# Observed world events during silence

## Summary

A periodic bus module reads structured Minecraft facts after user silence.
Changed facts enter durable memory before they reach main cognition.
The model can then choose activity within the configured action space.
The module itself never chooses a goal or mutates the world.
Stopped intervals produce no observations.

Parent: [cooperative survival](cooperative-survival.md).
Owner: [`workflows.minecraft_body.world_events.WorldEvents`][].
Assembly: [`workflows.companion.build_app`][].
Transport: [`workflows.minecraft_body.mcp_client.build_minecraft_tools`][].

## Activation and exact inputs

Real cognition enables the module by default.
Dummy cognition does not enable it.
`build_app` accepts `world_poll_seconds=120.0` and `world_quiet_seconds=120.0`, both measured in seconds.
`world_poll_seconds=None` disables this event source.
A poll period must be positive; a silence interval cannot be negative.

The endpoint is `minecraft.mcp_url` in the model configuration.
`MINECRAFT_MCP_URL` remains a test override.
Without a `minecraft` section, the module does not start.
Each poll calls only `minecraft_observe` with `include_image=false`.
The read shares the cross-process command lock with actor calls.
It never claims the body's mutation lease.
The transport has a 30-second request deadline; queued command-lock waits respect application cancellation.

[`workflows.minecraft_body.world_events.world_conditions`][] selects position, inventory, vitals, body life, nearby blocks, and coarse world time.
Game time uses 6,000-tick day phases.
Camera rotation and individual time ticks do not create another cycle.
The full original observation remains in the record's evidence metadata.
No image interpretation occurs.

## Commit and publication

A user input resets the module's local silence clock on delivery, including during a queued read.
Startup also starts a fresh silence clock.
Stopped time does not count as observed activity.
The first successful silent observation establishes the baseline and emits an event.
Later observations emit only when selected conditions change.
After a baseline exists, active game work permits change observations during conversation too.
The record's `user_silent` field distinguishes silence from active work.
`requested_at` records the request start; `executed_at` records the response time.

The memory store commits a `WorldEvent` record before publication on `minecraft.world`.
Its metadata contains `origin=minecraft.world`, `executed_at`, full `evidence`, `previous_conditions`, and `correlation_id`.
`self_origin=true` distinguishes this source from a user instruction.
The record remains an observed event rather than user speech or a private thought.

One pending record ID prevents a backlog while main cognition works.
A `gwt.updated` event clears it only when that accepted workspace contains the record ID.
A rejected cycle acknowledges each world source, including sources in a mixed user batch.
The next poll reobserves the world and can retry unchanged conditions with a new record ID.
Unchanged conditions cause no model call.

## State, cancellation, and failures

The module owns only its timer, last selected conditions, pending ID, and local silence clock.
Those values are local process state.
Accepted observations, cognition choices, and run receipts belong to their existing durable owners.
Restart does not replay a poll or fabricate stopped-time activity.

The local `world-events.jsonl` journal records committed events and unavailable reads under the runtime root.
Transport failures retain their error without producing a world fact.
The next timer poll can retry the read.
An unsuccessful structured response follows the same path.
Logic errors still stop the application.

Shutdown cancels a queued command-lock wait.
A response that arrives after module shutdown cannot enter memory.
An already dispatched read can finish within its transport deadline.
Actor effects remain governed by their own cancellation and uncertain-outcome rules.

## Inspect a real run

1. Open `world-events.jsonl` under the selected runtime root.
2. Read the committed memory record named by `knoxel_id`.
3. Follow that ID into the accepted workspace and its action records.
4. Follow any dispatch into the managed-run receipt and original actor journal.
5. Read current inventory or exact world cells before claiming useful progress.

The [A14 verification record](../work_items/verification/a14-cooperation.md) separates these boundaries from behavioral and human acceptance.
