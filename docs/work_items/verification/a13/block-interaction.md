# Block interaction evidence

## Summary

Block activation must distinguish a sent packet from an observed world effect.
The Minecraft body confirms door state changes and new container windows.
Original failures and independent server reads retain the evidence for this boundary.
Native M3 entry remains a separate acceptance check.

Parent: [actor controls](../../../subsystems/minecraft-actor-controls.md).
Related evidence: [scoped gameplay decisions](scoped-acceptance-evidence.md).

## Input and owning component

The `pm-minecraft` repository owns the Python MCP tools and Node Minecraft body.
Its public `minecraft_use_block` tool accepts one integer block-cell position.
The runtime checks the current block type and the five-block interaction range before activation.
`mcmcp.runtime.MinecraftRuntime.minecraft_use_block` returns the typed `mcmcp.models.UseBlockResult`.

[`runtime.minecraft_actor.ActorGateway.call`][] retains the original MCP result in the actor journal.
The companion does not convert an interaction label into a task certificate.
The selected leaf still needs its completion candidate and fresh independent verification.

## Observed activation

The Node body's `useBlock` closes the previous container window before sending a new activation.
This prevents old furnace or chest contents from appearing as the new target's window.
For doors, gates, and trapdoors, it observes the target's `open` property before and after activation.
The target must retain its block type, and the open state must change before a toggle label can succeed.

For containers and crafting tables, the body waits for a new window.
Other interaction kinds retain their existing packet-activation contract.
One ten-second confirmation wait bounds each toggle or window interaction.
The Python body client allows fifteen seconds to cover that wait and transport.

The public result supplies `open_before` and `open_after` when the target has this property.
An unconfirmed activation returns `ok=false`, `reason=activation_unconfirmed`, and no claimed action.
A door action returns no container window.

## State and collision observations

The public `minecraft_inspect_block` result retains the loaded block's state properties and collision boxes.
The Node body obtains these values from `getProperties()` and `shapes`.
Python transports the values without deriving orientation from the block name.
The managed workflow keeps both fields in its compact criterion observations.

Each collision box contains `[minX,minY,minZ,maxX,maxY,maxZ]` coordinates relative to the inspected cell.
Add the cell position to obtain world bounds.
An empty list means the observed block has no collision boxes.
A null value means the retained result supplied no geometry.
Older captured results therefore do not acquire invented geometry.
The legacy `is_solid` flag classifies non-empty collision bounds.
Partial blocks can return true; use their actual boxes to assess clearance.

An open property alone does not establish clearance in a travel direction.
The actor must account for the collision boxes and its own body dimensions.
These observations apply to all loaded block types, without task-specific repairs or acceptance changes.

## Cancellation and uncertain effects

Activation changes the live world or current interface.
Closing a window does not remove its stored contents or restore inventory.
Cancellation cannot undo a packet that the body already sent.
The [actor cancellation boundary](../../../subsystems/minecraft-actor-controls.md) waits for external work to finish before releasing ownership.

An unconfirmed result does not prove that the server will never apply a delayed effect.
The actor must obtain current evidence before another mutation.
Original before/after states and failures remain in the journal.

## Real failure and repaired boundary

M3-58 actor 43 submitted a roof candidate after eighteen Minecraft calls.
Fresh verification accepted both `roof_complete` and `house_build`, bringing the task to fourteen certificates.
The next actor reached the entrance approach but did not complete entry.
Its tool predicted `door_opened`; independent server reads then showed both door halves closed.
The reply also attached an old three-slot furnace window as a chest window at the door.
Later native interactions left the door open before the operator stopped the run.
Those partial effects remain preserved.

The first engineering probe incorrectly assumed a closed initial door and stopped before activation.
Its failed receipt remains under `.runtime/alpha/a13-block-activation59`.
The next probe used the actual initial door state.
It opened the furnace, toggled the door twice, and confirmed both halves with independent read-only server predicates.
Both toggles returned the door to its initial state through ordinary tool calls.
Position, inventory, and vitals remained unchanged, and the door replies contained no stale window.
All five independent protocol checks passed under `.runtime/alpha/a13-block-activation60`.

The captured source and digests remain in `pm-minecraft/body/test/fixtures/a13-block-activation.json`.
These protocol checks do not accept native entry or complete M3.

M3-59 confirmed activation but stalled at the doorway again.
Independent server reads identified an open north-facing door in the west wall.
The final native action mined the lower door block before cancellation completed.
The later read-only inspection observed air in both door cells.
This partial effect remains in the world; no operator restored the door.

The first geometry probe expected the door to remain and failed that assumption.
The next probe checked the observed air, floor, and existing furnace.
Furnace state properties matched an independent server predicate, including its false `lit` value and north orientation.
Air had no collision boxes, and the solid floor and furnace had full boxes.
All six read-only checks passed with unchanged position, inventory, and vitals.
Both original receipts and source digests remain in `test_data/captures/alpha/a13-block-geometry.json`.
These results verify observation transport; they do not establish the former door's collision geometry or native entry acceptance.
