# Minecraft navigator resource

Compiled from minecraft-survival-coach. Shared and specialist resources preserve the upstream sections.

## 10. Understand coordinates and body space

X and Z describe horizontal position.
Y describes height.
Positive X is east.
Positive Z is south.
The opposite directions decrease those coordinates.

The player position normally describes the feet.
A block coordinate describes one cell.
The player needs support below the feet and clearance around the body.
An empty feet cell alone does not prove a usable destination.

Check these three cells for a simple destination:

1. The floor below the feet.
2. The cell for the feet.
3. The cell for the head.

Special blocks have shapes that differ from full cubes.
Slabs, stairs, fences, gates, doors, and water need extra care.
Use inspection and observed movement for those blocks.

### Negative coordinates

Use the reported block position for block operations.
Do not truncate a negative position toward zero.
For example, X=-8.44 is in block column X=-9.
It is not in block column X=-8.

If you calculate a cell, use the floor operation for each coordinate.
If the tool already reports the cell, use that report.

### Floor, feet, and roof

For a floor at Y=69, feet on that floor are near Y=70.
A two-block interior has clear cells at Y=70 and Y=71.
A full roof can start at Y=72.

Check the doorway with the same geometry.
A decorative slab above the entrance can reduce clearance.
A floor step outside the entrance can raise the player into that obstruction.

Do not use a roof coordinate as a feet coordinate.
Do not use an ore coordinate as a standable destination without inspection.


## 11. Choose the navigation tool by the task

| Need | Preferred tool | Required attention |
|---|---|---|
| A nearby visible approach | `minecraft_walk_to_visible` | Actual final position and clearance |
| Ground travel toward a column | `minecraft_walk_to_surface` | The selected surface and its height |
| A known standable location | `minecraft_walk_to_exact` | Actual offset and tool tolerance |
| A route preview | `minecraft_find_path` | Reachable, blocked, or unknown |
| A small controlled adjustment | `minecraft_fine_control` | Pose before and after each pulse |
| One supported vertical rise | `minecraft_pillar_up` | Held block, support, and headroom |
| A descending excavation | `minecraft_staircase_down` | Depth achieved, hazards, and exit route |

Use the current schema for exact arguments.
Tool prefixes can differ between clients.
Names in this guide describe the underlying Minecraft tools.

Do not use raw movement for a long journey that a walker can perform.
Do not use repeated exact walks to compensate for an invalid target.
Do not assume a surface target is the ground beneath a tree canopy.
If the selected surface is unsuitable, choose a visible dry landing or another column.

### A path preview is evidence, not a guarantee

A reachable preview means the search found a route in its current model.
The world can change before movement finishes.
A blocked preview applies to the search area and movement rules.
An unknown preview means the search did not decide within its limit.

If the preview is unknown, consider a shorter target or another approach.
Do not label unknown as impossible.
If the preview is blocked, inspect the relevant geometry.
Do not assume the whole world lacks an exit.


## 12. Read every movement result

If the current tool supplies these fields, read them:

```text
ok
status
reason
message
requested
final_position
distance_remaining
target_offset
diagnostics
duration_ms
warnings
```

`status` describes the movement outcome.
`reason` names the broad problem.
`diagnostics.cause` gives a more specific cause.
The original exception can explain an interrupted or failed movement.
The target block fields describe the destination.
The feet and head block fields describe the final position.

`target_offset` is the requested target minus the final feet position.
Positive Y means the requested point remains above the player.
Negative Y means the requested point remains below the player.
X and Z use the same signed coordinate rule.

For example:

```json
{
  "status": "reached",
  "requested": {"x": 10, "y": 65, "z": 20},
  "final_position": {"x": 10, "y": 63, "z": 20},
  "target_offset": {"x": 0, "y": 2, "z": 0},
  "diagnostics": {"head_block": "water"}
}
```

This result does not complete a water escape.
The player remains two blocks below the target.
The head remains in water.

If the offset field is absent, subtract the final position from the target.
If the final position is absent, observe before a precise dependent action.

### Interpret causes without inventing obstacles

| Cause or result | Meaning | Next useful action |
|---|---|---|
| `goal_reached` | The movement reports arrival | Check the task completion conditions |
| `partial_goal_within_tolerance` | A partial destination counts as near enough | Check the offset and final surroundings |
| `outside_reached_distance` | The route ended too far away | Inspect the remaining route |
| `target_outside_search_distance` | The local target exceeds the search range | Select a nearer target or suitable long walk |
| `target_not_standable` | The destination fails a standing requirement | Inspect the feet, head, and floor cells |
| `no_standable_surface` | No suitable nearby surface was selected | Choose a different dry column |
| `search_no_path` | The search found no supported route | Inspect geometry or choose another approach |
| `search_timeout` | Initial planning exhausted its limit | Shorten the route or use intermediate targets |
| `replanning_timeout` | Planning during movement exhausted its limit | Observe the new position before another attempt |
| `movement_timeout` | Movement exhausted its limit | Compare actual progress before continuing |
| `movement_no_path` | Movement cannot continue through a supported route | Inspect the new obstruction or destination |
| `movement_error` | Another execution error occurred | Read the original error name and message |
| `stopped` | The action stopped before completion | Read the current state and interruption cause |
| `pillar_up_required` | A vertical rise remains | Check the final shaft position and required blocks |

These are examples from the tool family.
New versions can add or rename causes.
Read the live result instead of forcing it into an old label.

If `error_name` is `GoalChanged`, investigate the interrupted goal.
Do not call a goal change a stone wall.
If the target floor is missing, do not call the problem a timeout.
If the target head cell is solid, clear a safe route or choose another destination.

### Do not repeat a zero-movement success

If a walk reports success without useful movement, check the distance tolerance.
For close work, choose a clear standable cell that satisfies the task.
Use a small raw movement pulse only with clear local geometry.
Check the resulting pose after the pulse.
Do not issue ten identical walks to the same tolerated destination.


## 13. Aim before raw movement

The camera and the body have related but different jobs.
Turning changes the view.
Walking changes the position.
Looking up does not prove upward movement.
Looking at an ore does not place the player beside that ore.

Before a raw movement pulse:

1. Identify the intended direction from coordinates.
2. Check the next body space and floor.
3. Aim with the current tool schema.
4. Read the reported view direction.
5. Send one short movement pulse.
6. Read the pose change.

If the pose changes in the wrong direction, stop that movement method.
Reorient from the new coordinates.
Do not multiply an unproven pulse into a long loop.

The current control tool can use fixed pulse lengths.
Do not invent a duration argument.
Do not copy the older `controls` wrapper without checking the live schema.
Use only the documented control keys.

If a jump pulse ends, do not assume the jump remains held.
If a swim pulse ends, do not assume the player remains at that height.
Check the next state before the next pulse.


## 26. Scout with a return plan

Exploration finds animals, trees, exposed ore, and better routes.
Exploration also consumes food, time, and attention.
A useful scouting trip answers a specific question.

Before scouting, name the target class.
Examples include a nearby food source, an exposed stone face, or sheep for a bed.
Choose a modest search area or a short sequence of visible landmarks.
Save the home location before departure.

At each new area:

1. Read the new position and nearby threats.
2. Search for the intended resource or landmark.
3. Save useful discoveries.
4. Compare remaining supplies with the return route.
5. Choose the next segment or return.

If the first area lacks sheep, inspect another reasonable area.
Do not repeat the same sheep search at the same position.
If the search reaches the trip limit, return with the useful discoveries.
Do not make the map larger forever because one resource remains absent.

If another useful resource lies directly on the safe route, collect a bounded amount.
Do not abandon the food task for a deep ore chase.
Do not enter every cave encountered during a sheep search.

### Search coverage

`not_found` means the tool found no match within its actual coverage and visibility rules.
It does not mean the resource is absent from the world.
Nearby block summaries do not prove a reachable or exposed target.

If a search fails, change the search coverage through ordinary movement or safe exposure.
Use the declared search tools for ore discovery.
Do not inspect arbitrary distant underground coordinates to bypass visibility limits.
Do not use world files, seeds, or hidden server state to locate resources.

