# Minecraft builder resource

Compiled from minecraft-survival-coach. Shared and specialist resources preserve the upstream sections.

## 24. Build a usable first house

A first house needs an enclosure, an entrance, room for work, and a way to leave.
The first house can be small.
Its usefulness matters before roof detail or a second floor.

Choose a site with these properties:

- Stable ground.
- Enough room for the planned floor.
- A clear route to the entrance.
- Nearby useful resources.
- No active liquid flow through the floor.
- No required deep drop beside the doorway.
- Space for a chest, a crafting table, and a furnace.

If the ground is uneven, use a smaller footprint or a controlled foundation.
Do not flatten a large hillside before you have basic shelter.
Do not build across an unknown cave opening.

### Construction order

1. Mark the floor bounds and height.
2. Reserve the entrance cells.
3. Count the essential blocks.
4. Build the floor or foundation.
5. Build the walls in reachable sections.
6. Build the roof without filling the interior.
7. Place the entrance block through the supported method.
8. Place essential workstations and storage.
9. Add light.
10. Walk through the entrance in both directions.
11. Inspect missing cells or obstructions.
12. Save the home location.

If resources run short, finish the usable enclosure before decoration.
If the player becomes enclosed, open the planned exit safely.
Do not fill the entire building volume with solid blocks.

### A complete small-house example

The following coordinates are relative to a selected site.
They are not coordinates for the current world.

Let `F` be the foundation height.
Use X=0 through X=4 and Z=0 through Z=6 for the outer footprint.
The floor contains 35 cells.
The outer wall ring contains 20 cells per layer.
The interior contains 3 by 5 cells.

| Part | Relative coordinates |
|---|---|
| Floor | X=0..4, Y=F, Z=0..6 |
| Lower north wall | X=0..4, Y=F+1, Z=0 |
| Lower south wall | X=0..4, Y=F+1, Z=6 |
| Lower west wall | X=0, Y=F+1, Z=1..5 |
| Lower east wall, first part | X=4, Y=F+1, Z=1..2 |
| Lower east wall, second part | X=4, Y=F+1, Z=4..5 |
| Entrance lower cell | X=4, Y=F+1, Z=3 |
| Entrance upper cell | X=4, Y=F+2, Z=3 |
| Upper walls | The same wall ring at Y=F+2, except the entrance |
| Roof | X=0..4, Y=F+3, Z=0..6 |
| Player feet inside | Near Y=F+1 |

The two wall layers need 38 full blocks after the two-cell entrance opening.
A full floor and full roof need 35 blocks each.
The basic shell therefore needs 108 full blocks before furniture and entrance material.
Existing suitable ground can reduce the new floor material.

Use these numbers to plan a shell, not to demand one material everywhere.
Stone, planks, and other suitable blocks can share the work.
Keep flammable material away from exposed fire and lava.

Before a placement call, translate each relative coordinate into a world coordinate.
Check that the region lies within reach or supported automatic movement.
If a section is out of reach, build it from a new safe position.
Do not change the whole plan after one out-of-range cell.

### Doors, gates, and entrance checks

Read the current placement method for the entrance item.
Place multi-block items through the supported survival action.
Inspect the required cells after placement.
Open the entrance.
Walk through it.
Close it from a usable side.
Check that the closure fits the shelter purpose.

If a doorway fails, inspect the floor step, body clearance, door state, and nearby roof.
If a gate opens but movement fails, inspect clearance before another attempt.
If the opening and landing are visibly safe, a short raw pulse is an available alternative.
Do not permanently remove the shelter entrance merely to avoid an unexplained movement result.

### Decoration after function

After the house works, add a small improvement with available surplus.
Examples include windows, a porch, roof trim, or a clear path.
Keep the entrance clearance intact.
Keep the return supplies intact.
Do not turn the first house into an unlimited construction project.


## 25. Use storage to preserve progress

A chest converts carried resources into a reserve at a known place.
An inaccessible or lost chest provides no usable reserve.

After chest placement:

1. Inspect its location.
2. Check that the chest interaction works.
3. Deposit a small intended amount.
4. Read the transfer result.
5. Check the remaining inventory.
6. Save the chest coordinates and purpose.

Use separate mental groups for expedition supplies and stored surplus.
Do not deposit the required pickaxe before departure.
Do not withdraw all food for a short walk.
Do not count a chest deposit as a deletion.

After a valuable trip, store the surplus first.
Then restock the next trip.
If the user asks for diamonds, preserve the first collected diamonds before an optional extension.

If a transfer fails, read the actual count and reason.
Do not assume zero transfer from a partial result.
Do not repeat a deposit for the original count without checking the remaining inventory.
# Observed construction failures

Inspect the exact cell before mining it. Skip air cells instead of issuing a dig against them.
The repaired body can dig grass and tall_grass through the normal mining tool.
is_solid describes collision bounds. These plants have empty bounds and remain valid digging targets.
Verify the resulting air cell after clearing vegetation. Keep the solid support below future floor cells.

Before a bulk placement, stand outside its target cells on solid ground.
Do not place a floor through your feet or fill a box around your body.
Use the live shape schema to distinguish a wall perimeter from a filled volume.
When a wall needs a doorway, build only its accepted wall cells with separate thin fills or individual placements.
Do not fill the room and then mine it empty.
Read the ancestor criteria before placing a workbench or choosing an exit route.
After terrain removal, observe your actual feet height before placing the replacement.

The build tool places doors through one native inventory action.
Request the lower door cell. One item creates its lower and upper halves.
Inspect both halves after placement. The placed list includes the automatic upper cell.
Use the block interaction tool to open or close a door.
The item-use tool does not place held blocks.
