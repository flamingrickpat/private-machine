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
