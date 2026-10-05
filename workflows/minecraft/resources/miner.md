# Minecraft miner resource

Compiled from minecraft-survival-coach. Shared and specialist resources preserve the upstream sections.

## 30. Make a staircase that you can reverse

A mining route has two directions.
The descent is incomplete without a practical return.
A hole that lowers Y is not automatically a staircase.

Before the first descent:

1. Choose a safe entrance.
2. Save its coordinates and dimension.
3. Choose a direction that avoids important structures.
4. Check the initial cells and hazards.
5. Check tools, light, blocks, and food.
6. Choose a short first segment.

Use `minecraft_staircase_down` with the live schema.
Prefer a few steps for an unfamiliar area.
Read depth achieved, end position, blocks dug, hazards, and held-item warnings.
Do not assume the requested depth equals the actual depth.

After each segment:

1. Observe the new position and surroundings.
2. Check that the last landing remains reachable.
3. If the segment needs light, add light through a supported action.
4. Save a useful turn or landing.
5. Decide whether supplies support another segment.

Periodically travel back over a short completed section.
This checks the return geometry before the mine becomes deep.
Do not wait until the pickaxe breaks to discover that the exit is unusable.

### If the staircase stops

If it reports water or lava, stop the excavation at that boundary.
If it reports gravel, assess falling material from a safe position.
If it reports a cave, inspect the opening and landing.
If it reports no progress, do not call a larger depth at the same position.

A cave report can mean existing empty space.
The tool does not necessarily need to mine that space.
Use a known landing or a safe constructed connection instead.

If the direction differs from expectation, use the actual coordinate change.
Reorient before another segment.
Do not continue a long coordinate sequence based on the old heading.

### Manual geometry

Use manual excavation only with a clear local route plan.
Keep a floor beneath the current player position.
Clear the next head space and feet space from a safe side.
Preserve a return step with adequate headroom.
Probe each new excavation boundary.

Do not decrement Y repeatedly at the same X and Z beneath the player.
Do not remove the staircase landing to reach ore below it.
If a landing needs repair, repair it before deeper travel.


## 31. Explore a cave in small sections

A cave can offer exposed ore and a dangerous route at the same time.
The visible ore does not remove the route problem.

At a cave entrance:

1. Mark the entrance.
2. Observe the first chamber.
3. Identify a safe landing and return path.
4. Check hostile entities and liquid.
5. Add useful light.
6. Enter only the section that the current supplies support.

Keep the return path identifiable.
Use coordinates and landmarks together.
If a consistent torch convention helps, use it throughout that route.
Do not rely on torches alone after a complicated turn or branch.

At a branch, select one branch for the current task.
Save the other branch for later.
Do not alternate between several unexplored branches without a route record.

If a valuable ore lies across an unsafe gap, save the ore location.
Construct a safe approach on a suitable trip.
Do not treat ore value as evidence that the jump is safe.

If the return route becomes uncertain, stop exploration.
Return to the last known landmark or establish a new clear connection.
Keep route recovery active until that connection works.


## 32. Progress from iron to diamonds

An iron pickaxe is a useful dependency for diamond collection under ordinary mechanics.
The current server can change recipes or requirements.
Use its harvest information and recipes for the actual task.

For iron:

1. Choose the next useful iron item.
2. Read the recipe and required ingot count.
3. Collect a bounded amount of accessible raw iron.
4. Return or reach a safe furnace.
5. Smelt the required amount.
6. Craft the selected item.
7. Check that the item exists.
8. If the next task requires that item equipped, use the supported equipment method.

Do not assume crafted armor is worn armor.
If an equipment interface is available, use it.
If equipment slots are not exposed, report that limitation accurately.
Do not invent an armor benefit from inventory alone.

### Choose useful equipment

A suitable pickaxe supports ore collection.
Protection supports safer travel and combat.
A bucket, weapon, or other tool can help with a specific route problem.
The correct order depends on the task and current inventory.

Reserve the materials for the next essential item.
Do not spend the final required ingots on an optional duplicate.
Do not craft every possible item before the next small useful trip.

### Choose depth from the current world

Ore distribution depends on the game version and world generation.
An old tutorial depth is not proof of the best depth in this world.
A shallow Y value does not become a diamond level because you spent many calls reaching it.

Use the current version information or documented world rules for depth selection.
Use exposed ore and current searches as local evidence.
If the generation rules are unknown, state that uncertainty.
Continue a controlled route rather than inventing a precise optimum.

### Diamond collection procedure

1. Check the suitable pickaxe and its remaining durability.
2. Check food, light, blocks, and return reserves.
3. Travel through the known route to the planned work area.
4. Extend the route in short safe sections.
5. Search newly exposed areas under the visibility rules.
6. Inspect a discovered diamond target and its surroundings.
7. Address liquid or drops around the target.
8. Equip the suitable pickaxe immediately before mining.
9. Mine the ore.
10. Read the diamond inventory increase.
11. When the trip target is complete, return with the collected diamonds.
12. Store valuable surplus at home.

If no diamond appears in one segment, that segment can still establish useful access.
Choose the next segment according to supplies and route quality.
Do not interpret a negative search as a reason for a vertical shaft.

### Do not move the finish line

If the user asks for the first diamond, obtaining and preserving it completes that milestone.
Do not risk the first diamond for an unrequested larger amount.
If the user asks for a larger set, use several supported trips.
The next trip can start from the completed route.

