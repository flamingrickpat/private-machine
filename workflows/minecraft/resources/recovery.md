# Minecraft recovery resource

Compiled from minecraft-survival-coach. Shared and specialist resources preserve the upstream sections.

## 14. Escape water calmly and completely

The server mod can make drowning slow.
Slow drowning gives time to choose a good escape action.
It does not make underwater resource work the correct response to an unplanned immersion.
No fixed vanilla drowning countdown is required for this procedure.

Water at the feet does not always mean the head lacks air.
Use oxygen, head context, position, and an image together.
A shallow crossing with stable air differs from an unplanned underwater trap.

If the head is underwater or air decreases unexpectedly, start recovery.
Pause mining, crafting for ore, and new exploration.
Choose air and a stable position as the recovery target.

### Escape procedure

1. Identify the last dry position or a visible dry bank.
2. Check whether an ordinary route leads there.
3. Use a suitable walk tool toward that dry destination.
4. Read the final position and target offset.
5. Read the final feet and head context.
6. Observe oxygen and health.
7. If air or stability remains inadequate, continue recovery.

If a direct walk fails, inspect the nearby opening and vertical route.
If open water exists above, use a supported upward movement action.
If a ledge blocks the exit, choose a bank with a smaller rise.
If clear geometry permits a pulse, use one short jump or forward-jump pulse.
Check whether the pulse improved height or distance to the bank.

If a current prevents progress, choose a route across or away from the current.
If safe block placement can create a landing, build that landing from a stable position.
If a dry floor and headroom permit a pillar, use ordinary pillar actions.
If the pillar tool requires solid support, do not repeat it while unsupported in water.

If one block blocks the only escape, inspect that block and its surroundings.
If removal opens the escape without a worse hazard, mine that obstruction.
Do not call an optional ore block an escape obstruction.

### Escape completion

Check all of these facts:

- The head has air.
- Oxygen is restored or clearly recovers toward the reported normal level.
- The player has stable support or a controlled safe position.
- The current does not pull the player back into danger.
- Health does not decrease from the same threat.
- A next safe move exists.

If these facts hold, end water recovery.
Address low health or supplies before the old resource task.
If the route remains flooded, choose a dry work area.

### Example

```text
Goal: Obtain diamonds.
Observation: Feet Y=61. Head underwater. Oxygen below normal.
Current task: Escape to the dry bank.
First result: No path to the selected bank. Position changed slightly.
Next action: Inspect the nearby lower bank from the new position.
Paused task: Obtain diamonds.
```

The failed first route changes the escape method.
It does not change the need for air.


## 15. Handle lava, fire, gravel, and falls

### Lava and fire

WARNING: Do not enter lava to collect an item or shorten a route.
Lava can damage the player and destroy dropped items.

If lava threatens the current position, move away through a known safe route.
If a safe barrier can block the flow, place the barrier from safety.
If the documented tools support a suitable water action, use it from a safe position.
Check the resulting blocks before movement.
Do not assume a water action creates the exact solid block that you expect.

If the player is on fire, prioritize a supported method to end the fire or reach protection.
Check health after that action.
Do not open an optional recipe search during active fire damage.

If lava fills the removed cell, mining can still succeed.
That success proves block removal, not a safe opening.
If lava replaces the cell, stop the excavation and address the lava.

### Gravel and sand

WARNING: Do not remove support beneath falling material from directly below it.
The material can fall into the player or block the route.

If a safety probe reports gravel or sand, inspect the column from the side.
Keep a supported retreat position.
Clear material through a controlled method that preserves the exit.
If the safe method is unclear, choose another excavation direction.

A hazard stop is useful information.
Do not bypass it with the same excavation performed one block at a time.

### Falls

WARNING: Do not dig straight down beneath the player.
The next opening can contain a deep drop, liquid, or an inescapable shaft.

Use a staircase or a known supported route for descent.
Before a downward step, check its landing and the return step.
Do not trust a dark opening as a shallow opening.
Do not jump into a cave merely because ore is visible below.

If the player falls unexpectedly, observe from the new position.
If the exit or support is unknown, start recovery.
Do not continue the old coordinate sequence.


## 16. Use health and food as separate measurements

Health is not hunger.
Full hunger does not prove full health.
A mod can change healing, sleep, nutrition, movement, or damage timing.
The visible fields do not always expose every mod state.

Read the supplied measurements.
Use the user description of the mod as context.
Do not invent an invisible nutrition value or sleep timer.

For a normal 20-point health scale, 1 health point is half a heart.
It is not one full heart.
If the reported scale differs, use the reported maximum.

### Practical health rules

If health decreases unexpectedly, identify the threat before more resource work.
If health is critically low, choose nearby sheltered tasks and necessary recovery actions.
Do not start elective combat, deep exploration, or a dangerous crossing at critical health.
If health remains low despite food, inspect the available mod information.
Do not assume another identical meal will repair an unknown healing problem.

As a conservative default, treat one quarter of maximum health as low health.
Treat unexpected health loss as important at any starting value.
These are planning thresholds, not claims about the mod.

### Eating procedure

1. Check the food level and available food.
2. If food is needed, use `minecraft_eat_best` or the documented item-use method.
3. Read the consumed item and resulting food level.
4. If the result is unclear, obtain one fresh observation.
5. Update the inventory count.

If food is already full, do not repeat the meal.
If the inventory lacks food, obtain food through a supported survival method.
Do not mistake seeds for a prepared meal.
Do not count uncooked food as cooked food.
Do not assume every item with a plant name is edible.

If cooking adds unacceptable delay during immediate starvation, use an available suitable food according to current mechanics.
Otherwise, cook food before an expedition where practical.


## 17. Obtain food before a long trip

Choose a food source that fits the current equipment and terrain.
Possible sources include animals, crops, gathered food, and stored food.
Use current recipes and supported interactions.

For animals:

1. Obtain a fresh entity ID.
2. Check the surroundings and retreat route.
3. Equip a suitable weapon.
4. Attack through the ordinary combat tool.
5. Check the outcome and inventory.
6. If the output remains in safe reachable drops, collect those drops.
7. If furnace access is practical, cook the food.

Do not pursue an animal across dangerous terrain for one meal.
If food exists nearer home, use that source.
If breeding is practical, preserve enough animals for that plan.
An urgent meal and a long-term farm are different tasks.

For a small farm:

1. Select suitable soil near safe water access.
2. Obtain the required tool and planting item.
3. Read the current interaction schema.
4. Prepare one soil cell.
5. Inspect the changed soil.
6. Plant one item.
7. Inspect the crop.
8. Expand only after that first planting succeeds.

If the interaction is unsupported, record the specific limitation once.
Choose another food source.
Do not repeat the same rejected hoe or seed action across many cells.
Do not treat a planted crop as an immediate meal.
Do not harvest an immature crop merely because you need food now.


## 18. Handle night and sleep

A house, a bed, and sleep are separate achievements.
A roof does not reset spawn.
A bed in inventory does not prove a usable placed bed.
A placed bed does not prove a successful sleep.

If night approaches during a local expedition, compare remaining work with the return route.
Prefer a return before supplies or visibility become inadequate.
Inside the house, complete useful tasks such as cooking, crafting, storage, and repairs.

If a bed is available, use the sleep tool under its documented conditions.
Read the result.
Check the resulting time or spawn evidence that the tool supplies.
If sleep fails, address the reported condition.
Do not repeat sleep without a relevant change.

If sheep or bed materials are absent, use a secure lit shelter.
Continue safe tasks while the world advances.
Do not search indefinitely for sheep through unsafe terrain.
Do not assume an entire game night passes during one short tool call.

If the mod reports sleepiness, use that information in the expedition plan.
If the mod requires a bed, obtain one as a concrete task.
Do not substitute food for sleep unless the current mechanics explicitly support that effect.


## 33. Return before the trip fails

Return at the first planned return trigger.
Do not wait for every reserve to reach zero.

Return triggers include:

- The trip target is complete.
- Food reaches the return reserve.
- Tool durability threatens the return or required ore collection.
- Light or blocks reach the reserve.
- Inventory space prevents useful collection.
- The exit becomes uncertain.
- An unexpected hazard changes the trip conditions.
- Context is nearly exhausted.

For the return:

1. Select the next known landing toward home.
2. Use the suitable movement tool.
3. Read the actual position.
4. Repeat through the known route.
5. Check the home entrance.
6. Enter and secure the home area.
7. Store surplus.
8. Restore supplies.
9. Save the completed trip result.

If one return segment fails, inspect that segment.
Do not resume mining as a substitute for solving the return.
If a saved route is obsolete, record the changed section after recovery.


## 34. Use combat for a reason

Combat can provide food or remove a threat.
Combat also consumes health and attention.
A nearby entity is not automatically a useful target.

Before an attack, use a fresh entity ID.
Check entity type, distance, surroundings, and escape space.
Equip the intended weapon immediately before the attack.
Use the ordinary attack tool.
Read the attack result and fresh state.

If the entity survives, check whether another attack is safe.
Do not assume one call always kills the target.
If a different hostile entity appears, reassess the encounter.
If health decreases too far, retreat through a known route or use cover.

Do not fight an optional hostile entity at critical health.
Do not chase drops through an active threat.
Do not assume an old entity ID still refers to a present target.

If the only safe exit requires defense, use the minimum combat needed for escape.
Keep recovery as the current task.
After separation, check health and supplies before other work.


## 35. Handle death without another rushed loss

Death is a loss to recover from, not a travel tool for normal play.
Do not use suicide to replace a solvable exit or return route.
If the user explicitly authorizes a reset, obey the exact authorized scope.

After an unexpected death:

1. Observe the respawn state.
2. Read the current position and inventory.
3. Recall the death location and route.
4. Identify the cause of death.
5. Prepare enough supplies for a safe recovery attempt.
6. Approach only through a viable route.

If the dropped items lie in the same active hazard, address that hazard first.
Do not repeat the fatal movement with an empty inventory.
If recovery is not practical, preserve the remaining base and rebuild essential supplies.
Report what was lost and what remains uncertain.

