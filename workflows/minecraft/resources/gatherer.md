# Minecraft gatherer resource

Compiled from minecraft-survival-coach. Shared and specialist resources preserve the upstream sections.

## 19. Follow a flexible survival progression

The following sequence describes useful dependencies.
It is not a demand to repeat work that already exists.

| Stage | Useful outcome | Evidence for the next stage |
|---|---|---|
| Orientation | Known position, threats, and useful existing assets | A current task based on real state |
| Immediate support | Air, stable ground, accessible food | No active threat prevents work |
| Work camp | Crafting access and basic tools | Required recipes succeed |
| Shelter | A usable entrance, enclosure, and light | The player can enter and leave |
| Food and storage | Supplies for a trip and space for resources | Inventory and chest counts |
| Local resources | Wood, stone, coal, and accessible iron | Useful inventory increases |
| Better equipment | Suitable tools and available protection | Items exist and equipment works |
| Controlled descent | A lit route with a tested return | Actual landings and return travel |
| Diamond expedition | Appropriate depth, tool, food, and exit | Exposed ore harvested successfully |
| Return | Resources reach home | Chest or inventory evidence at home |

If the player already has iron tools, skip the wooden-tool stage.
If the player has tools but no food, address food.
If coal is absent, consider charcoal through the current recipe system.
If a cave is hazardous, collect safer surface resources or build another entrance.

A useful session can include several short trips.
Each return supplies the next trip.
The first trip does not need to reach the final depth.


## 20. Gather wood without losing the day

Wood supports tools, workstations, fuel, torches, and construction.
Logs and planks are different resources.
Converting every log into planks can remove the supply for charcoal or a building detail.

Before wood collection, choose the amount and purpose.
Reserve wood for essential recipes before decoration.
If a suitable axe is available, use it.
If no axe exists, use ordinary log mining under the tool rules.

For one tree:

1. Locate the trunk through a current observation or block search.
2. Choose an accessible trunk base.
3. Approach safe ground near the trunk.
4. Equip the suitable tool.
5. Use `minecraft_harvest_tree` or ordinary block mining.
6. Read the log count and inventory delta.
7. Check any uncollected drops or remaining useful logs.

If the helper returns zero logs, do not count the tree as collected.
Check the selected trunk and actual player position.
Use a nearer accessible trunk or individual exposed logs.
Do not repeat the same zero-log call without a changed condition.

If upper logs are inaccessible, compare their value with another nearby tree.
Do not build a tall risky tower for a small wood shortage.
If a safe temporary platform is useful, preserve a supported descent.

If saplings are available and planting is supported, replant suitable spaces near home.
Keep new trees clear of the entrance and main path.


## 21. Craft from counts and recipes

Use the current recipe search for unfamiliar or disputed recipes.
Read ingredient counts, output counts, and crafting-table requirements.
Do not infer a recipe from an item name alone.

`repetitions` normally means recipe applications.
It does not necessarily mean output item count.
A recipe can produce several items per application.

For a recipe output of four items:

```text
Needed output: 8 items.
Output per recipe: 4 items.
Required repetitions: 2.
```

Check the actual output quantity in the live recipe.
If the desired quantity is not an exact multiple, use ceiling division.
Account for items already in inventory.

### Crafting procedure

1. Choose the item and required count.
2. Read the recipe.
3. Compare each ingredient with the inventory.
4. If the recipe requires a crafting table, check access to that table.
5. Craft the required batch.
6. Read the actual crafted count and inventory delta.
7. Update the resource budget.

If a crafting table is required, place one within usable reach.
Check that the placed block exists.
If the player is outside the documented range, move closer.
If the current tool requires a table interaction, use that interaction.

A table in inventory is not a placed table.
A table several blocks above the player can be outside usable reach.
Opening a distant table does not prove that every crafting tool can use it from that position.

If crafting reports a missing ingredient, obtain that ingredient.
If crafting reports a missing table, address table access.
Do not alternate between these causes without reading the result.

### Use `craft_max` deliberately

If all possible output serves the task, `craft_max` is suitable.
Otherwise, craft a counted batch.
Do not convert the entire wood reserve to sticks by accident.
Do not spend all iron on an optional item before the required pickaxe.
Read the consumed counts after every large batch.


## 22. Smelt and cook with a plan

A furnace needs input, suitable fuel, and time.
The recipe output can differ from the input name.
Raw iron is not an iron ingot.
Raw food is not cooked food.

Before smelting:

1. Check the furnace location and access.
2. Check the input count.
3. Check the selected fuel and count.
4. Reserve fuel for other essential tasks.
5. Choose the required output amount.

Use the current smelting tool schema.
Read the consumed input, consumed fuel, and output.
If the result is partial, use the actual output count.
If the output remains in the furnace, retrieve it through the supported interaction.
Check inventory before a recipe that needs that output.

If a tool moves the player to reach a furnace, update the player position.
If the result omits position and access required movement, observe before the next precise task.
Do not assume the player remains beside the original door.

If fuel is missing, use a suitable available fuel or obtain more.
If charcoal is a supported alternative, keep enough logs and fuel for its recipe.
Do not claim coal exists because charcoal is possible.


## 23. Keep an inventory budget

Divide supplies by purpose.

| Supply | Main use | Reserve purpose |
|---|---|---|
| Food | Hunger support | Return trip and delay |
| Pickaxe | Excavation and ore | Exit repair and required ore |
| Blocks | Construction | Steps, barriers, and landings |
| Torches | Visibility and lighting | Return route and unexpected branch |
| Wood | Recipes | Replacement tools and light |
| Fuel | Cooking and smelting | Return meal or essential ingots |
| Empty slots | New items | Valuable ore and recovered tools |

Choose quantities for the actual trip.
A short visible coal trip needs less than a long descent.
Do not treat one fixed inventory list as a law for every situation.

For a modest early expedition, use this starting example:

```text
Food: Several useful meals, with part reserved for return.
Blocks: About one available stack of ordinary solid blocks.
Light: Enough torches for the planned segment and its return.
Tools: A suitable pickaxe plus a spare or replacement materials.
Wood: A small unspent reserve.
Space: Several empty inventory slots.
Route: A known home location and marked exit.
```

This example is a planning aid.
It is not a claim that every trip needs a full stack or a particular food count.

Before departure, define the return trigger.
Use the first of task completion, reserve use, route uncertainty, or a new threat.
If a reserve trigger occurs, return before optional work.

### Full inventory

If inventory is full, store surplus before another resource task.
If home is distant, preserve valuable and necessary items first.
Discard only genuinely expendable items under the user permissions.
Do not discard the only food, pickaxe, light source, or escape blocks for decorative stone.

If drops overflow, record their safe location.
Make space before collection.
Do not step into lava or a deep hole to recover a low-value drop.


## 27. Mine a known block correctly

Block identity, tool suitability, reach, hazards, and loot are separate checks.
Skipping one check can make the whole action useless.

For a resource block:

1. Obtain its position from a current visible search or observation.
2. If target identity or geometry is unclear, inspect the target.
3. Check that a safe approach exists.
4. Probe the target before excavation.
5. Equip the correct tool immediately before mining.
6. Mine the target through the ordinary tool.
7. Read the mined block and harvest result.
8. Read the inventory delta.
9. Address any new opening or liquid.

If the tool reports stone, record stone.
If the inventory gains cobblestone, record cobblestone.
Do not record iron from a texture guess.

### Blocks and drops

Common block and item names differ.
Examples include stone and cobblestone, iron ore and raw iron, or coal ore and coal.
The exact result depends on current mechanics and tool properties.

Read the actual drop and inventory fields.
Use the item name for inventory searches and recipes.
Use the block name for block searches.

This server family can insert mined loot directly into inventory.
Other configurations leave item entities for collection.
Inventory deltas and pickup fields decide which case occurred.

If the inventory already gained the full output, do not chase an imaginary drop.
If the result reports an uncollected drop, locate the actual item and its safe approach.
If inventory space is the problem, make space before collection.

### Held-tool drift

A previous action can change the held item.
A torch, dirt block, or plank can remain in hand after useful work.
An inventory pickaxe does not prove a held pickaxe.

Re-equip the correct tool immediately before mining after movement, construction, or lighting actions.
Read warnings about tool loss or held-item changes.
If the tool is absent, obtain a replacement before the dependent ore task.

### Replacement liquid

A mined cell can become water or lava after block removal.
That replacement does not automatically mean the dig failed.
It also does not mean the cell is safe to enter.

If the inventory delta and world result show success, keep that progress.
If water or lava appears, assess the new hazard before more mining.
Do not repeat mining against liquid in place of the removed ore.


## 28. Interpret excavation failures

| Result | Meaning to check | Next action |
|---|---|---|
| `target_out_of_range` | The player cannot reach the target | Approach through a known safe route |
| `unharvestable` | The held tool cannot collect the target | Equip or obtain a suitable tool |
| `not_found` | The expected solid target is absent | Inspect the current block |
| `target_changed` | The target changed or the dig outcome differs | Read inventory delta and inspect the cell |
| `no_path` from a vein helper | The helper cannot reach its selected work | Inspect the approach or choose an exposed block |
| Zero mined count | No resource progress occurred | Change the relevant condition |
| Positive count with remaining ore | Some work remains | Continue only within the task budget |
| Tool exception | A tool call did not complete normally | Read the error and fresh state |

If a target is out of range, do not search nearby coordinates at random.
The player position causes the reach problem.
Move to a valid approach before another mining attempt.

If a result lacks a reliable final state, inspect before you repeat a destructive action.
If a successful action exposes more ore, use the new evidence.
Do not assume a connected vein is exhausted because one helper call ended.


## 29. Collect a vein without chasing it forever

Before a vein task, set a useful quantity or local boundary.
Examples include coal for the next trip or iron for the next required recipes.

If its assumptions fit the visible state, use the vein helper.
Read its actual count, drops, and stop reason.
If more exposed ore remains, choose a safe bounded continuation.

For a manual continuation:

1. Search for the next exposed matching block.
2. Check its distance and approach.
3. Probe the excavation.
4. Equip the tool.
5. Mine one block.
6. Read the inventory change.
7. Check the task target and reserves.

If the task target is complete, stop that vein task.
If a hazard appears, address the hazard or save the remaining vein.
If ore leaves the safe work area, save its position for another trip.

A vein location is useful progress despite some remaining ore.
The next expedition can use that location.
There is no requirement to empty every connected vein immediately.

