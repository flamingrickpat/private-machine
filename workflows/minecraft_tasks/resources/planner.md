# Plan a retained world

Turn the requested goal into a generic goal/story/leaf task tree.
Use the supplied schema exactly. Return one JSON object.
Use compact JSON. Omit optional fields that use their default values.
Group exact cells with identical accepted names in one block condition's positions list.
For a block condition, omit end and minimum. Every listed position must match.
For an inventory or region condition, minimum is a count threshold.
Only a region condition accepts end. It measures a visible histogram, not individual hidden cells.
Use stable short IDs. Runtime code assigns revision counters.
Only the root goal has parent=null. Set every other node's parent to its containing goal or story ID.
List each child in that parent's children. Parent and child references must agree.
Keep specifications separate from attempts and observed effects.
A leaf describes one meaningful outcome. It can require many tool calls.
Use gathering, crafting, construction, and usability prerequisites when the requested outcome requires them.
Do not limit goals to a fixed catalog.
Choose explicit, feasible acceptance conditions from the observation primitives.
Inventory names accept globs. Block names use exact Minecraft identifiers.
Use actual world coordinates. Do not invent a surveyed building site.
Preserve the requested site and explicit distance bound when the body moves.
Exact cell reads establish existing construction. Retain correct blocks instead of repeating completed physical work.
The repaired body can dig grass and tall_grass. is_solid describes collision bounds; it does not restrict digging to solid cubes.
For preparation before block placement, allow observed replaceable vegetation when placement will remove it, or request ordinary plant digging.
Keep final required air cells strict after construction. Require fresh block evidence after vegetation clearance.
project_initial_observation is the original project baseline. observation is current state after retained world effects.
For an unchanged request, keep relative inventory targets based on the original baseline. Do not add the requested increment again after each failed attempt.
When the user changes instructions, use the new wording and current state to define the revised outcome.
Use a navigation survey leaf before construction when the current site is unsuitable.
Use previous_site_survey when supplied. Its retained cells describe the earlier surveyed site, not current completion.
Use latest_verification for failed acceptance. It identifies the original actor and verifier, exact operands, checks, and observed reads.
A region predicate measures blocks currently visible in that volume. It does not establish where earlier harvesting occurred.
Do not require harvested logs to remain as world blocks after gathering them into inventory.
Describe all usability requirements with observable criteria, including access, interior clearance, walls, and roof when requested.
Root criteria must cover the complete finished structure, rather than only a roof or one corner.
An installed Minecraft door occupies both its lower and upper cells. Both cells contain the door block, not air.
The region tool excludes hidden cells. Prefer exact block conditions for enclosed structures.
Acceptance records past attainment. Use assumptions for facts that must remain true.
Do not require gathered materials to remain in inventory after a dependent crafting task consumes them.
Use kind=accepted with node_id to cite independently verified earlier work in aggregate criteria.
For goals and stories, cite a verified descendant or declared prerequisite node. For leaves, cite a node whose leaves are explicitly declared prerequisites.
References remain valid when a prerequisite leaf becomes a story. Its original criteria stay on that story.
The selector verifies a completed story before dependent work can consume its materials.
For a material story, cite accepted gathering and crafting leaves rather than requiring consumed stock to remain.
For entry and exit, cite both accepted leaves in aggregate criteria. The body cannot finish at both positions simultaneously.
Keep fresh block criteria for the finished structure. Accepted references do not establish current inventory or current position.
Preserve stable IDs and all useful completed work during replanning.
Copy unchanged specifications exactly, including wording, criteria order, and assumptions.
Do not paraphrase accepted nodes. Change them only when the revised request or original evidence requires it.
When rejected_proposal is supplied, repair the stated contract error before submitting that proposal again.
When refinement names a target, change only that node and its descendants.
Preserve its original acceptance conditions exactly.
Split an oversized leaf into a story with prerequisite-ordered child outcomes.
Do not reuse Git rollback assumptions. All world effects remain.
When world facts are missing, return survey with explicit queries using the condition operands.
Use block queries for exact terrain layers or clearance cells. Inventory, position, alive, and region queries are also available.
Survey thresholds do not accept work. Use requested_world_survey as retained planning evidence.
Request world reads instead of asking the user to inspect terrain that MCP can read.
When an external service or a user preference is missing, return waiting with the specific question.
When the goal is impossible, return blocked with the actual reason.
