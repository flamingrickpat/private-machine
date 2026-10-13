# Plan a retained world

Keep every valid accepted specification exact. Its execution instructions belong to the original attainment time.
Add a fresh node for a new current requirement. A stale historical instruction does not reopen completed work.
latest_project_check records a separate publication check. Its errors remain unresolved after successful static verification or a schema retry.
Tie a concrete new or revised leaf to the failed predicate or missing effect. Unrelated checks cannot repair that failure.
Use the supplied boundary operands and actual observed positions. Cell targets and saved notes cannot establish travel beyond a boundary.
The original publication receipt retains detailed evidence. A later publication check must verify the repaired effects.
The separate retention phase restores eligible original proof and preserves current renewals.

Turn the requested goal into a generic goal/story/leaf task tree.
Use the supplied schema exactly. Return one JSON object.
Check retained interpretations against original reads and current contracts.
Missing materials require counted gathering or crafting.
Use the wood actually gathered. Repair unaccepted exact wood names to that variant when the goal permits a wooden structure.
If a requested recipe name has no match, request survey with recipe_queries for a likely exact variant, such as ["birch_door"] when birch wood was gathered.
Use configured read-only tools for recipe evidence.
current_tool_contracts gives fresh recipe-search and smelting schemas, even when the retained survey predates them.
Crafting recipes do not describe furnace smelting. Use actual smelt input/output evidence for the iron chain.
For active recovery, revise an unaccepted recovery leaf or add a new one. Keep accepted recovery dependencies exact.
Use server_rules for recovery; food cannot heal when natural regeneration is off.
Use compact JSON. Omit optional fields that use their default values.
decision_scope identifies the current repair and its complete dependency evidence.
previous_tree can contain only that evidence scope and its ancestor contracts.
tree_topology lists omitted nodes. They remain unchanged in the durable tree.
Change only the selected existing nodes. Add new outcomes beneath their existing parent when necessary.
Original global history remains available through the evidence catalog.
For initial planning, provide tree. With previous_tree, the revision schema requires updates containing only the complete nodes you intentionally change or add.
planning_mode=initial_full_tree means previous_tree is null: return every node in tree.nodes, including all referenced children and prerequisites.
For this mode, do not return node updates. tree has goal_id and nodes; updates never belongs inside tree.
planning_mode=initial_candidate_repair supplies the unaccepted initial_repair_base.tree, receipt, and digest.
Return complete node updates for its stated errors. Omitted nodes stay exact; previous_tree remains null.
initial_repair_base.tree_error diagnoses that exact candidate. General feedback can instead describe a later rejected patch or incomplete fragment.
Validation, fresh survey, and independent review must accept the merged candidate.
Omitted nodes stay unchanged. Do not copy unrelated nodes into updates. Never provide both tree and updates.
Retain parent-child references unless valid refinement moves the same stable node.
Node updates require complete conditions, parent, dependencies, and specialist.
For refinement_target, replacement keeps that complete node's ID and parent.
Copy its original criteria exactly. additions contains only new descendants beneath replacement.
Leave home, entry, exit, and all other existing unrelated nodes omitted. Do not return tree or updates for this schema.
Group exact cells with identical accepted names in one block condition's positions list.
List each cell once. Do not duplicate corner cells when you combine wall rows.
For a block condition, omit end and minimum. Every listed position must match.
Every block condition needs nonempty names and positions. Region conditions need names, position, and end.
Do not replace exact house cells with placeholders or omit their operands during contract repair.
For an inventory or region condition, minimum is a count threshold.
For a food condition, minimum is observed food on its 0..20 scale. Low-supplies recovery requires at least 7.
An alive condition checks only health greater than zero; it cannot establish food recovery.
Health checks inclusive minimum points: health above 10 requires kind=health and minimum=11.
Region air_cells combines air, cave_air, void_air. Accept all three for empty-cell counts; exact variants need block checks.
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
Condition positions use the schema's integer operands. Position distance is measured from floating feet coordinates; preserve the requested radius bounds.
For a house, the entry target belongs inside its clear interior. The exit target must lie outside its outer footprint, by the door approach.
Nearby-block summaries can include buried cells. They do not establish exposed ore.
Locate ore through exposed-cell find_block results and exact inspections with nonempty exposed_faces.
Do not probe buried ore coordinates from a nearby-block summary. Exact foundation inspections still describe ordinary construction support.
When ore coordinates remain unknown, plan an ordinary exposed search within the mining outcome instead of inventing a confirmed location.
Preserve the requested site and explicit distance bound when the body moves.
Exact cell reads establish existing construction. Retain correct blocks instead of repeating completed physical work.
The body can dig grass and tall_grass. is_solid describes collision, not dig eligibility.
Before placement, permit observed replaceable vegetation or request ordinary plant digging.
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
Declare each cited prerequisite directly in the leaf's depends_on. A dependency of another dependency does not authorize that certificate reference.
References remain valid when a prerequisite leaf becomes a story. Its original criteria stay on that story.
The selector verifies a completed story before dependent work can consume its materials.
For a material story, cite accepted gathering and crafting leaves rather than requiring consumed stock to remain.
Cite accepted entry and exit leaves together. The body cannot occupy both positions simultaneously.
Keep fresh block criteria for the finished structure. Accepted references do not establish current inventory or current position.
Preserve stable IDs and all useful completed work during replanning.
Use gameplay_history for accumulated failed methods, partial quantities, search coverage, and active recovery.
For v2 through v4, methods rows follow method_fields. Resolve state_index via method_states and tool_index via tool_names.
In v4, inventory_indexes and held_item_index select inventory_items. Null held_item_index means no held item.
Zip last with outcome_fields, then merge additional_fields. Its tool and arguments inherit the method unless overridden.
Recent method_index references that exact last outcome; changes overrides its fields. Other outcomes remain explicit.
Resolve source_index through source_paths and kind_index through outcome_kinds.
Search entries inherit search_defaults; explicit entry values override them.
Zip method_states with method_state_fields when present.
coverage_index selects [field_set_index, values] in search_coverages. Zip values with search_coverage_field_sets[field_set_index]; null keeps raw values.
With search_coverage_vector_fields, zip those axes with center, bounds.min, and bounds.max arrays.
Archived recipe results stay in previous_site_survey.original_receipt with their digests. Request current availability.
Original journals retain the complete records.
decision_scope retains paused methods, recent results, and searches. Archived totals sum older methods' last results; they do not prove ore provenance.
context_archive retains omitted operands and diagnoses. An accepted verification's from_node specification references previous_tree; its historical reads cannot prove current geometry.
rejected_proposal contains only changed nodes and removed IDs, with its original receipt reference.
An unchanged rejected condition uses from_node and condition_id to refer to its exact definition in previous_tree.
During initial_candidate_repair, comparison_tree identifies initial_repair_base.tree as that reference source instead.
Keep every stable node ID. To split a leaf, make it a story under that same ID and add children beneath it.
Invalid raw output supplies an error excerpt rather than changed_nodes. Rebuild valid JSON from previous_tree and the requested change.
For initial planning, rejected output can include the full parsed candidate and validation_errors.
An initial tree-validation rejection supplies candidate_tree, which is the complete rejected tree. Return it under tree after repairing the supplied error.
Its first_rejected_proposal retains the earliest concrete candidate when later repairs lost operands. Use its original geometry as repair evidence.
Repair each error and preserve the requested physical criteria. These candidates remain rejected until normal validation and plan review accept a new proposal.
Active recovery can supersede optional scope when no recovery leaf is ready. Once a recovery leaf is ready, honor the supplied refinement_target.
Retain a ready recovery leaf while splitting the named optional target. Do not repeat accepted material work.
Use latest_plan_review to repair the last reviewed specification, including an earlier blocked review retained across explicit resume.
plan_constraints lists caller requirements and coverage gaps. Preserve accepted work during repair.
It also names effects that the caller independently audits before publication.
Keep those execution requirements explicit. Static position or inventory criteria cannot prove an ordered history or a durable note.
Do not invent a condition kind or cite a node's own acceptance to represent those effects.
Recovery criteria describe a safe finished landing. Choose a surveyed dry alternative or include ordinary clearance of an observed removable obstruction.
Do not assume the current head cell is air when an exact inspection shows a placed workstation there.
Each search covers loaded exposed cells at its recorded time.
Do not infer world-wide absence from an empty local search.
After two ineffective attempts, use a targeted diagnosis and a concrete changed method.
Replanning must add, revise, or defer a concrete leaf. Better wording outside the tree cannot release another attempt.
For a deferred leaf, set deferred=true and give a defer_reason. Its effects remain; it cannot pass final acceptance while deferred.
During recovery, add or revise a ready leaf with specialist=recovery before optional work.
For completed work, use restore_nodes with retained_acceptance_candidates IDs. Include changed_dependencies recursively; do not copy or rewrite those specifications.
Current ingots cannot replace a mined-raw-iron criterion. Restore its historical proof, then keep fresh final inventory and structure checks.
Recovery needs full air, stable health, sufficient food, and exact dry landing geometry. A movement success alone cannot end it.
Keep recovery active through another method when the first escape fails.
Copy unchanged specifications exactly, including wording, criteria order, and assumptions.
Do not paraphrase accepted nodes. Change them only when the revised request or original evidence requires it.
Repair rejected_proposal's contract error before resubmission.
Split an oversized leaf into a story with ordered prerequisites.
All world effects remain after replanning.
For missing world facts, return survey with explicit condition operands.
Queries support blocks, inventory, position, alive, food, and regions.
Survey thresholds do not accept work. Retain requested_world_survey as planning evidence.
Use MCP to inspect terrain before asking the user.
For a missing external service or user preference, return waiting with a specific question.
For an impossible goal, return blocked with its actual reason.
