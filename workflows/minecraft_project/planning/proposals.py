"""Apply planner proposals to the task tree, and shape rejection feedback for the next attempt."""
from __future__ import annotations

import json

from pydantic import ValidationError

from workflows.minecraft_project.schemas import Condition, Node, PlanProposal, RefinementProposal, TaskTree
from workflows.minecraft_project.tree_store import TreeStore, digest


def scoped_rejected_tree(previous: TaskTree | None, rejected: TaskTree, source: str) -> dict:
    """Describe a rejected revision through changes to the supplied old tree.

    Unchanged conditions reference their exact previous node and condition ID.
    Changed conditions retain their full operands. Original rejected proposals
    remain on disk, including nodes omitted from this repair-only projection.
    Without an accepted tree, every candidate node stays explicit.
    """
    from workflows.minecraft_project.tree_store import node_digest
    prior_nodes = previous.nodes if previous is not None else []
    old = {node.id: node for node in prior_nodes}
    conditions = {digest(row.model_dump(mode="json")): {"from_node": node.id, "condition_id": row.id}
                  for node in prior_nodes for row in node.criteria + node.assumptions}
    changed = []
    for node in rejected.nodes:
        if node.id in old and node_digest(node) == node_digest(old[node.id]):
            continue
        value = node.model_dump(mode="json", exclude_defaults=True)
        for field in ("criteria", "assumptions"):
            if field in value:
                value[field] = [conditions.get(digest(row.model_dump(mode="json")),
                                              row.model_dump(mode="json", exclude_defaults=True))
                                for row in getattr(node, field)]
        changed.append(value)
    return {"original_receipt": source, "original_digest": digest(rejected.model_dump(mode="json")),
            "goal_id": rejected.goal_id, "removed_nodes": sorted(old.keys() - {node.id for node in rejected.nodes}),
            "changed_nodes": changed}


def scoped_rejection(previous: TaskTree | None, document: dict, source: str) -> dict:
    """Keep invalid raw model output distinct from a typed rejected tree.

    Syntax feedback includes a bounded excerpt around the parse error.
    Initial contract repairs retain all parsed operands because no accepted
    tree can supply omitted geometry. Neither form accepts the candidate.
    """
    if "text" not in document:
        rejected = TaskTree.model_validate(document)
        if previous is None:
            return {"original_receipt": source, "original_digest": digest(rejected.model_dump(mode="json")),
                    "kind": "invalid_initial_tree", "candidate_tree": rejected.model_dump(mode="json", exclude_defaults=True),
                    "repair": "No specification exists yet. Return a complete initial tree with every referenced node. Repair the supplied tree error; this candidate is not accepted work."}
        scoped = scoped_rejected_tree(previous, rejected, source)
        if scoped["removed_nodes"]:
            # A candidate missing stable identities fails before any operand
            # review. Start the repair from the authoritative retained tree.
            return {key: value for key, value in scoped.items() if key != "changed_nodes"} | {
                "kind": "removed_stable_nodes", "repair": "Use explicit updates against previous_tree. Preserve every stable node and parent-child reference; this rejected candidate never reached world execution."}
        return scoped
    text = document["text"]
    if text.strip().startswith("```"):
        text = "\n".join(text.strip().splitlines()[1:-1])
    offset, error = 0, "The output failed the proposal contract; use the supplied validation feedback."
    try:
        candidate = json.loads(text)
    except json.JSONDecodeError as invalid:
        offset, error = max(0, invalid.pos - 800), str(invalid)
    else:
        if previous is None:
            try:
                PlanProposal.model_validate(candidate)
            except ValidationError as invalid:
                errors = invalid.errors(include_url=False, include_input=False, include_context=False)
            else:
                errors = []
            return {"original_receipt": source, "original_digest": digest(document),
                    "kind": "invalid_model_output", "candidate": candidate, "validation_errors": errors,
                    "repair": "This is rejected output, not an accepted tree. Preserve its concrete operands while repairing the supplied contract errors."}
    return {"original_receipt": source, "original_digest": digest(document), "kind": "invalid_model_output",
            "original_characters": len(text), "error": error, "excerpt_start": offset,
            "excerpt": text[offset:offset + 2000]}


def rejected_context(store: TreeStore, source: str) -> dict:
    """Read a rejected proposal without inventing an initial tree revision.

    Initial planning can reject output before any specification exists.
    Retained revisions still supply their authoritative comparison tree.
    Initial repairs also retain the first contract rejection. Later invalid
    proposals cannot erase its concrete geometry. Original files stay exact.
    """
    document = json.loads(store.path_for(source).read_text(encoding="utf-8"))
    previous = store.load() if store.state["revision"] else None
    result = scoped_rejection(previous, document, source)
    if previous is None:
        for decision in store.state["decisions"]:
            if decision["role"] != "planner" or decision["identity"]["tree_revision"] != 0:
                continue
            first_source = decision["path"] + "/raw.json"
            path = store.path_for(first_source)
            if not path.is_file() or store.path_for(decision["path"] + "/output.json").exists():
                continue
            first = scoped_rejection(None, json.loads(path.read_text(encoding="utf-8")), first_source)
            if not first.get("validation_errors"):
                continue
            if first_source != source:
                result["first_rejected_proposal"] = first
            break
    return result


def initial_candidate(store: TreeStore) -> tuple[str, TaskTree] | None:
    """Read the latest fully declared initial candidate from rejection history.

    Every referenced identity must already exist in this immutable proposal.
    Later root-only fragments cannot replace its physical operands. The
    candidate remains unaccepted; explicit model updates still need the full
    tree validator, fresh world survey, and independent plan review.
    No candidate is available after a specification has been committed.
    """
    if store.state["revision"]:
        return None
    for number in range(store.state.get("rejections", 0), 0, -1):
        source = f"planning/rejected/{number:04d}.json"
        tree = TaskTree.model_validate(json.loads(store.path_for(source).read_text(encoding="utf-8")))
        ids = {node.id for node in tree.nodes}
        references = [key for node in tree.nodes for key in node.children + node.depends_on]
        references.extend(node.parent for node in tree.nodes if node.parent is not None)
        references.extend(condition.node_id for node in tree.nodes for condition in node.criteria + node.assumptions
                          if condition.kind == "accepted")
        if len(ids) == len(tree.nodes) and tree.goal_id in ids and all(key in ids for key in references):
            return source, tree
    return None


def apply_node_updates(previous: TaskTree, updates: list[Node]) -> TaskTree:
    """Merge explicit node replacements into a retained complete tree.

    Omitted nodes remain exact copies. The normal revision validator still
    checks stable IDs, parents, scope, conditions, and prerequisite references.
    """
    replacements = {node.id: node for node in updates}
    if not replacements or len(replacements) != len(updates):
        raise ValueError("Node updates need a nonempty list of unique stable IDs.")
    old_ids = {node.id for node in previous.nodes}
    nodes = [replacements.get(node.id, node).model_copy(deep=True) for node in previous.nodes]
    nodes.extend(node.model_copy(deep=True) for node in updates if node.id not in old_ids)
    return previous.model_copy(deep=True, update={"nodes": nodes})


def revision_candidate(previous: TaskTree, proposal: PlanProposal) -> TaskTree:
    """Build a candidate without confusing proof restoration with node edits.

    Restoration-only proposals retain the exact tree for the later proof
    boundary. They do not establish a concrete leaf repair or acceptance.
    Invalid node replacements raise ValueError before the task state changes.
    """
    if proposal.tree is not None:
        return proposal.tree
    if proposal.updates:
        return apply_node_updates(previous, proposal.updates)
    if proposal.restore_nodes:
        return previous.model_copy(deep=True)
    raise ValueError("A revision needs node updates or explicit proof restoration.")


def refinement_scope(store: TreeStore, instruction_revision: int, history: dict) -> str:
    """Open optional scope only when recovery still needs a ready leaf."""
    if instruction_revision != store.state["instruction_revision"]:
        return ""
    if history["recovery"]:
        nodes = {node.id: node for node in store.load().nodes}
        if not any(nodes[key].specialist == "recovery" for key in store.eligible()):
            return ""
    return store.state.get("refinement_target", "")


def refinement_plan(previous: TaskTree, target: str, proposal: RefinementProposal) -> PlanProposal:
    """Bind a model's replacement to the actual named refinement target."""
    updates = []
    if proposal.status == "planned":
        if proposal.replacement.id != target:
            raise ValueError(f"replacement must retain target ID {target!r}, not {proposal.replacement.id!r}.")
        old = {node.id: node for node in previous.nodes}
        copied = sorted(node.id for node in proposal.additions if node.id in old)
        if copied:
            raise ValueError("additions cannot replace existing unrelated nodes: " + ", ".join(copied)
                             + ". Keep them unchanged; add new children under replacement.")
        updates = [proposal.replacement, *proposal.additions]
    return PlanProposal(status=proposal.status, summary=proposal.summary, updates=updates, queries=proposal.queries,
                        recipe_queries=proposal.recipe_queries)


def survey_conditions(nodes: list[Node]) -> list[Condition]:
    """Inspect proposed physical operands once, without requiring completion.

    Survey facts describe the current site. Unmet future conditions are normal.
    Deduplication reduces repeated reads across parent and child criteria.
    """
    queries = {}
    for node in nodes:
        for condition in node.criteria + node.assumptions:
            if condition.kind == "block":
                for point in condition.positions or [condition.position]:
                    key = digest({"kind": "block", "position": point.model_dump(), "dimension": condition.dimension})
                    queries.setdefault(key, condition.model_copy(update={"position": point, "positions": []}))
            elif condition.kind == "region":
                key = digest({"kind": "region", "start": condition.position.model_dump(), "end": condition.end.model_dump(), "dimension": condition.dimension})
                queries.setdefault(key, condition.model_copy())
    return [Condition(id="survey-body", description="Current body state.", kind="alive")] + [
        condition.model_copy(update={"id": f"survey-{index:04d}"}) for index, condition in enumerate(queries.values(), 1)]


def restart_review(store: TreeStore, phase: str, instruction_revision: int, answer: str) -> bool:
    """Require fresh review of a retained tree before resuming unchanged work.

    Mark interrupted actor attempts without changing their original evidence.
    New instructions, human answers, and a required concrete leaf change still
    enter planning. A saved waiting interpretation cannot authorize mutation.
    """
    if phase != "plan" or not store.state["revision"]:
        return False
    for attempt in store.state["attempts"]:
        if attempt["state"] == "running":
            attempt.update(state="interrupted", journal=attempt["path"] + "/scratch/actor-journal.jsonl")
    store.state["recovery_observation"] = store.state["planning_observation"]
    store.save()
    return (instruction_revision == store.state["instruction_revision"] and not answer
            and not store.state.get("require_leaf_change"))

