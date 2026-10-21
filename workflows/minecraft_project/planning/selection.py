"""Selection role: check plan assumptions and choose the next eligible leaf."""
from __future__ import annotations

from pm_workflows.python_role import RoleContext

from workflows.minecraft_project.schemas import Node
from workflows.minecraft_project.tree_store import TreeStore
from workflows.minecraft_project.acceptance import evaluate, observe_conditions
from workflows.minecraft_project.planning.run_context import boundary, identity
from workflows.minecraft_project.history_encoding import gameplay_history


def check_assumptions(context: RoleContext, store: TreeStore, node: Node) -> bool:
    """Retain each current assumption read before it can release world work."""
    reads = observe_conditions(context, node.assumptions)
    checks = evaluate(node.assumptions, reads, certificates=store.state["acceptances"])
    number = store.state.get("assumption_checks", 0) + 1
    store.write(f"planning/assumptions/{number:04d}-{node.id}.json", {
        "identity": identity(context, store, node=node), "reads": reads, "checks": checks,
        "outcome": "preconditions_confirmed" if all(checks.values()) else "changed_assumptions"})
    store.state["assumption_checks"] = number
    store.save()
    return all(checks.values())


def select(context: RoleContext, store: TreeStore) -> dict:
    """Select only an independently reviewed leaf with valid prerequisites."""
    if not boundary(context, store):
        return {"status": "replan", "summary": "Project instructions changed before leaf selection."}
    if store.state.get("reviewed_revision") != store.state["revision"]:
        raise ValueError("The current tree has no independent scope review.")
    invalid = store.invalidate()
    for node in store.load().nodes:
        if store.valid(node.id) and node.assumptions:
            if not check_assumptions(context, store, node):
                store.revoke(node.id)
                invalid.append(node.id)
    invalid.extend(store.invalidate())
    store.save()
    if invalid:
        return {"status": "replan", "summary": "Acceptance evidence or current assumptions changed: " + ", ".join(invalid)}
    aggregates = store.ready_aggregates()
    eligible = store.eligible()
    renewals = [key for key in store.state.get("renewal_nodes", []) if not store.valid(key)]
    if renewals:
        aggregates = []
        eligible = [key for key in eligible if key in renewals]
    history = gameplay_history(store)
    if history["service_wait"]:
        return {"status": "waiting", "summary": "Restore the Minecraft service and observe retained effects before more work: " + str(history["service_wait"])}
    if history["recovery"]:
        aggregates = []
        recovery_nodes = {node.id for node in store.load().nodes if node.specialist == "recovery"}
        eligible = [key for key in eligible if key in recovery_nodes]
        if not eligible:
            return {"status": "replan", "summary": "Recovery remains active. Add or revise a prerequisite-ready recovery leaf: " + str(history["recovery"])}
    if aggregates or eligible:
        key = (aggregates or eligible)[0]
        nodes = {node.id: node for node in store.load().nodes}
        node, cursor = nodes[key], key
        while cursor is not None:
            assumptions = nodes[cursor].assumptions
            if assumptions and not check_assumptions(context, store, nodes[cursor]):
                return {"status": "replan", "summary": f"Current assumptions failed for {cursor} before dispatch."}
            cursor = nodes[cursor].parent
        actor_attempt = ""
        if aggregates:
            actor_attempt = next(row["id"] for row in reversed(store.state["attempts"])
                                 if row.get("state") == "complete" and row.get("report"))
        store.state["selection"] = identity(context, store, node=node, actor_attempt=actor_attempt)
        store.save()
        if aggregates:
            return {"status": "aggregate", "summary": f"Verify aggregate {key} before releasing dependent work."}
        return {"status": "selected", "summary": f"Selected verified-prerequisite leaf {key}."}
    if all(store.valid(node.id) for node in store.load().nodes):
        return {"status": "complete", "summary": "All leaf, story, and goal certificates passed."}
    return {"status": "waiting", "summary": "No leaf has valid prerequisite evidence. Inspect the retained task tree."}

