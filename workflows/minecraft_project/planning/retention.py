"""Retention role: keep valid accepted work across a revised plan."""
from __future__ import annotations

from pm_workflows.python_role import RoleContext

from workflows.minecraft_body.lease import BODY_LEASES
from workflows.minecraft_project.schemas import RetentionChoice
from workflows.minecraft_project.tree_store import TreeStore
from workflows.minecraft_project.evidence_selection import retention_operation, retention_tree
from workflows.minecraft_project.planning.run_context import boundary, identity, request
from workflows.minecraft_project.planning.decisions import reason


def retain(context: RoleContext, store: TreeStore, observation: dict | None = None) -> dict:
    """Run a separate native proof decision before a material actor is eligible."""
    if not store.state["revision"] or not store.acceptance_candidates():
        return {"status": "unchanged", "summary": "No historical specification restoration is pending."}
    if observation is None:
        with BODY_LEASES.command(context.runtime.tools["minecraft_endpoint"](), context.runtime.stop_event):
            reply = context.runtime.tools["minecraft_call"]("minecraft_observe", {"include_image": False})
        if not reply["ok"]:
            return {"status": "waiting", "summary": "Historical restoration needs an identified current body observation."}
        observation = reply["result"]
    tree = store.load()
    candidates = store.acceptance_candidates()
    payload = {"identity": identity(context, store), "goal": request(context).goal or request(context).instruction,
               "observation": observation, "feedback": store.state.get("retention_feedback", ""),
               "current_nodes": [node.model_dump(mode="json") for node in tree.nodes
                                 if node.id in {row["specification"]["id"] for row in candidates}],
               "candidates": [{**row, "currently_valid": store.valid(row["specification"]["id"])} for row in candidates]}
    choice = reason(context, store, "retention", RetentionChoice, payload)
    if choice.status != "planned":
        return {"status": choice.status, "summary": choice.summary}
    try:
        proposal = retention_operation(store, choice)
        store.write(store.state["decisions"][-1]["path"] + "/operation.json", proposal.model_dump(mode="json"))
        proposed = retention_tree(store, proposal)
        if not boundary(context, store):
            return {"status": "unchanged", "summary": "Project instructions changed before historical restoration."}
        store.revise(proposed)
    except ValueError as error:
        store.state["retention_feedback"] = str(error)
        store.save()
        return {"status": "invalid", "summary": "Native historical restoration was rejected: " + str(error)}
    current = context.runtime.tools["managed_instructions"]()
    store.state.update(instruction_revision=current["revision"],
                       goal=current["instruction"] or request(context).goal or request(context).instruction,
                       renewal_nodes=list(dict.fromkeys([*store.state.get("renewal_nodes", []),
                                                         *[row.node.id for row in proposal.renewals]])), retention_feedback="", feedback="")
    store.state.pop("require_leaf_change", None)
    store.save()
    return {"status": "planned", "summary": proposal.summary}

