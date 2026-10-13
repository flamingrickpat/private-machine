"""Survey role: read-only world reads that the planner requested."""
from __future__ import annotations

import time

from pm_workflows.python_role import RoleContext

from workflows.minecraft_body.lease import BODY_LEASES
from workflows.minecraft_project.schemas import Condition
from workflows.minecraft_project.tree_store import TreeStore
from workflows.minecraft_project.acceptance import observe_conditions
from workflows.minecraft_project.planning.run_context import identity


def survey(context: RoleContext, store: TreeStore) -> dict:
    """Read selected world operands and recipe patterns without an actor."""
    pending = store.state["pending_survey"]
    current = context.runtime.tools["managed_instructions"]()
    if pending["instruction_revision"] != current["revision"]:
        store.state.pop("pending_survey")
        store.save()
        return {"status": "replan", "summary": "Instructions changed before the requested world survey."}
    queries = [Condition.model_validate(row) for row in pending["queries"]]
    relative = f"planning/requested-surveys/{len(store.state['decisions']):04d}.json"
    reads = observe_conditions(context, queries)
    recipes = recipe_reads(context, pending.get("recipe_queries", []))
    store.write(relative, {"identity": identity(context, store), "instruction_revision": current["revision"],
                           "queries": pending["queries"], "reads": reads,
                           "recipe_queries": pending.get("recipe_queries", []), "recipes": recipes,
                           "scope": "Planning observations only; these reads do not accept work."})
    store.state.update(requested_survey=relative)
    store.state.pop("pending_survey")
    store.save()
    return {"status": "observed", "summary": "Retained the requested world reads for planning."}


def recipe_reads(context: RoleContext, patterns: list[str]) -> list[dict]:
    """Retain actual read-only recipe lookups at serialized command boundaries.

    Check cancellation before each lookup. These planning facts cannot accept
    a leaf or claim that ingredients have been gathered or consumed.
    """
    reads = []
    for pattern in patterns:
        context.runtime.check_cancelled()
        arguments = {"pattern": pattern}
        with BODY_LEASES.command(context.runtime.tools["minecraft_endpoint"](), context.runtime.stop_event):
            observed = context.runtime.tools["minecraft_call"]("minecraft_recipe_search", arguments)
        reads.append({"tool": "minecraft_recipe_search", "arguments": arguments,
                      "time": time.time(), "ok": observed["ok"], "result": observed["result"]})
    return reads

