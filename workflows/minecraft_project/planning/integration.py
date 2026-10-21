"""Finish role: integrate all certificates, run the project check, and publish the report."""
from __future__ import annotations

import json

from pm.utils import yaml_util
from pm_workflows.python_role import RoleContext

from workflows.minecraft_body.lease import BODY_LEASES
from workflows.minecraft_project.tree_store import TreeStore
from workflows.minecraft_project.acceptance import evaluate, observe_conditions, verified_project
from workflows.minecraft_project.planning.run_context import boundary, identity, request
from workflows.minecraft_project.history_encoding import gameplay_history


def finish(context: RoleContext, store: TreeStore) -> dict:
    """Verify aggregate usability and close instructions before reporting success."""
    req = request(context)
    if not boundary(context, store):
        return {"status": "replan", "summary": "Project instructions changed before final verification."}
    if not req.plan_only:
        if not all(store.valid(node.id) for node in store.load().nodes):
            raise ValueError("Final publication requires verified leaf, story, and goal certificates.")
        tree = store.load()
        root = next(node for node in tree.nodes if node.id == tree.goal_id)
        reads = observe_conditions(context, root.criteria)
        checks = evaluate(root.criteria, reads, certificates=store.state["acceptances"])
        original = json.loads(store.path_for(store.state["acceptances"][root.id]["verification"]).read_text(encoding="utf-8"))
        original_lives = {(row["result"].get("body_session"), row["result"].get("death_count"))
                          for row in original["reads"] if row["tool"] == "minecraft_observe"}
        current_lives = {(row["result"].get("body_session"), row["result"].get("death_count"))
                         for row in reads if row["tool"] == "minecraft_observe"}
        integration = {"identity": identity(context, store, node=root), "reads": reads, "checks": checks,
                       "same_life": not original_lives or original_lives == current_lives}
        relative = f"verification/integration-{len(store.state['decisions']):04d}.json"
        store.write(relative, integration)
        if not all(checks.values()) or not integration["same_life"]:
            return {"status": "replan", "summary": "Fresh final integration differs from the accepted root. Original reads: " + relative}
        check = context.runtime.tools.get("minecraft_project_check")
        if check is not None:
            result = check(store.root, store.load())
            number = store.state.get("project_checks", 0) + 1
            relative = f"verification/project-check-{number:04d}.json"
            store.write(relative, {"identity": identity(context, store), "result": result})
            store.state.update(project_checks=number, project_check=relative)
            store.save()
            if result["errors"]:
                return {"status": "replan", "summary": "Project completion evidence needs a concrete leaf repair: " + " ".join(result["errors"]) + " Original evidence: " + relative}
    final = context.runtime.tools["managed_instructions"](close=True)
    if final["revision"] != store.state["instruction_revision"]:
        return {"status": "replan", "summary": "The final instruction boundary changed the project."}
    evidence = verified_project(store.root) if not req.plan_only else {"verified": False}
    if not req.plan_only and not evidence["verified"]:
        raise ValueError("The completed project failed retained-evidence recomputation.")
    if not req.plan_only:
        from workflows.minecraft.steps import build_world
        with BODY_LEASES.command(context.runtime.tools["minecraft_endpoint"](), context.runtime.stop_event):
            observation = context.runtime.tools["minecraft_call"]("minecraft_observe", {"include_image": False})
        if not observation["ok"]:
            raise ValueError("The project has no usable final body observation.")
        req.goal = store.state["goal"]
        world = build_world(req, observation, None, [])
        world["outcome"] = world["episodes"][0]["outcome"] = "success"
        world["verification"] = evidence
        world["episodes"][0]["source_attempts"] = [row["id"] for row in store.state["attempts"]]
        world["episodes"][0]["gameplay"] = gameplay_history(store)
        context.runtime.environment.write_text("outbox/world.yaml", yaml_util.safe_dump(world, allow_unicode=True, sort_keys=False))
    report = {"schema": "pm.minecraft-project-report.v1", "accepted": True, "verified": evidence["verified"],
              "summary": "Reviewed plan retained without execution." if req.plan_only else "The project passed independent world verification.",
              "goal": store.state["goal"], "tree_revision": store.state["revision"], "verification": evidence,
              "plan_only": req.plan_only, "attempts": store.state["attempts"], "body_id": req.body_id}
    context.runtime.environment.write_text("outbox/report.yaml", yaml_util.safe_dump(report, allow_unicode=True, sort_keys=False))
    return {"status": "done", "summary": report["summary"]}

