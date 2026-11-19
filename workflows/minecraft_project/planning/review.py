"""Review role: independent review of one plan revision."""
from __future__ import annotations
from workflows.minecraft_project.schemas import ProjectPhaseResult
from workflows.minecraft.runtime import require_minecraft_runtime
from pm.runs.managed_boundary import BoundaryReadSettings

import json

from pm_workflows.python_role import RoleContext
from pm.runs.review import ReviewMode, PASSING_REVIEWS, review_passes, review_policy

from workflows.minecraft_project.schemas import PlanReview, TaskTree
from workflows.minecraft_project.tree_store import TreeStore, validate_tree
from workflows.minecraft_project.acceptance import observe_conditions
from workflows.minecraft_project.evidence_selection import accepted_inputs, dependency_scope, review_nodes
from workflows.minecraft_project.planning.run_context import boundary, identity, project_intent, request, scoped_reads
from workflows.minecraft_project.planning.context_selection import select_recorded_reads
from workflows.minecraft_project.planning.decisions import reason
from workflows.minecraft_project.planning.context_contracts import (ReviewContext, ReviewPolicy, RecordedRead, SurveyEvidence,
    NodeTopology, ReviewScope, AcceptedInput, PrecedingReview, ProjectCheck, DecisionRole)
from workflows.minecraft_project.planning.context_sources import decision_identity, decision_intent, read_gameplay_history
from workflows.minecraft_body.planning_prompt import MinecraftActionTool
from workflows.minecraft_project.history_encoding import gameplay_history
from workflows.minecraft_project.planning.proposals import survey_conditions
from workflows.minecraft_project.planning.survey import recipe_reads
from workflows.minecraft_project.planning.constraints import caller_plan_constraints, project_check_context, record_plan_constraints


def review(context: RoleContext, store: TreeStore) -> ProjectPhaseResult:
    """Review acceptance coverage independently before selecting any leaf."""
    if not boundary(context, store):
        return ProjectPhaseResult(status="refine", summary="Project instructions changed before plan review.")
    tree = store.load()
    mode = request(context).review_mode
    result = record_plan_constraints(context, store)
    if result is not None:
        if result.errors:
            store.state.pop("reviewed_revision", None)
            rejected = PlanReview(status="revise", summary="The configured project contract requires repair: " + " ".join(result.errors))
            record_review(store, rejected, mode=mode)
            return ProjectPhaseResult(status=rejected.status, summary=rejected.summary)
        store.save()
    ordered = review_nodes(store)
    survey_path = f"planning/surveys/{len(store.state['decisions']) + 1:04d}.json"
    survey_reads = observe_conditions(context, survey_conditions(ordered))
    names = sorted({name.removeprefix("minecraft:") for node in ordered for condition in node.criteria
        if condition.kind == "inventory" for name in condition.names if not any(c in name for c in "*?[" )})
    recipes = recipe_reads(context, names)
    tool_contracts = tuple(row for row in require_minecraft_runtime(context.runtime).minecraft_transport.tool_schemas()
        if row.name in {"minecraft_recipe_search", "minecraft_smelt_item", "minecraft_eat_best"})
    survey = SurveyEvidence(identity=decision_identity(context, store),
        reads=select_recorded_reads(tuple(RecordedRead.model_validate(row) for row in survey_reads)),
        recipes=select_recorded_reads(tuple(RecordedRead.model_validate(row) for row in recipes)), tool_contracts=tool_contracts)
    store.write(survey_path, survey)
    store.state["planning_survey"] = survey_path
    store.save()
    policy = ReviewPolicy.model_validate(review_policy(mode))
    intent = decision_intent(context, store)
    plan_constraints = caller_plan_constraints(store)
    check = project_check_context(store)
    latest_check = ProjectCheck.model_validate(check) if check is not None else None
    server_rules = (json.loads(store.path_for(store.state["server_rules"]).read_text(encoding="utf-8")).get("server_rules", "")
        if store.state.get("server_rules") else "")
    prerequisites = {key: tuple(sorted(value)) for key, value in validate_tree(tree).items()}
    initial_observation = json.loads(store.path_for(store.state["initial_observation"]).read_text(encoding="utf-8"))
    planning_observation = json.loads(store.path_for(store.state["planning_observation"]).read_text(encoding="utf-8"))
    topology = tuple(NodeTopology(id=node.id, parent=node.parent, children=tuple(node.children),
        depends_on=tuple(node.depends_on), kind=node.kind) for node in tree.nodes)
    historical = accepted_inputs(store, {node.id for node in tree.nodes})
    reviews = [PrecedingReview(node_id=key, status="accepted", scope="Original independently verified historical attainment.",
        original_receipt=value["certificate"]["verification"]) for key, value in historical.items()]
    for node in ordered:
        if not boundary(context, store):
            return ProjectPhaseResult(status="refine", summary="Project instructions changed during scoped plan review.")
        selected = dependency_scope(tree, [node.id])
        accepted = {key: AcceptedInput.model_validate(value) for key, value in accepted_inputs(store, selected).items()}
        current_survey = SurveyEvidence(identity=survey.identity,
            reads=select_recorded_reads(tuple(RecordedRead.model_validate(row) for row in observe_conditions(context, survey_conditions([node])))),
            recipes=survey.recipes, tool_contracts=survey.tool_contracts)
        scoped = ReviewContext(identity=decision_identity(context, store), goal=store.state["goal"],
            tree=TaskTree(goal_id=tree.goal_id, revision=tree.revision, nodes=[node]), project_intent=intent,
            review_policy=policy, plan_constraints=plan_constraints, latest_project_check=latest_check,
            server_rules=server_rules, leaf_prerequisites=prerequisites, accepted_nodes=tuple(store.state["acceptances"]),
            active_recovery=read_gameplay_history(store).recovery, project_initial_observation=initial_observation,
            original_observation=planning_observation, observation_path=store.state["planning_observation"],
            current_site_survey=current_survey, survey_path=survey_path, tree_topology=topology,
            review_scope=ReviewScope(node_id=node.id, kind=node.kind),
            dependency_specifications=tuple(row for row in tree.nodes if row.id in selected and row.id != node.id and row.kind == "leaf" and not store.valid(row.id)),
            accepted_dependencies=accepted, preceding_reviews=tuple(reviews))
        result = reason(context, store, DecisionRole.Reviewer, PlanReview, scoped)
        if not boundary(context, store):
            return ProjectPhaseResult(status="refine", summary="Project instructions changed during scoped plan review.")
        reviews.append(PrecedingReview(node_id=node.id, status=result.status, summary=result.summary, target_id=result.target_id,
            original_receipt=store.state["decisions"][-1]["path"] + "/output.json"))
        if not review_passes(result.status, mode):
            if result.status in PASSING_REVIEWS:
                result = PlanReview(status="revise", target_id=node.id,
                    summary=f"The {mode} review mode requires an accepted judgment. Received {result.status}: {result.summary}")
            if not result.target_id:
                result.target_id = node.id
            store.state["review_target"] = result.target_id
            break
    if not boundary(context, store):
        return ProjectPhaseResult(status="refine", summary="Project instructions changed during plan review.")
    record_review(store, result, mode=mode)
    return ProjectPhaseResult(status="accepted" if review_passes(result.status, mode) else result.status, summary=result.summary)


def record_review(store: TreeStore, result: PlanReview, *, mode: ReviewMode = "flexible") -> None:
    """Keep every review outcome available to later explicit recovery."""
    if result.status == "refine":
        store.state.pop("reviewed_revision", None)
        if result.target_id not in {node.id for node in store.load().nodes}:
            raise ValueError("The reviewer named an unknown refinement target.")
        store.state.update(feedback=result.summary, refinement_target=result.target_id)
    elif review_passes(result.status, mode):
        store.state.update(reviewed_revision=store.state["revision"], feedback="", refinement_target="",
                           review_judgment=result.status, review_mode=mode)
        store.state.pop("review_target", None)
    else:
        store.state.pop("reviewed_revision", None)
        store.state.update(feedback=result.summary, refinement_target="")
    store.save()

