from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from pm_workflows.python_role import RoleContext
from pm.utils import yaml_util
from workflows.minecraft_project.tree_store import TreeStore
from workflows.minecraft_project.schemas import Node
from workflows.minecraft_project.planning.run_context import request
from workflows.minecraft_project.planning.context_contracts import DecisionIdentity, ProjectIntent, GameplayHistory, GameplayScope
from workflows.minecraft_project.history_encoding import gameplay_history, scoped_gameplay
from workflows.minecraft_body.call_outcomes import GameplayEvidence
from workflows.minecraft_body.journal import read_actor_journal


class PublicRequestAuthority(BaseModel):
    model_config = ConfigDict(extra="ignore")
    original_request: str = Field(default="", validation_alias="_public_original_request")


def decision_identity(context: RoleContext, store: TreeStore, node: Node | None = None,
                      actor_attempt: str = "", verifier_attempt: str = "") -> DecisionIdentity:
    return DecisionIdentity(task_id=context.task_id, tree_revision=store.state["revision"],
        leaf_id=node.id if node is not None else "", spec_revision=node.revision if node is not None else 0,
        actor_attempt=actor_attempt, verifier_attempt=verifier_attempt)


def decision_intent(context: RoleContext, store: TreeStore, instruction: str = "") -> ProjectIntent:
    runtime = context.runtime
    if runtime is None:
        raise RuntimeError("Project intent requires its workflow runtime.")
    authority = PublicRequestAuthority.model_validate(yaml_util.safe_load(runtime.environment.read_text("inbox/request.yaml")))
    req = request(context)
    return ProjectIntent(original_request=authority.original_request or req.goal or req.instruction,
        original_goal=req.goal, constraints=tuple(req.constraints),
        current_instruction=instruction or store.state.get("goal", req.instruction))


def read_gameplay_history(store: TreeStore) -> GameplayHistory:
    return GameplayHistory.model_validate(gameplay_history(store))


def scoped_planning_history(store: TreeStore, selected: set[str], full: GameplayHistory,
                            targets: set[str] | None = None) -> GameplayHistory:
    evidence = GameplayEvidence()
    journals: list[str] = []
    methods = {key for key in selected if not store.valid(key)} | (targets or set())
    for attempt in store.state["attempts"]:
        if attempt["leaf_id"] not in methods:
            continue
        relative = attempt.get("journal", attempt["path"] + "/scratch/actor-journal.jsonl")
        journals.append(relative)
        for row in read_actor_journal(store.path_for(relative)):
            evidence.consume(row, relative)
    result = GameplayHistory.model_validate(scoped_gameplay(evidence.snapshot()))
    result.recovery = full.recovery.model_copy(deep=True) if full.recovery is not None else None
    result.service_wait = full.service_wait.model_copy(deep=True) if full.service_wait is not None else None
    result.life = full.life.model_copy(deep=True) if full.life is not None else None
    result.failed_recoveries = tuple(failure.model_copy(deep=True) for failure in full.failed_recoveries)
    result.scope = GameplayScope(node_ids=tuple(sorted(selected)), method_node_ids=tuple(sorted(methods)),
        journals=tuple(journals), meaning="Selected task methods; original global journals remain available through retrieval.")
    return result
