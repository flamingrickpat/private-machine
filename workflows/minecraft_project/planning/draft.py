from __future__ import annotations
from workflows.minecraft.runtime import require_minecraft_runtime

from dataclasses import dataclass
from typing import Literal
from fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, ValidationError
from pm_workflows.runtime import WorkflowRuntime
from pm.runs.draft_session import DraftValidation
from workflows.minecraft_project.schemas import Condition, PlanProposal, Point, TaskTree
from workflows.minecraft_project.tree_store import TreeStore, node_digest, validate_tree
from workflows.minecraft_project.evidence_selection import protect_accepted_specifications
from workflows.minecraft_project.planning.run_context import InstructionsChanged
from workflows.minecraft_project.planning.draft_contracts import (
    DraftNode, NodeEdit, TaskAddRequest, ConditionSetRequest,
    ConditionCopyRequest, PlanOutcomeRequest, DraftToolResult, DraftSnapshot)


class InstructionRevision(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    revision: int


@dataclass(frozen=True, slots=True)
class PlanningBoundary:
    runtime: WorkflowRuntime
    instruction_revision: int

    def check_current(self) -> None:
        self.runtime.check_cancelled()
        instruction = InstructionRevision.model_validate(require_minecraft_runtime(self.runtime).instructions())
        if instruction.revision != self.instruction_revision:
            raise InstructionsChanged("Project instructions changed during planning. Retain effects and read the new request.")


@dataclass(frozen=True, slots=True)
class DraftSettings:
    base: TaskTree | None = None
    scope: frozenset[str] | None = None
    refinement_target: str = ""
    require_leaf_change: bool = False


class TaskDraft:
    def __init__(self, store: TreeStore, goal: str, artifact_path: str, boundary: PlanningBoundary,
                 settings: DraftSettings = DraftSettings()) -> None:
        self.store = store
        self.artifact_path = artifact_path
        self.boundary = boundary
        self.base = settings.base
        self.scope = settings.scope
        self.refinement_target = settings.refinement_target
        self.require_leaf_change = settings.require_leaf_change
        self.goal_id = settings.base.goal_id if settings.base is not None else "goal"
        self.nodes: dict[str, DraftNode] = ({node.id: DraftNode.from_node(node) for node in settings.base.nodes}
            if settings.base is not None else {"goal": DraftNode(id="goal", kind="goal", instruction=goal, specialist="builder")})
        self.counter = len(self.nodes)
        self.status: Literal["planned", "survey", "waiting", "blocked"] = "planned"
        self.summary = ""
        self.queries: list[Condition] = []
        self.recipe_queries: list[str] = []
        self.restore_nodes: list[str] = []
        self._save()

    def check_cancelled(self) -> None:
        self.boundary.check_current()

    def _save(self) -> None:
        self.check_cancelled()
        snapshot = DraftSnapshot(goal_id=self.goal_id, nodes=tuple(self.nodes.values()), status=self.status,
            summary=self.summary, queries=tuple(self.queries), recipe_queries=tuple(self.recipe_queries),
            restore_nodes=tuple(self.restore_nodes))
        self.store.write(self.artifact_path, snapshot)

    def _node(self, node_id: str) -> DraftNode:
        if node_id not in self.nodes:
            raise ValueError(f"Unknown task {node_id!r}. Use task_list to read existing IDs.")
        return self.nodes[node_id]

    def task_list(self) -> DraftSnapshot:
        self.check_cancelled()
        return DraftSnapshot(goal_id=self.goal_id, nodes=tuple(self.nodes.values()), status=self.status,
            summary=self.summary, queries=tuple(self.queries), recipe_queries=tuple(self.recipe_queries),
            restore_nodes=tuple(self.restore_nodes))

    def task_get(self, node_id: str) -> DraftToolResult:
        self.check_cancelled()
        return DraftToolResult(task=self._node(node_id).model_copy(deep=True))

    def task_add(self, request: TaskAddRequest) -> DraftToolResult:
        self.check_cancelled()
        parent = request.parent or self.goal_id
        owner = self._node(parent)
        self.counter += 1
        node_id = f"task_{self.counter}"
        while node_id in self.nodes:
            self.counter += 1
            node_id = f"task_{self.counter}"
        self.nodes[node_id] = DraftNode(id=node_id, kind=request.kind, instruction=request.instruction,
            parent=parent, specialist=request.specialist, depends_on=list(request.depends_on))
        if owner.kind == "leaf":
            owner.kind = "story"
        owner.children.append(node_id)
        self._save()
        return DraftToolResult(node_id=node_id, parent=parent)

    def task_edit(self, node_id: str, fields: NodeEdit) -> DraftToolResult:
        self.check_cancelled()
        node = self._node(node_id)
        if any(value is None for value in fields.model_dump(exclude_unset=True).values()):
            raise ValueError("Editable task fields cannot be null.")
        if fields.kind is not None:
            node.kind = fields.kind
        if fields.depends_on is not None:
            node.depends_on = list(fields.depends_on)
        if fields.instruction is not None:
            node.instruction = fields.instruction
        if fields.specialist is not None:
            node.specialist = fields.specialist
        if fields.deferred is not None:
            node.deferred = fields.deferred
        if fields.defer_reason is not None:
            node.defer_reason = fields.defer_reason
        if fields.criteria is not None:
            node.criteria = [condition.model_copy(deep=True) for condition in fields.criteria]
        if fields.assumptions is not None:
            node.assumptions = [condition.model_copy(deep=True) for condition in fields.assumptions]
        self._save()
        return DraftToolResult(node_id=node_id, changed_fields=tuple(sorted(fields.model_fields_set)))

    def condition_set(self, request: ConditionSetRequest) -> DraftToolResult:
        self.check_cancelled()
        node = self._node(request.node_id)
        current = node.assumptions if request.assumption else node.criteria
        value = request.condition
        positions = list(value.positions)
        if value.box is not None:
            if value.kind != "block" or value.position is not None or value.positions:
                raise ValueError("A box supplies exact block positions. Do not combine it with other cell operands.")
            start, end = value.box.min, value.box.max
            for x in range(min(start.x, end.x), max(start.x, end.x) + 1):
                for y in range(min(start.y, end.y), max(start.y, end.y) + 1):
                    for z in range(min(start.z, end.z), max(start.z, end.z) + 1):
                        positions.append(Point(x=x, y=y, z=z))
        checked = Condition(id=value.id or f"condition_{len(current) + 1}",
            description=value.description or f"{value.kind} outcome for {request.node_id}.", kind=value.kind,
            node_id=value.node_id, names=list(value.names), minimum=value.minimum, position=value.position,
            positions=positions, end=value.end, radius=value.radius, dimension=value.dimension)
        remaining = [condition for condition in current if condition.id != checked.id]
        remaining.append(checked)
        if request.assumption:
            node.assumptions = remaining
        else:
            node.criteria = remaining
        self._save()
        return DraftToolResult(node_id=node.id, condition_id=checked.id, exact_cells=len(checked.positions))

    def condition_copy(self, request: ConditionCopyRequest) -> DraftToolResult:
        self.check_cancelled()
        source, target = self._node(request.source_id), self._node(request.target_id)
        selected = [condition for condition in source.criteria if not request.condition_ids or condition.id in request.condition_ids]
        copied_ids: list[str] = []
        for condition in selected:
            copied = condition.model_copy(deep=True)
            copied.id = request.source_id + "_" + condition.id
            target.criteria = [previous for previous in target.criteria if previous.id != copied.id]
            target.criteria.append(copied)
            copied_ids.append(copied.id)
        self._save()
        return DraftToolResult(node_id=target.id, copied_conditions=tuple(copied_ids))

    def plan_outcome(self, request: PlanOutcomeRequest) -> DraftToolResult:
        self.check_cancelled()
        self.status = request.status
        self.summary = request.summary
        self.queries = list(request.queries)
        self.recipe_queries = list(request.recipe_queries)
        self.restore_nodes = list(request.restore_nodes)
        self._save()
        return DraftToolResult()

    def plan_check(self) -> DraftValidation[PlanProposal]:
        return self.validate(self.summary or "Draft check.")

    def validate(self, response: str) -> DraftValidation[PlanProposal]:
        """Validate a turn against the actual tree and retained proof boundaries."""
        self.check_cancelled()
        try:
            if self.status != "planned":
                proposal = PlanProposal(status=self.status, summary=self.summary or response,
                    queries=self.queries, recipe_queries=self.recipe_queries)
                return DraftValidation(True, proposal)
            tree = TaskTree(goal_id=self.goal_id, nodes=[node.specification() for node in self.nodes.values()])
            validate_tree(tree)
            tree = self.store.restore_specifications(tree, self.restore_nodes)
            if self.base is not None:
                protect_accepted_specifications(self.store, tree)
            self.store.check_revision(tree, target=self.refinement_target)
            if self.base is not None:
                old = {node.id: node for node in self.base.nodes}
                for node in tree.nodes:
                    if node.id in old and node.parent != old[node.id].parent:
                        raise ValueError(f"Task {node.id} must retain its parent {old[node.id].parent!r}.")
                    if self.scope is not None and node.id in old and node.id not in self.scope and node_digest(node) != node_digest(old[node.id]):
                        raise ValueError(f"Task {node.id} belongs to another repair scope. Retain its exact fields.")
                if self.refinement_target:
                    prior = old[self.refinement_target]
                    changed = next(node for node in tree.nodes if node.id == self.refinement_target)
                    if prior.criteria != changed.criteria:
                        raise ValueError("Refinement retains the target's original criteria. Split its remaining work beneath the same ID.")
                if self.require_leaf_change and not any(node.kind == "leaf" and (
                        node.id not in old or node_digest(node) != node_digest(old[node.id])) for node in tree.nodes):
                    raise ValueError("Revise a concrete leaf after the failed method. An explanation alone cannot release another attempt.")
            if self.restore_nodes:
                return DraftValidation(True, PlanProposal(status="planned", summary=self.summary or response,
                    updates=[node for node in tree.nodes if node.id not in self.restore_nodes], restore_nodes=list(self.restore_nodes)))
            return DraftValidation(True, PlanProposal(status="planned", summary=self.summary or response, tree=tree))
        except ValidationError as error:
            return DraftValidation(False, feedback=str(error.errors(include_input=False, include_url=False, include_context=False)))
        except ValueError as error:
            return DraftValidation(False, feedback=str(error))

    def server(self) -> FastMCP:
        server = FastMCP("Minecraft task draft")
        server.tool(name="task_list")(self.task_list)
        server.tool(name="task_get")(self.task_get)
        server.tool(name="task_add")(self.task_add)
        server.tool(name="task_edit")(self.task_edit)
        server.tool(name="condition_set")(self.condition_set)
        server.tool(name="condition_copy")(self.condition_copy)
        server.tool(name="plan_outcome")(self.plan_outcome)
        server.tool(name="plan_check")(self.plan_check)
        return server
