"""Edit a private task draft through small MCP operations."""
from __future__ import annotations

import copy
from itertools import product
from typing import Callable, Literal

from fastmcp import FastMCP
from pydantic import create_model, ValidationError

from pm.runs.draft_session import DraftValidation
from workflows.minecraft_project.schemas import Condition, Contract, Node, PlanProposal, Point, TaskTree
from workflows.minecraft_project.tree_store import TreeStore, node_digest, validate_tree
from workflows.minecraft_project.evidence_selection import protect_accepted_specifications


class CellBox(Contract):
    """Two inclusive corners of an explicitly requested block surface."""

    min: Point
    max: Point


ConditionInput = create_model("ConditionInput", __base__=Contract, **{
    **{name: (field.annotation, copy.deepcopy(field)) for name, field in Condition.model_fields.items()},
    "id": (str, ""), "description": (str, ""), "box": (CellBox | None, None)})
ConditionInput.__doc__ = "A condition edit with backend-owned defaults and an optional exact block box."


class TaskDraft:
    """Own uncommitted edits; the existing project store owns accepted plans.

    Each tool saves the draft. New IDs and reciprocal child references come
    from this service. Existing fields stay exact unless an edit names them.
    Failed validation retains edits without committing a specification or proof.
    """

    def __init__(self, store: TreeStore, goal: str, save: Callable[[dict], None],
                 check_cancelled: Callable[[], None], *, base: TaskTree | None = None,
                 scope: set[str] | None = None, refinement_target: str = ""):
        self.store, self.save, self.check_cancelled = store, save, check_cancelled
        self.base = base
        self.scope, self.refinement_target = scope, refinement_target
        self.goal_id = base.goal_id if base else "goal"
        self.nodes = {node.id: node.model_dump(mode="json") for node in base.nodes} if base else {
            "goal": {"id": "goal", "kind": "goal", "parent": None, "children": [], "depends_on": [],
                     "instruction": goal, "criteria": [], "assumptions": [], "specialist": "builder"}}
        self.counter, self.status, self.summary = len(self.nodes), "planned", ""
        self.queries, self.recipe_queries, self.restore_nodes = [], [], []
        self._save()

    def _save(self) -> None:
        self.check_cancelled()
        self.save({"goal_id": self.goal_id, "nodes": list(self.nodes.values()), "status": self.status,
                   "summary": self.summary, "queries": self.queries, "recipe_queries": self.recipe_queries,
                   "restore_nodes": self.restore_nodes})

    def _node(self, node_id: str) -> dict:
        if node_id not in self.nodes:
            raise ValueError(f"Unknown task {node_id!r}. Use task_list to read existing IDs.")
        return self.nodes[node_id]

    def _call(self, operation: Callable[[], dict]) -> dict:
        self.check_cancelled()
        try:
            result = operation()
        except ValidationError as error:
            return {"ok": False, "errors": error.errors(include_input=False, include_url=False, include_context=False)}
        except ValueError as error:
            return {"ok": False, "error": str(error)}
        self._save()
        return {"ok": True, **result}

    def add(self, kind: str, instruction: str, parent: str, specialist: str, depends_on: list[str]) -> dict:
        """Allocate a task ID and attach it to its named parent."""
        owner = self._node(parent)
        self.counter += 1
        node_id = f"task_{self.counter}"
        while node_id in self.nodes:
            self.counter += 1
            node_id = f"task_{self.counter}"
        self.nodes[node_id] = {"id": node_id, "kind": kind, "parent": parent, "children": [],
            "depends_on": list(depends_on), "instruction": instruction, "specialist": specialist,
            "criteria": [], "assumptions": []}
        if owner["kind"] == "leaf":
            owner["kind"] = "story"
        owner["children"].append(node_id)
        return {"node_id": node_id, "parent": parent}

    def edit(self, node_id: str, fields: dict) -> dict:
        """Patch named fields without copying the node's other specifications."""
        allowed = set(Node.model_fields) - {"id", "revision", "parent", "children"}
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"These fields belong to the backend: {sorted(unknown)}. Editable fields: {sorted(allowed)}.")
        self._node(node_id).update(copy.deepcopy(fields))
        return {"node_id": node_id, "changed_fields": sorted(fields)}

    def condition(self, node_id: str, condition: dict, assumption: bool = False) -> dict:
        """Set one condition; expand an explicit block box into exact cells."""
        node = self._node(node_id)
        field = "assumptions" if assumption else "criteria"
        value = copy.deepcopy(condition)
        value.setdefault("id", f"condition_{len(node[field]) + 1}")
        value.setdefault("description", f"{value.get('kind', 'condition')} outcome for {node_id}.")
        box = value.pop("box", None)
        if box is not None:
            if value.get("kind") != "block" or value.get("position") or value.get("positions"):
                raise ValueError("A box supplies exact block positions. Do not combine it with other cell operands.")
            corners = CellBox.model_validate(box)
            start, end = corners.min, corners.max
            value["positions"] = [{"x": x, "y": y, "z": z} for x, y, z in product(*(
                range(min(getattr(start, axis), getattr(end, axis)), max(getattr(start, axis), getattr(end, axis)) + 1)
                for axis in ("x", "y", "z")))]
        checked = Condition.model_validate(value).model_dump(mode="json")
        node[field] = [row for row in node[field] if row.get("id") != checked["id"]] + [checked]
        return {"node_id": node_id, "condition_id": checked["id"], "exact_cells": len(checked["positions"])}

    def copy_conditions(self, target_id: str, source_id: str, condition_ids: list[str]) -> dict:
        """Copy selected exact conditions without asking the model to rewrite them."""
        source, target = self._node(source_id), self._node(target_id)
        selected = [row for row in source["criteria"] if not condition_ids or row["id"] in condition_ids]
        for row in selected:
            copied = copy.deepcopy(row)
            copied["id"] = source_id + "_" + row["id"]
            target["criteria"] = [old for old in target["criteria"] if old["id"] != copied["id"]] + [copied]
        return {"node_id": target_id, "copied_conditions": [source_id + "_" + row["id"] for row in selected]}

    def validate(self, response: str) -> DraftValidation[PlanProposal]:
        """Validate a turn against the actual tree and retained proof boundaries."""
        self.check_cancelled()
        try:
            if self.status != "planned":
                proposal = PlanProposal(status=self.status, summary=self.summary or response,
                    queries=self.queries, recipe_queries=self.recipe_queries)
                return DraftValidation(True, proposal)
            tree = TaskTree(goal_id=self.goal_id, nodes=list(self.nodes.values()))
            validate_tree(tree)
            tree = self.store.restore_specifications(tree, self.restore_nodes)
            if self.store.state["revision"]:
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
                if self.store.state.get("require_leaf_change") and not any(node.kind == "leaf" and (
                        node.id not in old or node_digest(node) != node_digest(old[node.id])) for node in tree.nodes):
                    raise ValueError("Revise a concrete leaf after the failed method. An explanation alone cannot release another attempt.")
            values = ({"updates": [node for node in tree.nodes if node.id not in self.restore_nodes],
                       "restore_nodes": self.restore_nodes} if self.restore_nodes else {"tree": tree})
            return DraftValidation(True, PlanProposal(status="planned", summary=self.summary or response, **values))
        except ValidationError as error:
            return DraftValidation(False, feedback=str(error.errors(include_input=False, include_url=False, include_context=False)))
        except ValueError as error:
            return DraftValidation(False, feedback=str(error))

    def server(self) -> FastMCP:
        """Expose draft operations; none accepts work or changes the game world."""
        server = FastMCP("Minecraft task draft")

        @server.tool()
        def task_list() -> dict:
            """List task IDs, hierarchy, instructions, and condition IDs."""
            return {"goal_id": self.goal_id, "tasks": [{key: node.get(key) for key in
                ("id", "kind", "parent", "children", "depends_on", "instruction")} | {
                "condition_ids": [row.get("id") for row in node["criteria"]]} for node in self.nodes.values()]}

        @server.tool()
        def task_get(node_id: str) -> dict:
            """Read one complete current task before editing it."""
            return self._call(lambda: {"task": copy.deepcopy(self._node(node_id))})

        @server.tool()
        def task_add(kind: Literal["story", "leaf"], instruction: str, parent: str = "",
                     specialist: Literal["gatherer", "navigator", "recovery", "builder", "miner"] = "gatherer",
                     depends_on: list[str] = []) -> dict:
            """Add an outcome. The backend assigns its ID and parent child link."""
            return self._call(lambda: self.add(kind, instruction, parent or self.goal_id, specialist, depends_on))

        @server.tool()
        def task_edit(node_id: str, fields: dict) -> dict:
            """Change only named fields, such as instruction, depends_on, or specialist."""
            return self._call(lambda: self.edit(node_id, fields))

        @server.tool()
        def condition_set(node_id: str, condition: ConditionInput, assumption: bool = False) -> dict:
            """Set one criterion or assumption. Omit id and description to use defaults.

            For a rectangular block surface, box={min:{x,y,z},max:{x,y,z}} expands
            every inclusive cell. Other operands use the normal Condition schema.
            """
            return self._call(lambda: self.condition(node_id, condition.model_dump(exclude_defaults=True), assumption))

        @server.tool()
        def condition_copy(target_id: str, source_id: str, condition_ids: list[str] = []) -> dict:
            """Copy exact selected criteria into an aggregate task. Empty IDs copies all."""
            return self._call(lambda: self.copy_conditions(target_id, source_id, condition_ids))

        @server.tool()
        def plan_outcome(status: Literal["planned", "survey", "waiting", "blocked"], summary: str,
                         queries: list[dict] = [], recipe_queries: list[str] = [], restore_nodes: list[str] = []) -> dict:
            """Choose a plan, read-only survey, external question, blocker, or proof restoration."""
            self.status, self.summary = status, summary
            self.queries, self.recipe_queries, self.restore_nodes = queries, recipe_queries, restore_nodes
            self._save()
            return {"ok": True, "status": status}

        @server.tool()
        def plan_check() -> dict:
            """Read deterministic validation errors without ending this conversation."""
            decision = self.validate(self.summary or "Draft check.")
            return {"accepted": decision.accepted, "feedback": decision.feedback}

        return server
