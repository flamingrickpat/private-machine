"""Shared helpers of the project roles: request, tree store, decision identity, and instruction boundary."""
from __future__ import annotations

from pathlib import Path
from typing import TypeVar

from pm.utils import yaml_util
from pm_workflows.python_role import RoleContext
from pydantic import BaseModel

from workflows.minecraft_project.schemas import Node, ProjectRequest
from workflows.minecraft_project.tree_store import TreeStore, digest
from pm.paths import REPO_ROOT


ROOT = REPO_ROOT


RESOURCES = Path(__file__).resolve().parents[1] / "resources"


Decision = TypeVar("Decision", bound=BaseModel)


class InstructionsChanged(RuntimeError):
    """A leaf became stale before its next action. The world remains intact."""


def request(context: RoleContext) -> ProjectRequest:
    """Read a scoped project request without public transport metadata."""
    payload = yaml_util.safe_load(context.runtime.environment.read_text("inbox/request.yaml"))
    return ProjectRequest.model_validate({key: value for key, value in payload.items() if not key.startswith("_public_")})


def project_intent(context: RoleContext, store: TreeStore, instruction: str = "") -> dict:
    """Keep the original request and explicit constraints beside current work.

    A newer instruction overrides its named changes. It does not erase other
    original requirements. This context describes intent, not completed work.
    """
    payload = yaml_util.safe_load(context.runtime.environment.read_text("inbox/request.yaml"))
    req = request(context)
    return {"original_request": payload.get("_public_original_request") or req.goal or req.instruction,
            "original_goal": req.goal, "constraints": req.constraints,
            "current_instruction": instruction or store.state.get("goal", req.instruction)}


def store_for(context: RoleContext) -> TreeStore:
    """Use the working target's task folder as the sole durable tree owner."""
    return TreeStore(context.runtime.environment._path("."))


def identity(context: RoleContext, store: TreeStore, *, node: Node | None = None, actor_attempt: str = "", verifier_attempt: str = "") -> dict:
    """Bind every decision to the task, specification, and exact attempts."""
    return {"task_id": context.task_id, "tree_revision": store.state["revision"],
            "leaf_id": node.id if node else "", "spec_revision": node.revision if node else 0,
            "actor_attempt": actor_attempt, "verifier_attempt": verifier_attempt}


def boundary(context: RoleContext, store: TreeStore) -> bool:
    """Detect a newer project instruction before another role accepts work."""
    current = context.runtime.tools["managed_instructions"]()
    return current["revision"] == store.state["instruction_revision"]


def progress_facts(observation: dict) -> dict:
    """Exclude clock and view changes from the repeated-yield guard."""
    player = observation.get("player", {})
    return {"inventory": observation.get("inventory", []),
            "position": observation.get("camera", {}).get("feet_position"),
            "player": {key: player.get(key) for key in ("health", "food", "oxygen", "dimension")}}


def scoped_reads(reads: list[dict]) -> list[dict]:
    """Retain exact evidence and distinguish known air from an unavailable read."""
    from workflows.minecraft_body.gateway import observed_target
    scoped = []
    for row in reads:
        data = row["result"]
        relevant = {key: data[key] for key in ("position", "block_name", "is_solid", "properties", "collision_shapes", "reason", "blocks", "total_cells", "unseen_cells",
                                              "node_id", "verified", "certificate_digest", "verification",
                                              "source_specification", "source_identity", "source_checks", "source_observed_times",
                                              "pattern", "matches", "total_recipes") if key in data}
        if row["tool"] == "minecraft_observe":
            relevant.update(inventory=data.get("inventory", []), camera=data.get("camera", {}), player=data.get("player", {}))
            relevant.update({key: data[key] for key in ("body_session", "death_count") if key in data})
        available = (observed_target(row, row["arguments"]["position"])
                     if row["tool"] == "minecraft_inspect_block" else bool(row["ok"]))
        scoped.append({**{key: value for key, value in row.items() if key != "result"},
                       "observation_available": available,
                       "result": relevant, "original_digest": digest(data)})
    return scoped

