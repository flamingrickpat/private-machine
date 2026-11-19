from __future__ import annotations

from dataclasses import dataclass
import json
from pydantic import JsonValue
from workflows.minecraft_project.tree_store import digest
from workflows.minecraft_project.schemas import Node
from workflows.minecraft_project.planning.context_contracts import (
    PlanningContext, DecisionContext, NodeReference, GameplayHistory, ArchivedMethodSummary, ArchivedDiagnosis, RecordedRead)


@dataclass(frozen=True, order=True, slots=True)
class MethodRecency:
    time: float
    index: int


def compact_plan_context(context: PlanningContext) -> PlanningContext:
    selected = context.model_copy(deep=True)
    if selected.latest_plan_review is not None and selected.feedback == selected.latest_plan_review.summary:
        selected.feedback = "See latest_plan_review.summary for the original review feedback."
    if selected.previous_site_survey is not None:
        survey = selected.previous_site_survey
        shared = [tool for tool in survey.tool_contracts if tool in selected.current_tool_contracts]
        if shared:
            survey.tool_contracts = tuple(tool for tool in survey.tool_contracts if tool not in shared)
            survey.shared_tool_contracts = "current_tool_contracts"
    return selected


def select_gameplay_history(history: GameplayHistory) -> GameplayHistory:
    selected = history.model_copy(deep=True)
    ranked = sorted(MethodRecency(method.last.time, index) for index, method in enumerate(history.methods))
    retained = {item.index for item in ranked[-32:]}
    retained.update(index for index, method in enumerate(history.methods) if method.ineffective >= 2)
    by_method = {(method.tool, json.dumps(method.arguments, sort_keys=True)): index
                 for index, method in enumerate(history.methods)}
    for outcome in history.recent_outcomes:
        key = (outcome.tool, json.dumps(outcome.arguments, sort_keys=True))
        if key in by_method:
            retained.add(by_method[key])
    archived: list[ArchivedMethodSummary] = []
    omitted_tools = sorted({method.tool for index, method in enumerate(history.methods) if index not in retained})
    for tool in omitted_tools:
        omitted = [method for index, method in enumerate(history.methods) if index not in retained and method.tool == tool]
        sequences: dict[str, list[int]] = {}
        reasons: dict[str, int] = {}
        quantities: dict[str, float] = {}
        for method in omitted:
            last = method.last
            source = last.source or str(last.source_index)
            sequences.setdefault(source, []).append(last.seq)
            if last.reason:
                reasons[last.reason] = reasons.get(last.reason, 0) + 1
            for resource, quantity in last.quantities.items():
                quantities[resource] = quantities.get(resource, 0) + quantity
        archived.append(ArchivedMethodSummary(tool=tool, method_count=len(omitted),
            source_sequences={source: tuple(values) for source, values in sequences.items()}, reasons=reasons, quantities=quantities))
    selected.methods = tuple(method for index, method in enumerate(history.methods) if index in retained)
    selected.archived_methods = (*history.archived_methods, *archived)
    selected.diagnoses = history.diagnoses[-4:]
    selected.archived_diagnoses = (*history.archived_diagnoses, *tuple(ArchivedDiagnosis(source=diagnosis.source or str(diagnosis.source_index),
        time=diagnosis.time, evidence_sequences=diagnosis.evidence_sequences) for diagnosis in history.diagnoses[:-4]))
    selected.decision_scope = ("Current methods retain exact operands and preconditions. Archived totals sum each older method's LAST result. "
        "Original journals retain exact outcomes. Current inventory and independent certificates take precedence.")
    return selected


def decision_context(context: DecisionContext) -> DecisionContext:
    if not isinstance(context, PlanningContext):
        return context.model_copy(deep=True)
    selected = compact_plan_context(context)
    survey = selected.previous_site_survey
    if survey is not None and selected.latest_attempt is not None and survey.original_receipt and survey.original_digest:
        survey.recipe_scope = "Recipes observed before the latest actor attempt are historical. Request fresh inventory after world effects."
        for recipe in survey.recipes:
            if recipe.time < selected.latest_attempt.started and recipe.original_digest:
                recipe.result = None
                recipe.result_scope = "Historical output remains in its original receipt."
    survey = selected.requested_world_survey
    if survey is not None and survey.identity.tree_revision < selected.identity.tree_revision and survey.original_receipt and survey.original_digest:
        survey.archived_read_count = len(survey.reads)
        survey.archived_recipe_count = len(survey.recipes)
        survey.reads = ()
        survey.recipes = ()
        survey.read_scope = "Historical survey from an earlier revision. Query operands remain explicit; use fresh world reads for current truth."
    proof = selected.latest_verification
    if proof is not None and proof.verified:
        proof.reads = ()
        proof.read_scope = "Past accepted attainment. Original reads remain in original_receipt; request fresh reads for current geometry."
        if isinstance(proof.specification, Node) and selected.previous_tree is not None and proof.specification in selected.previous_tree.nodes:
            proof.specification = NodeReference(from_node=proof.specification.id, collection="previous_tree")
    selected.gameplay_history = select_gameplay_history(selected.gameplay_history)
    return selected


def native_decision_context(context: PlanningContext) -> PlanningContext:
    selected = decision_context(context)
    assert isinstance(selected, PlanningContext)
    return selected


def select_recorded_reads(reads: tuple[RecordedRead, ...]) -> tuple[RecordedRead, ...]:
    selected: list[RecordedRead] = []
    for receipt in reads:
        result = receipt.result
        if not isinstance(result, dict):
            raise ValueError(f"World read {receipt.tool} must return a JSON object.")
        relevant: dict[str, JsonValue] = {}
        for key in ("position", "block_name", "is_solid", "properties", "collision_shapes", "reason", "blocks", "total_cells", "unseen_cells",
                    "node_id", "verified", "certificate_digest", "verification", "source_specification", "source_identity", "source_checks",
                    "source_observed_times", "pattern", "matches", "total_recipes"):
            if key in result:
                relevant[key] = result[key]
        if receipt.tool == "minecraft_observe":
            relevant["inventory"] = result.get("inventory", [])
            relevant["camera"] = result.get("camera", {})
            relevant["player"] = result.get("player", {})
            for key in ("body_session", "death_count", "held_item"):
                if key in result:
                    relevant[key] = result[key]
        observed = receipt.ok
        if receipt.tool == "minecraft_inspect_block":
            observed = observed or (result.get("reason") == "not_found" and result.get("block_name") == "air"
                and result.get("position") == receipt.arguments["position"])
        item = receipt.model_copy(deep=True)
        item.observation_available = observed
        item.original_digest = digest(result)
        item.result = relevant
        selected.append(item)
    return tuple(selected)
