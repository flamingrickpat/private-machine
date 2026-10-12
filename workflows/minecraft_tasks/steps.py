"""Declared planner, selector, specialist, and verifier workflow roles."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
from typing import TypeVar

import yaml
from pm_workflows.python_role import RoleContext
from pydantic import BaseModel, ValidationError

from runtime.body_lease import BODY_LEASES
from pm.host.cancellation import LinkedStopEvent
from runtime.minecraft_actor import read_actor_journal
from runtime.minecraft_outcomes import GameplayEvidence, signature
from pm.runs.workflow_runner import WorkflowRequest, WorkflowRunner, WorkflowStatus
from pm.host.folder_target import FolderEnvironment
from workflows.minecraft.schemas import MinecraftRequest
from workflows.minecraft_tasks.schemas import (
    Condition, Node, PlanProposal, PlanReview, ProjectRequest, RefinementProposal, RetentionChoice, RevisionProposal, TaskTree, VerificationReview,
)
from workflows.minecraft_tasks.tree import TreeStore, digest, validate_tree
from workflows.minecraft_tasks.verification import evaluate, observe_conditions, verified_project
from workflows.minecraft_tasks.evidence import accepted_inputs, catalog, dependency_scope, protect_accepted_specifications, read_source, retention_operation, retention_tree, review_nodes, scoped_history

ROOT = Path(__file__).resolve().parents[2]
RESOURCES = Path(__file__).with_name("resources")
Decision = TypeVar("Decision", bound=BaseModel)


class InstructionsChanged(RuntimeError):
    """A leaf became stale before its next action. The world remains intact."""


def request(context: RoleContext) -> ProjectRequest:
    """Read a scoped project request without public transport metadata."""
    payload = yaml.safe_load(context.runtime.environment.read_text("inbox/request.yaml"))
    return ProjectRequest.model_validate({key: value for key, value in payload.items() if not key.startswith("_public_")})


def store_for(context: RoleContext) -> TreeStore:
    """Use the working target's task folder as the sole durable tree owner."""
    return TreeStore(context.runtime.environment._path("."))


def identity(context: RoleContext, store: TreeStore, *, node: Node | None = None, actor_attempt: str = "", verifier_attempt: str = "") -> dict:
    """Bind every decision to the task, specification, and exact attempts."""
    return {"task_id": context.task_id, "tree_revision": store.state["revision"],
            "leaf_id": node.id if node else "", "spec_revision": node.revision if node else 0,
            "actor_attempt": actor_attempt, "verifier_attempt": verifier_attempt}


def decision_schema(contract: type[BaseModel]) -> dict:
    """Require explicit node operands in new model-generated replacements.

    Persistent readers still accept older omitted defaults. A new complete
    node replacement must state its parent, children, prerequisites, and
    conditions; omission must not silently erase those retained operands.
    This returns a fresh schema and does not change the stored contracts.
    """
    schema = contract.model_json_schema()
    node = schema.get("$defs", {}).get("Node")
    if node is not None:
        node["required"] = [key for key in node["properties"] if key != "revision"]
    return schema


def reason(context: RoleContext, store: TreeStore, role: str, contract: type[Decision], payload: dict) -> Decision:
    """Retrieve declared original evidence before accepting a scoped decision.

    Every retrieval response remains beside its native inference receipt.
    Repeated identical requests fail instead of silently looping or inventing
    evidence. An instruction change prevents publication after retrieval.
    """
    sources = catalog(context, store)
    relative = f"planning/context/{len(store.state['decisions']) + 1:04d}-catalog.json"
    store.write(relative, sources)
    catalog_digest = digest(sources)
    index = store.path_for(relative)
    sources["evidence_index"] = {"path": str(index.resolve()), "sha256": hashlib.sha256(index.read_bytes()).hexdigest()}
    groups = {}
    for name in sources:
        group = name.split('/')[0]
        groups[group] = groups.get(group, 0) + 1
    payload = {**payload, "evidence_catalog": {"index_source": "evidence_index", "source_groups": groups,
                    "named_sources": [name for name in sources if '/' not in name]},
               "evidence_catalog_receipt": {"original_receipt": relative, "original_digest": catalog_digest}}
    seen = set()
    while True:
        result = reason_once(context, store, role, contract, payload)
        requests = getattr(result, "evidence_requests", [])
        if not requests:
            return result
        reads = []
        for query in requests:
            key = digest(query.model_dump())
            if key in seen:
                raise ValueError("The decision repeated an identical evidence request without progress.")
            seen.add(key)
            reads.append(read_source(sources, query))
        relative = store.state["decisions"][-1]["path"] + "/retrieved.json"
        store.write(relative, reads)
        payload = {**payload, "retrieved_evidence": [*payload.get("retrieved_evidence", []), *reads]}
        context.runtime.check_cancelled()


def reason_once(context: RoleContext, store: TreeStore, role: str, contract: type[Decision], payload: dict) -> Decision:
    """Run one bounded LLM decision and retain its exact input and output.

    Direct calls receive a scoped snapshot, never an accumulating transcript.
    Serving-context token percentages govern selection and response headroom.
    Cancellation and stale instructions prevent acceptance after inference.
    """
    # A different process can accept cancellation between native decisions.
    # Poll its existing control boundary before starting another request.
    cancelled = getattr(context.runtime.tools.get("managed_instructions"), "cancelled", None)
    if cancelled is not None:
        cancelled()
    context.runtime.check_cancelled()
    schema = decision_schema(contract)
    original_payload = payload
    system = ((RESOURCES / f"{role}.md").read_text(encoding="utf-8")
              + "\nTo inspect an original source, set evidence_requests with its catalog source ID and numbered line window or query. "
              "Read evidence_index to find original source IDs when the selected contract does not supply their names. "
              "That response requests retrieval only; no plan or acceptance from it is committed. "
              "After retrieval, return your final decision with evidence_requests empty."
              + "\nJSON schema:\n" + json.dumps(schema, separators=(",", ":")))
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}]
    name = f"decisions/{len(store.state['decisions']) + 1:04d}-{role}"
    generation = {"chat_template_kwargs": {"enable_thinking": False}, "temperature": 0.1,
                  "response_format": {"type": "json_schema", "json_schema": {
                      "name": contract.__name__, "strict": True, "schema": schema}}}
    # The caller owns provider controls. Keep model routing and the result
    # schema fixed while recording every supplied generation setting.
    configured_generation = context.runtime.tools.get("minecraft_decision_generation")
    if configured_generation is not None:
        settings = configured_generation()
        if not isinstance(settings, dict) or set(settings) & {"model", "messages", "stream", "response_format"}:
            raise ValueError("Decision generation settings cannot change routing, messages, or the result schema.")
        generation.update(settings)
    measure = context.runtime.tools.get("minecraft_context_measure")
    if measure is None:
        raise ValueError("Project decisions need serving-context measurement from the configured model.")
    original_measurement = measurement = measure(messages, cancel=context.runtime.stop_event, **generation)
    if measurement["input_tokens"] > measurement["input_budget_tokens"]:
        payload = scoped_decision_context(payload)
        store.write(name + "/unscoped-context.json", original_payload)
        payload["context_archive"] = {"original_receipt": name + "/unscoped-context.json", "original_digest": digest(original_payload)}
        messages[1]["content"] = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        measurement = measure(messages, cancel=context.runtime.stop_event, **generation)
    input_characters = sum(len(row["content"]) for row in messages)
    store.write(name + "/input.json", {"identity": payload["identity"], "messages": messages,
                                        "generation": generation, "input_characters": input_characters,
                                        "context_budget": measurement, "original_context_budget": original_measurement})
    store.state["decisions"].append({"role": role, "path": name, "identity": payload["identity"]})
    store.save()
    if measurement["input_tokens"] > measurement["input_budget_tokens"]:
        raise ValueError(f"The selected decision has {measurement['input_tokens']} tokens; its input budget is "
                         f"{measurement['input_budget_tokens']} tokens ({measurement['input_fraction']:.0%} of "
                         f"the {measurement['context_window']}-token serving context). Original input: {name}/input.json")
    complete = context.runtime.tools["minecraft_reason"]
    if complete is None:
        raise ValueError("Minecraft project decisions need the configured normal LLM.")
    # Use the same normal-chat mode as selection and native actor sessions.
    # Retain the setting with the decision; do not cap the output mid-object.
    text = complete(messages, cancel=context.runtime.stop_event, **generation)
    store.write(name + "/raw.json", {"text": text})
    context.runtime.check_cancelled()
    if text.strip().startswith("```"):
        text = "\n".join(text.strip().splitlines()[1:-1])
    result = contract.model_validate_json(text)
    store.write(name + "/output.json", result.model_dump(mode="json"))
    return result


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
    from runtime.minecraft_actor import observed_target
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


def gameplay_history(store: TreeStore) -> dict:
    """Rebuild cumulative outcomes from every retained actor journal.

    Failed attempts remain useful evidence. A specification revision cannot
    erase inventory deltas, searched cells, paused methods, or active recovery.
    """
    evidence = GameplayEvidence()
    for attempt in store.state["attempts"]:
        relative = attempt.get("journal", attempt["path"] + "/scratch/actor-journal.jsonl")
        for row in read_actor_journal(store.path_for(relative)):
            evidence.consume(row, relative)
    if store.state.get("planning_observation"):
        path = store.path_for(store.state["planning_observation"])
        observation = json.loads(path.read_text(encoding="utf-8"))
        # Planning can discover danger. It cannot supply an actor's geometry
        # checks or grant an escape completion certificate.
        evidence.consume({"event": "result", "seq": 0, "tool": "minecraft_observe", "arguments": {},
                          "time": path.stat().st_mtime, "ok": True, "result": observation}, str(path))
    return scoped_gameplay(evidence.snapshot())


def scoped_gameplay(history: dict) -> dict:
    """Share repeated method preconditions without discarding cumulative facts.

    The actor projection and original journals retain their full structure.
    Planning keeps every method, search, quantity, and original reference.
    Named columns describe method and outcome rows. Shared source paths replace
    repeated journal strings. Changed result operands remain explicit, so this
    representation preserves quantities, identities, and paused counters.
    A recent result can reference its method's last result plus `changes`.
    Apply those exact field replacements to recover the original sequence,
    time, quantities, and source. Missing fields keep the full older record.
    """
    states, indexes, methods = [], {}, []
    for method in history["methods"]:
        state = {**method["state"],
                 "inventory": [{key: value for key, value in item.items() if key != "display_name" and value is not None}
                               for item in method["state"]["inventory"]]}
        if state["held_item"] is not None:
            state["held_item"] = {key: value for key, value in state["held_item"].items()
                                  if key != "display_name" and value is not None}
        key = digest(state)
        if key not in indexes:
            indexes[key] = len(states)
            states.append(state)
        methods.append({"tool": method["tool"], "arguments": method["arguments"],
                        "ineffective": method["ineffective"], "state_index": indexes[key],
                        "last": {key: value for key, value in method["last"].items()
                                 if key not in {"tool", "arguments"} or value != method[key]}})
    sources, source_indexes = [], {}

    def share_sources(value):
        if isinstance(value, list):
            return [share_sources(row) for row in value]
        if not isinstance(value, dict):
            return value
        result = {key: share_sources(row) for key, row in value.items()}
        # Only journal provenance has both source and seq. A Minecraft tool's
        # source operand remains an ordinary callable argument.
        if isinstance(value.get("source"), str) and "seq" in value:
            source = result.pop("source")
            if source not in source_indexes:
                source_indexes[source] = len(sources)
                sources.append(source)
            result["source_index"] = source_indexes[source]
        return result

    kinds, kind_indexes, tool_names, tool_indexes = [], {}, [], {}
    outcome_fields = ["kind_index", "unsuccessful", "reason", "quantities", "inventory_delta", "made_progress",
                      "completion_verified", "source_index", "seq", "time"]
    rows = []
    for method in methods:
        tool = method["tool"]
        if tool not in tool_indexes:
            tool_indexes[tool] = len(tool_names)
            tool_names.append(tool)
        last = share_sources(method["last"])
        kind = last.pop("kind")
        if kind not in kind_indexes:
            kind_indexes[kind] = len(kinds)
            kinds.append(kind)
        last["kind_index"] = kind_indexes[kind]
        outcome = [last[key] for key in outcome_fields]
        outcome.append({key: value for key, value in last.items() if key not in outcome_fields})
        rows.append([tool_indexes[tool], method["arguments"], method["ineffective"], method["state_index"], outcome])
    # Repeated no-ops differ mainly in sequence and time. Reference the
    # method's last result and state every changed field explicitly. This
    # preserves each call's provenance without replaying identical failures.
    method_indexes = {signature(method["tool"], method["arguments"]): index
                      for index, method in enumerate(history["methods"])}
    recent = []
    for outcome in history["recent_outcomes"]:
        index = method_indexes.get(signature(outcome["tool"], outcome["arguments"]))
        if index is None:
            recent.append(outcome)
            continue
        last = history["methods"][index]["last"]
        changes = {key: value for key, value in outcome.items() if key not in last or value != last[key]}
        # A missing old field cannot be represented by an added null value.
        # Keep that uncommon older shape explicit instead of changing it.
        reference = {"method_index": index, **({"changes": changes} if changes else {})}
        recent.append(reference if last.keys() <= outcome.keys()
                      and len(json.dumps(reference)) < len(json.dumps(outcome)) else outcome)
    remaining = share_sources({**{key: value for key, value in history.items() if key != "methods"},
                               "recent_outcomes": recent})
    search_defaults = ({key: remaining["searches"][0][key] for key in ("world", "method", "uncertainty")}
                       if remaining["searches"] else {})
    remaining["searches"] = [{key: value for key, value in search.items()
                             if key not in search_defaults or value != search_defaults[key]}
                            for search in remaining["searches"]]
    # Tools and stock recur across different body positions. Share complete
    # item facts once; a changed count, slot, or durability gets its own row.
    items, item_indexes = [], {}
    def item_index(item):
        key = digest(item)
        if key not in item_indexes:
            item_indexes[key] = len(items)
            items.append(item)
        return item_indexes[key]
    shared_states = [{**{key: value for key, value in state.items() if key not in {"inventory", "held_item"}},
                      "inventory_indexes": [item_index(item) for item in state["inventory"]],
                      "held_item_index": item_index(state["held_item"]) if state["held_item"] is not None else None}
                     for state in states]
    return {**remaining, "schema": "pm.gameplay-planning.v4", "methods": rows, "method_states": shared_states,
            "inventory_items": items,
            "method_fields": ["tool_index", "arguments", "ineffective", "state_index", "last"], "tool_names": tool_names,
            "outcome_fields": [*outcome_fields, "additional_fields"], "source_paths": sources, "outcome_kinds": kinds,
            "search_defaults": search_defaults}


def scoped_rejected_tree(previous: TaskTree | None, rejected: TaskTree, source: str) -> dict:
    """Describe a rejected revision through changes to the supplied old tree.

    Unchanged conditions reference their exact previous node and condition ID.
    Changed conditions retain their full operands. Original rejected proposals
    remain on disk, including nodes omitted from this repair-only projection.
    Without an accepted tree, every candidate node stays explicit.
    """
    from workflows.minecraft_tasks.tree import node_digest
    prior_nodes = previous.nodes if previous is not None else []
    old = {node.id: node for node in prior_nodes}
    conditions = {digest(row.model_dump(mode="json")): {"from_node": node.id, "condition_id": row.id}
                  for node in prior_nodes for row in node.criteria + node.assumptions}
    changed = []
    for node in rejected.nodes:
        if node.id in old and node_digest(node) == node_digest(old[node.id]):
            continue
        value = node.model_dump(mode="json", exclude_defaults=True)
        for field in ("criteria", "assumptions"):
            if field in value:
                value[field] = [conditions.get(digest(row.model_dump(mode="json")),
                                              row.model_dump(mode="json", exclude_defaults=True))
                                for row in getattr(node, field)]
        changed.append(value)
    return {"original_receipt": source, "original_digest": digest(rejected.model_dump(mode="json")),
            "goal_id": rejected.goal_id, "removed_nodes": sorted(old.keys() - {node.id for node in rejected.nodes}),
            "changed_nodes": changed}


def scoped_rejection(previous: TaskTree | None, document: dict, source: str) -> dict:
    """Keep invalid raw model output distinct from a typed rejected tree.

    Syntax feedback includes a bounded excerpt around the parse error.
    Initial contract repairs retain all parsed operands because no accepted
    tree can supply omitted geometry. Neither form accepts the candidate.
    """
    if "text" not in document:
        rejected = TaskTree.model_validate(document)
        if previous is None:
            return {"original_receipt": source, "original_digest": digest(rejected.model_dump(mode="json")),
                    "kind": "invalid_initial_tree", "candidate_tree": rejected.model_dump(mode="json", exclude_defaults=True),
                    "repair": "No specification exists yet. Return a complete initial tree with every referenced node. Repair the supplied tree error; this candidate is not accepted work."}
        scoped = scoped_rejected_tree(previous, rejected, source)
        if scoped["removed_nodes"]:
            # A candidate missing stable identities fails before any operand
            # review. Start the repair from the authoritative retained tree.
            return {key: value for key, value in scoped.items() if key != "changed_nodes"} | {
                "kind": "removed_stable_nodes", "repair": "Use explicit updates against previous_tree. Preserve every stable node and parent-child reference; this rejected candidate never reached world execution."}
        return scoped
    text = document["text"]
    if text.strip().startswith("```"):
        text = "\n".join(text.strip().splitlines()[1:-1])
    offset, error = 0, "The output failed the proposal contract; use the supplied validation feedback."
    try:
        candidate = json.loads(text)
    except json.JSONDecodeError as invalid:
        offset, error = max(0, invalid.pos - 800), str(invalid)
    else:
        if previous is None:
            try:
                PlanProposal.model_validate(candidate)
            except ValidationError as invalid:
                errors = invalid.errors(include_url=False, include_input=False, include_context=False)
            else:
                errors = []
            return {"original_receipt": source, "original_digest": digest(document),
                    "kind": "invalid_model_output", "candidate": candidate, "validation_errors": errors,
                    "repair": "This is rejected output, not an accepted tree. Preserve its concrete operands while repairing the supplied contract errors."}
    return {"original_receipt": source, "original_digest": digest(document), "kind": "invalid_model_output",
            "original_characters": len(text), "error": error, "excerpt_start": offset,
            "excerpt": text[offset:offset + 2000]}


def rejected_context(store: TreeStore, source: str) -> dict:
    """Read a rejected proposal without inventing an initial tree revision.

    Initial planning can reject output before any specification exists.
    Retained revisions still supply their authoritative comparison tree.
    Initial repairs also retain the first contract rejection. Later invalid
    proposals cannot erase its concrete geometry. Original files stay exact.
    """
    document = json.loads(store.path_for(source).read_text(encoding="utf-8"))
    previous = store.load() if store.state["revision"] else None
    result = scoped_rejection(previous, document, source)
    if previous is None:
        for decision in store.state["decisions"]:
            if decision["role"] != "planner" or decision["identity"]["tree_revision"] != 0:
                continue
            first_source = decision["path"] + "/raw.json"
            path = store.path_for(first_source)
            if not path.is_file() or store.path_for(decision["path"] + "/output.json").exists():
                continue
            first = scoped_rejection(None, json.loads(path.read_text(encoding="utf-8")), first_source)
            if not first.get("validation_errors"):
                continue
            if first_source != source:
                result["first_rejected_proposal"] = first
            break
    return result


def initial_candidate(store: TreeStore) -> tuple[str, TaskTree] | None:
    """Read the latest fully declared initial candidate from rejection history.

    Every referenced identity must already exist in this immutable proposal.
    Later root-only fragments cannot replace its physical operands. The
    candidate remains unaccepted; explicit model updates still need the full
    tree validator, fresh world survey, and independent plan review.
    No candidate is available after a specification has been committed.
    """
    if store.state["revision"]:
        return None
    for number in range(store.state.get("rejections", 0), 0, -1):
        source = f"planning/rejected/{number:04d}.json"
        tree = TaskTree.model_validate(json.loads(store.path_for(source).read_text(encoding="utf-8")))
        ids = {node.id for node in tree.nodes}
        references = [key for node in tree.nodes for key in node.children + node.depends_on]
        references.extend(node.parent for node in tree.nodes if node.parent is not None)
        references.extend(condition.node_id for node in tree.nodes for condition in node.criteria + node.assumptions
                          if condition.kind == "accepted")
        if len(ids) == len(tree.nodes) and tree.goal_id in ids and all(key in ids for key in references):
            return source, tree
    return None


def apply_node_updates(previous: TaskTree, updates: list[Node]) -> TaskTree:
    """Merge explicit node replacements into a retained complete tree.

    Omitted nodes remain exact copies. The normal revision validator still
    checks stable IDs, parents, scope, conditions, and prerequisite references.
    """
    replacements = {node.id: node for node in updates}
    if not replacements or len(replacements) != len(updates):
        raise ValueError("Node updates need a nonempty list of unique stable IDs.")
    old_ids = {node.id for node in previous.nodes}
    nodes = [replacements.get(node.id, node).model_copy(deep=True) for node in previous.nodes]
    nodes.extend(node.model_copy(deep=True) for node in updates if node.id not in old_ids)
    return previous.model_copy(deep=True, update={"nodes": nodes})


def revision_candidate(previous: TaskTree, proposal: PlanProposal) -> TaskTree:
    """Build a candidate without confusing proof restoration with node edits.

    Restoration-only proposals retain the exact tree for the later proof
    boundary. They do not establish a concrete leaf repair or acceptance.
    Invalid node replacements raise ValueError before the task state changes.
    """
    if proposal.tree is not None:
        return proposal.tree
    if proposal.updates:
        return apply_node_updates(previous, proposal.updates)
    if proposal.restore_nodes:
        return previous.model_copy(deep=True)
    raise ValueError("A revision needs node updates or explicit proof restoration.")


def refinement_scope(store: TreeStore, instruction_revision: int, history: dict) -> str:
    """Open optional scope only when recovery still needs a ready leaf."""
    if instruction_revision != store.state["instruction_revision"]:
        return ""
    if history["recovery"]:
        nodes = {node.id: node for node in store.load().nodes}
        if not any(nodes[key].specialist == "recovery" for key in store.eligible()):
            return ""
    return store.state.get("refinement_target", "")


def refinement_plan(previous: TaskTree, target: str, proposal: RefinementProposal) -> PlanProposal:
    """Bind a model's replacement to the actual named refinement target."""
    updates = []
    if proposal.status == "planned":
        if proposal.replacement.id != target:
            raise ValueError(f"replacement must retain target ID {target!r}, not {proposal.replacement.id!r}.")
        old = {node.id: node for node in previous.nodes}
        copied = sorted(node.id for node in proposal.additions if node.id in old)
        if copied:
            raise ValueError("additions cannot replace existing unrelated nodes: " + ", ".join(copied)
                             + ". Keep them unchanged; add new children under replacement.")
        updates = [proposal.replacement, *proposal.additions]
    return PlanProposal(status=proposal.status, summary=proposal.summary, updates=updates, queries=proposal.queries,
                        recipe_queries=proposal.recipe_queries)


def survey_conditions(nodes: list[Node]) -> list[Condition]:
    """Inspect proposed physical operands once, without requiring completion.

    Survey facts describe the current site. Unmet future conditions are normal.
    Deduplication reduces repeated reads across parent and child criteria.
    """
    queries = {}
    for node in nodes:
        for condition in node.criteria + node.assumptions:
            if condition.kind == "block":
                for point in condition.positions or [condition.position]:
                    key = digest({"kind": "block", "position": point.model_dump(), "dimension": condition.dimension})
                    queries.setdefault(key, condition.model_copy(update={"position": point, "positions": []}))
            elif condition.kind == "region":
                key = digest({"kind": "region", "start": condition.position.model_dump(), "end": condition.end.model_dump(), "dimension": condition.dimension})
                queries.setdefault(key, condition.model_copy())
    return [Condition(id="survey-body", description="Current body state.", kind="alive")] + [
        condition.model_copy(update={"id": f"survey-{index:04d}"}) for index, condition in enumerate(queries.values(), 1)]


def restart_review(store: TreeStore, phase: str, instruction_revision: int, answer: str) -> bool:
    """Require fresh review of a retained tree before resuming unchanged work.

    Mark interrupted actor attempts without changing their original evidence.
    New instructions, human answers, and a required concrete leaf change still
    enter planning. A saved waiting interpretation cannot authorize mutation.
    """
    if phase != "plan" or not store.state["revision"]:
        return False
    for attempt in store.state["attempts"]:
        if attempt["state"] == "running":
            attempt.update(state="interrupted", journal=attempt["path"] + "/scratch/actor-journal.jsonl")
    store.state["recovery_observation"] = store.state["planning_observation"]
    store.save()
    return (instruction_revision == store.state["instruction_revision"] and not answer
            and not store.state.get("require_leaf_change"))


def compact_plan_context(payload: dict) -> dict:
    """Reference identical review text and tool contracts once in one decision.

    Keep differing feedback and contracts verbatim. Original survey paths and
    digests still identify the full historical documents.
    """
    result = dict(payload)
    review = result.get("latest_plan_review", {})
    if result.get("feedback") and result["feedback"] == review.get("summary"):
        result["feedback"] = "See latest_plan_review.summary for the original review feedback."
    survey = result.get("previous_site_survey")
    current = result.get("current_tool_contracts", [])
    if survey and survey.get("tool_contracts"):
        shared = [row for row in survey["tool_contracts"] if row in current]
        if shared:
            result["previous_site_survey"] = {**survey,
                "tool_contracts": [row for row in survey["tool_contracts"] if row not in current],
                "shared_tool_contracts": "current_tool_contracts"}
    return result


def scoped_decision_context(payload: dict) -> dict:
    """Select current decision evidence while retaining the full disk archive.

    Keep every paused method, twenty recent outcomes, and the latest thirty-two
    methods by result time. Earlier methods supply quantities, failure counts,
    and original journal references. Their old preconditions and explanations
    cannot establish current world geometry. Actors receive the same selection,
    with complete history available in their inbox archive.
    Named columns retain complete method states and typed search coordinates.
    Repeated coverage values share a table; every original search remains.
    Accepted historical reads remain in their original certificate; failed
    verification and all search coverage stay explicit in this decision.
    Requested survey outputs from an earlier tree revision remain in their
    original receipt. Keep their query operands and provenance here, because
    those old outputs cannot establish the current landing or inventory.
    This selection cannot change a specification or authorize acceptance.
    """
    result = dict(payload)
    # Recipe availability belongs to its observed inventory. An older attempt
    # boundary cannot make those quantities current after more world effects.
    prior_survey = result.get("previous_site_survey", {})
    attempt_started = result.get("latest_attempt", {}).get("started")
    if (isinstance(attempt_started, (int, float)) and prior_survey.get("original_receipt")
            and prior_survey.get("original_digest")):
        result["previous_site_survey"] = {**prior_survey,
            "recipe_scope": "Outputs before latest_attempt.started stay in original_receipt with each row's original_digest. Request fresh inventory availability after world effects.", "recipes": [
            {**{key: value for key, value in row.items() if key != "result"},
             "result_scope": "Historical output; see recipe_scope."}
            if isinstance(row.get("time"), (int, float)) and row["time"] < attempt_started and row.get("original_digest")
            else row for row in prior_survey.get("recipes", [])]}
    survey = result.get("requested_world_survey", {})
    revision = result.get("identity", {}).get("tree_revision")
    surveyed_revision = survey.get("identity", {}).get("tree_revision")
    if (isinstance(revision, int) and isinstance(surveyed_revision, int)
            and surveyed_revision < revision and survey.get("original_receipt")
            and survey.get("original_digest")):
        result["requested_world_survey"] = {key: value for key, value in survey.items()
            if key not in {"reads", "recipes"}}
        result["requested_world_survey"].update(
            archived_read_count=len(survey.get("reads", [])),
            archived_recipe_count=len(survey.get("recipes", [])),
            read_scope="Historical survey from an earlier tree revision. Query operands remain explicit; exact outputs stay in original_receipt with original_digest. Use fresh reads for current geometry, supplies, and harvesting prerequisites.")
    proof = result.get("latest_verification", {})
    if proof.get("verified") is True:
        result["latest_verification"] = {key: value for key, value in proof.items() if key != "reads"}
        result["latest_verification"]["read_scope"] = "Past accepted attainment. Original reads remain in original_receipt; request fresh world reads for current geometry."
        specification = proof.get("specification")
        if specification and result.get("previous_tree"):
            old = Node.model_validate(specification)
            prior = TaskTree.model_validate(result["previous_tree"])
            if any(node.model_dump() == old.model_dump() for node in prior.nodes):
                result["latest_verification"]["specification"] = {"from_node": old.id, "in": "previous_tree"}
    history = result.get("gameplay_history")
    if not history or history.get("schema") != "pm.gameplay-planning.v4":
        return result
    methods = history["methods"]
    fields = {key: index for index, key in enumerate(history["outcome_fields"])}
    newest = sorted(range(len(methods)), key=lambda index: methods[index][4][fields["time"]])[-32:]
    selected = set(newest) | {index for index, row in enumerate(methods) if row[2] >= 2}
    selected.update(row["method_index"] for row in history["recent_outcomes"] if "method_index" in row)
    indexes = sorted(selected)
    remap = {old: new for new, old in enumerate(indexes)}
    archived = {}
    for index, row in enumerate(methods):
        if index in selected:
            continue
        last = dict(zip(history["outcome_fields"], row[4]))
        tool = history["tool_names"][row[0]]
        group = archived.setdefault(tool, {"method_count": 0, "source_sequences": {}, "reasons": {},
                                          "quantities": {}, "inventory_changes": {}})
        group["method_count"] += 1
        group["source_sequences"].setdefault(str(last["source_index"]), []).append(last["seq"])
        if last["reason"]:
            reason = last["reason"]
            group["reasons"][reason] = group["reasons"].get(reason, 0) + 1
        for key, count in last["quantities"].items():
            if isinstance(count, (int, float)) and not isinstance(count, bool):
                group["quantities"][key] = group["quantities"].get(key, 0) + count
        for change in last["inventory_delta"].get("changes", []):
            item = change["item"]
            counts = group["inventory_changes"].setdefault(item, {"gained": 0, "removed": 0})
            counts["gained" if change["delta"] >= 0 else "removed"] += abs(change["delta"])
    sources = list(history["source_paths"])
    def diagnosis_source(source: str) -> int:
        if source not in sources:
            sources.append(source)
        return sources.index(source)
    scoped = {**history, "source_paths": sources, "methods": [methods[index] for index in indexes],
              "recent_outcomes": [{**row, "method_index": remap[row["method_index"]]} if "method_index" in row else row
                                  for row in history["recent_outcomes"]],
              "diagnoses": history["diagnoses"][-4:],
              "archived_diagnosis_fields": ["source_index", "time", "evidence_sequences"],
              "archived_diagnoses": [[diagnosis_source(row["source"]), row["time"], row["evidence_sequences"]]
                                     for row in history["diagnoses"][:-4]],
              "archived_methods": archived,
              "decision_scope": "Current methods retain exact operands and preconditions. Archived totals sum each older method's LAST result, not every call or mined-resource provenance. Original journals and unscoped context retain all operands and diagnoses. Current inventory and independent certificates take precedence."}
    states = sorted({row[3] for row in scoped["methods"]})
    state_map = {old: new for new, old in enumerate(states)}
    scoped["methods"] = [[*row[:3], state_map[row[3]], row[4]] for row in scoped["methods"]]
    scoped["method_states"] = [history["method_states"][index] for index in states]
    used_items = sorted({index for state in scoped["method_states"] for index in
                         state["inventory_indexes"] + ([state["held_item_index"]] if state["held_item_index"] is not None else [])})
    item_map = {old: new for new, old in enumerate(used_items)}
    scoped["inventory_items"] = [history["inventory_items"][index] for index in used_items]
    scoped["method_states"] = [{**state, "inventory_indexes": [item_map[index] for index in state["inventory_indexes"]],
                                "held_item_index": item_map[state["held_item_index"]] if state["held_item_index"] is not None else None}
                               for state in scoped["method_states"]]
    states = scoped["method_states"]
    if states and all(state.keys() == states[0].keys() for state in states):
        state_fields = list(states[0])
        scoped["method_state_fields"] = state_fields
        scoped["method_states"] = [[state[key] for key in state_fields] for state in states]
    # Named columns preserve every search's exact coverage and availability.
    # Keep unusual older record shapes unchanged rather than inventing fields.
    searches = scoped["searches"]
    if searches and all(isinstance(row, dict) and row.keys() == searches[0].keys() for row in searches):
        search_fields = list(searches[0])
        if "coverage" in search_fields:
            # Several searches can observe identical local coverage. Share its
            # exact typed value, while keeping every search's time and source.
            coverages, coverage_indexes = [], {}
            records = []
            for row in searches:
                coverage = row["coverage"]
                key = json.dumps(coverage, ensure_ascii=False, separators=(",", ":"))
                if key not in coverage_indexes:
                    coverage_indexes[key] = len(coverages)
                    coverages.append(coverage)
                records.append({**{key: value for key, value in row.items() if key != "coverage"},
                                "coverage_index": coverage_indexes[key]})
            search_fields[search_fields.index("coverage")] = "coverage_index"
            axes = ["x", "y", "z"]
            dictionaries = [coverage for coverage in coverages if isinstance(coverage, dict)]
            if dictionaries and all(isinstance(coverage.get("bounds"), dict)
                    and all(isinstance(point, dict) and set(point) == set(axes)
                            for point in (coverage.get("center"), coverage["bounds"].get("min"),
                                          coverage["bounds"].get("max"))) for coverage in dictionaries):
                scoped["search_coverage_vector_fields"] = axes
                coverages = [{**coverage, "center": [coverage["center"][axis] for axis in axes],
                    "bounds": {**coverage["bounds"],
                               **{key: [coverage["bounds"][key][axis] for axis in axes] for key in ("min", "max")}}}
                    if isinstance(coverage, dict) else coverage for coverage in coverages]
            coverage_fields, coverage_rows = [], []
            for coverage in coverages:
                if not isinstance(coverage, dict):
                    coverage_rows.append([None, coverage])
                    continue
                keys = list(coverage)
                if keys not in coverage_fields:
                    coverage_fields.append(keys)
                coverage_rows.append([coverage_fields.index(keys), [coverage[key] for key in keys]])
            scoped["search_coverage_field_sets"] = coverage_fields
            scoped["search_coverages"] = coverage_rows
            searches = records
        defaults = dict(scoped["search_defaults"])
        for key in list(search_fields):
            value = searches[0][key]
            encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            if all(json.dumps(row[key], ensure_ascii=False, separators=(",", ":")) == encoded for row in searches):
                defaults[key] = value
                search_fields.remove(key)
        scoped["search_defaults"] = defaults
        scoped["search_fields"] = search_fields
        scoped["searches"] = [[row[key] for key in search_fields] for row in searches]
    result["gameplay_history"] = scoped
    return result


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


def plan(context: RoleContext, store: TreeStore) -> dict:
    """Create or revise a retained tree from fresh observations and feedback."""
    req = request(context)
    current = context.runtime.tools["managed_instructions"]()
    goal = current["instruction"] or req.goal or req.instruction
    with BODY_LEASES.command(context.runtime.tools["minecraft_endpoint"](), context.runtime.stop_event):
        observation = context.runtime.tools["minecraft_call"]("minecraft_observe", {"include_image": False})
        server_info = context.runtime.tools["minecraft_call"]("minecraft_info", {})
    if not observation["ok"]:
        return {"status": "waiting", "summary": "A fresh Minecraft observation is unavailable. Restore the body connection."}
    if not server_info["ok"]:
        return {"status": "waiting", "summary": "Current server rules are unavailable. Restore the server connection before choosing recovery."}
    observation_path = f"planning/observations/{len(store.state['decisions']) + 1:04d}.json"
    store.write(observation_path, observation["result"])
    rules_path = f"planning/server-rules/{len(store.state['decisions']) + 1:04d}.json"
    store.write(rules_path, server_info["result"])
    store.state["server_rules"] = rules_path
    store.state["planning_observation"] = observation_path
    first = "planning/observations/0001.json"
    store.state.setdefault("initial_observation", first if store.path_for(first).is_file() else observation_path)
    store.save()
    retention = retain(context, store, observation["result"])
    if retention["status"] != "unchanged":
        return retention
    # A resumed caller can revise its audit operands. Read the current
    # contract before planning instead of reusing an older review receipt.
    if store.state["revision"]:
        record_plan_constraints(context, store)
    # Re-review the retained tree against fresh facts and current contracts.
    # A historical waiting review may have misread evidence already provided.
    if restart_review(store, context.phase, current["revision"], context.answer or req.answer):
        return {"status": "planned", "summary": "Retained the task tree after fresh restart observation. Review it before another actor."}
    # The supplied planner schema defines omitted defaults. This representation
    # round-trips exactly, while rejected raw output remains intact for repair.
    previous = store.load().model_dump(mode="json", exclude_defaults=True) if store.state["revision"] else None
    repair_base = initial_candidate(store)
    payload = {"identity": identity(context, store), "goal": goal, "observation": observation["result"],
               "server_rules": server_info["result"].get("server_rules", ""),
               "planning_mode": "initial_candidate_repair" if repair_base else "initial_full_tree" if previous is None else "retained_revision",
               "project_initial_observation": json.loads(store.path_for(store.state["initial_observation"]).read_text(encoding="utf-8")),
               "previous_tree": previous, "accepted_nodes": list(store.state["acceptances"]),
               "feedback": store.state["feedback"], "refinement_target": store.state.get("refinement_target", ""),
               "persona_context": req.persona_context, "answer": context.answer or req.answer,
               "latest_attempt": store.state["attempts"][-1] if store.state["attempts"] else None}
    payload["latest_project_check"] = project_check_context(store)
    if repair_base:
        source, candidate = repair_base
        payload["initial_repair_base"] = {"original_receipt": source, "original_digest": digest(candidate.model_dump(mode="json")),
                                        "accepted": False, "tree": candidate.model_dump(mode="json", exclude_defaults=True)}
        # A later fragment or malformed patch can have different feedback.
        # Diagnose this exact immutable base before requesting node repairs.
        try:
            validate_tree(candidate)
        except ValueError as error:
            payload["initial_repair_base"]["tree_error"] = str(error)
    payload["current_tool_contracts"] = [row for row in context.runtime.tools["minecraft_tool_schemas"]()
                                         if row["name"] in {"minecraft_recipe_search", "minecraft_smelt_item"}]
    payload["gameplay_history"] = gameplay_history(store)
    candidates = store.acceptance_candidates()
    if candidates:
        from workflows.minecraft_tasks.tree import node_digest
        current_nodes = {node.id: node for node in store.load().nodes}
        payload["retained_acceptance_candidates"] = [
            {"specification": ({"from_node": row["specification"]["id"], "in": "previous_tree"}
                               if row["specification"]["id"] in current_nodes
                               and node_digest(current_nodes[row["specification"]["id"]]) == row["certificate"]["spec_digest"]
                               else Node.model_validate(row["specification"]).model_dump(mode="json", exclude_defaults=True)),
             "archive": row["archive"],
             "certificate_digest": digest(row["certificate"]),
             "required_certificates": row["certificate"]["dependencies"],
             "changed_dependencies": [key for key, expected in row["certificate"]["dependencies"].items()
                                      if key not in store.state["acceptances"] or digest(store.state["acceptances"][key]) != expected],
             "verification": row["certificate"]["verification"]} for row in candidates]
    payload["refinement_target"] = refinement_scope(store, current["revision"], payload["gameplay_history"])
    if payload["gameplay_history"]["recovery"]:
        payload["refinement_scope_reason"] = (
            "A ready recovery leaf already exists. Honor refinement_target and retain that recovery leaf unchanged."
            if payload["refinement_target"] else
            "Active recovery can supersede optional scope when it needs a ready leaf. Follow the latest review and retain useful completed work.")
    latest_review = next((row for row in reversed(store.state["decisions"]) if row["role"] == "reviewer"), None)
    if latest_review:
        review_path = latest_review["path"] + "/output.json"
        if store.path_for(review_path).is_file():
            payload["latest_plan_review"] = {**json.loads(store.path_for(review_path).read_text(encoding="utf-8")),
                                             "original_receipt": review_path}
    latest_verifier = next((row for row in reversed(store.state["decisions"]) if row["role"] == "verifier"), None)
    if latest_verifier:
        decision = json.loads(store.path_for(latest_verifier["path"] + "/input.json").read_text(encoding="utf-8"))
        selected = json.loads(decision["messages"][1]["content"])
        relative = selected["original_receipt"]
        proof = json.loads(store.path_for(relative).read_text(encoding="utf-8"))
        payload["latest_verification"] = {"identity": proof["identity"], "checks": proof["checks"],
            "verified": proof["verified"],
            "review": proof["review"], "specification": selected["specification"], "reads": scoped_reads(proof["reads"]),
            "original_receipt": relative, "original_digest": digest(proof)}
    if store.state.get("planning_survey"):
        survey = json.loads(store.path_for(store.state["planning_survey"]).read_text(encoding="utf-8"))
        payload["previous_site_survey"] = {"identity": survey["identity"], "original_receipt": store.state["planning_survey"],
                                           "original_digest": digest(survey)}
        if not payload["refinement_target"]:
            payload["previous_site_survey"]["recipes"] = scoped_reads(survey.get("recipes", []))
            payload["previous_site_survey"]["tool_contracts"] = survey.get("tool_contracts", [])
        # Prefer the latest verifier's exact scope during repair. The earlier
        # site survey stays retained; every revised plan receives a new survey.
        if not latest_verifier:
            payload["previous_site_survey"]["reads"] = scoped_reads(survey["reads"])
    rejected = store.state.get("rejected_proposal")
    if rejected:
        if repair_base:
            document = json.loads(store.path_for(rejected).read_text(encoding="utf-8"))
            payload["rejected_proposal"] = scoped_rejection(repair_base[1], document, rejected)
            payload["rejected_proposal"]["comparison_tree"] = "initial_repair_base.tree (unaccepted candidate)"
        else:
            payload["rejected_proposal"] = rejected_context(store, rejected)
    if store.state.get("requested_survey"):
        relative = store.state["requested_survey"]
        observed = json.loads(store.path_for(relative).read_text(encoding="utf-8"))
        payload["requested_world_survey"] = {**observed, "reads": scoped_reads(observed["reads"]),
                                             "recipes": scoped_reads(observed.get("recipes", [])),
                                             "original_receipt": relative, "original_digest": digest(observed)}
    if store.state.get("planning_constraints"):
        payload["plan_constraints"] = caller_plan_constraints(store)
    selected_scope = None
    if previous:
        tree = store.load()
        target = payload["refinement_target"] or store.state.get("review_target") or store.state.get("selection", {}).get("leaf_id")
        if target:
            selected_scope = dependency_scope(tree, [target])
            full = payload["gameplay_history"]
            relative = f"planning/context/{len(store.state['decisions']) + 1:04d}-history.json"
            store.write(relative, full)
            payload["gameplay_history"] = scoped_history(store, selected_scope, full, {target})
            payload["full_history"] = {"original_receipt": relative, "original_digest": digest(full)}
            # Complete criteria remain on selected nodes. Ancestor integration
            # contracts stay explicit; unrelated siblings appear as topology.
            ancestors, cursor = set(), target
            nodes = {node.id: node for node in tree.nodes}
            while cursor is not None:
                ancestors.add(cursor)
                cursor = nodes[cursor].parent
            payload["previous_tree"] = {**previous, "nodes": [row for row in previous["nodes"]
                if row["id"] in selected_scope | ancestors]}
            payload["tree_topology"] = [{"id": node.id, "parent": node.parent, "children": node.children,
                                          "depends_on": node.depends_on, "kind": node.kind}
                                         for node in tree.nodes]
            payload["decision_scope"] = {"target": target, "node_ids": sorted(selected_scope),
                                         "meaning": "Evidence scope only. Omitted task nodes remain unchanged in the durable tree."}
    restore_nodes = []
    if req.proposed_tree is not None and previous is None and not store.state.get("source_proposal_used"):
        store.state["source_proposal_used"] = True
        store.save()
        proposed = req.proposed_tree
    else:
        try:
            contract = RefinementProposal if payload["refinement_target"] else RevisionProposal if previous or repair_base else PlanProposal
            if payload["refinement_target"]:
                payload["replacement_binding"] = {"id": payload["refinement_target"],
                    "parent": next(node.parent for node in store.load().nodes if node.id == payload["refinement_target"]),
                    "conditions": "Copy this node's original criteria from previous_tree exactly. Keep unrelated nodes omitted."}
            proposal = reason(context, store, "planner", contract, compact_plan_context(payload))
        except ValidationError as error:
            store.state.update(feedback="The proposal violates its JSON contract: " + str(error),
                               rejected_proposal=store.state["decisions"][-1]["path"] + "/raw.json")
            store.save()
            return {"status": "invalid", "summary": store.state["feedback"]}
        if isinstance(proposal, RefinementProposal):
            try:
                proposal = refinement_plan(store.load(), payload["refinement_target"], proposal)
            except ValueError as error:
                store.state.update(feedback="The refinement violates its target binding: " + str(error),
                                   rejected_proposal=store.state["decisions"][-1]["path"] + "/raw.json")
                store.save()
                return {"status": "invalid", "summary": store.state["feedback"]}
        elif isinstance(proposal, RevisionProposal):
            proposal = PlanProposal(status=proposal.status, summary=proposal.summary, updates=proposal.updates,
                                    restore_nodes=proposal.restore_nodes,
                                    queries=proposal.queries, recipe_queries=proposal.recipe_queries)
        if proposal.status == "survey":
            store.state["pending_survey"] = {"instruction_revision": current["revision"],
                                              "queries": [row.model_dump(mode="json") for row in proposal.queries],
                                              "recipe_queries": proposal.recipe_queries}
            store.save()
        if proposal.status != "planned":
            return {"status": proposal.status, "summary": proposal.summary}
        if (proposal.updates or proposal.restore_nodes) and previous is None and repair_base is None:
            store.state.update(feedback="Initial planning requires a full tree. Node updates need previous_tree.",
                               rejected_proposal=store.state["decisions"][-1]["path"] + "/raw.json")
            store.save()
            return {"status": "invalid", "summary": store.state["feedback"]}
        try:
            proposed = proposal.tree if proposal.tree is not None else revision_candidate(
                repair_base[1] if repair_base else store.load(), proposal)
        except ValueError as error:
            store.state.update(feedback="The node revision was rejected before dispatch: " + str(error),
                               rejected_proposal=store.state["decisions"][-1]["path"] + "/raw.json")
            store.save()
            return {"status": "invalid", "summary": store.state["feedback"]}
        restore_nodes = proposal.restore_nodes
        if selected_scope is not None:
            original_nodes = {node.id: node for node in store.load().nodes}
            from workflows.minecraft_tasks.tree import node_digest
            protected = set(original_nodes) - selected_scope - ancestors
            if any(node.id in protected and node_digest(node) != node_digest(original_nodes[node.id])
                   for node in proposal.updates):
                return {"status": "invalid", "summary": "A scoped repair changed an unrelated existing task. Select its own scope first."}
    target = payload["refinement_target"]
    try:
        # Restoring proof is an explicit native plan operation. The model
        # need not reproduce every unchanged historical field by hand.
        proposed = store.restore_specifications(proposed, restore_nodes)
        if previous:
            protect_accepted_specifications(store, proposed)
        if context.phase == "replan" and previous and store.state.get("require_leaf_change"):
            old_nodes = {node.id: node for node in store.load().nodes}
            from workflows.minecraft_tasks.tree import node_digest
            changed = [node for node in proposed.nodes if node.kind == "leaf"
                       and (node.id not in old_nodes or node_digest(node) != node_digest(old_nodes[node.id]))]
            if not changed:
                raise ValueError("Replanning must add, revise, or defer a concrete leaf. An explanation alone cannot release another attempt.")
        store.revise(proposed, target=target)
    except ValueError as error:
        rejection = store.state.get("rejections", 0) + 1
        relative = f"planning/rejected/{rejection:04d}.json"
        store.write(relative, proposed.model_dump(mode="json"))
        store.state.update(feedback="The proposed tree was rejected before dispatch: " + str(error),
                           rejected_proposal=relative, rejections=rejection)
        store.save()
        return {"status": "invalid", "summary": store.state["feedback"]}
    store.state.update(instruction_revision=current["revision"], refinement_target="", goal=goal, rejected_proposal="")
    store.state.pop("require_leaf_change", None)
    store.save()
    return {"status": "planned", "summary": f"Retained specification revision {store.state['revision']}."}


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


def caller_plan_constraints(store: TreeStore) -> dict | None:
    """Read complete caller requirements with their original receipt digest.

    These requirements describe coverage and separate publication audits.
    They cannot establish world truth or accept a task. This read changes no
    task state and returns ``None`` when no caller contract was recorded.
    """
    relative = store.state.get("planning_constraints")
    if not relative:
        return None
    proof = json.loads(store.path_for(relative).read_text(encoding="utf-8"))
    return {**proof["result"], "original_receipt": relative, "original_digest": digest(proof)}


def record_plan_constraints(context: RoleContext, store: TreeStore) -> dict | None:
    """Retain current caller operands before planning or independent review.

    The callback describes the plan contract and changes no world state.
    Keep prior receipts intact when the caller changes its requirements.
    Errors remain planning evidence; the reviewer decides whether to dispatch.
    """
    constraint = context.runtime.tools.get("minecraft_plan_constraints")
    if constraint is None:
        return None
    result = constraint(store.load())
    number = store.state.get("constraint_reviews", 0) + 1
    relative = f"planning/constraints/{number:04d}.json"
    store.write(relative, {"identity": identity(context, store), "tree_digest": digest(store.load().model_dump(mode="json")),
                           "result": result})
    store.state.update(constraint_reviews=number, planning_constraints=relative)
    store.save()
    return result


def project_check_context(store: TreeStore) -> dict | None:
    """Keep the latest publication check explicit across planning retries.

    Schema errors can replace feedback text, but cannot erase the separate
    publication failure. Detailed effects remain in the original receipt.
    This read changes no task state and cannot accept a completion candidate.
    """
    relative = store.state.get("project_check")
    if not relative:
        return None
    proof = json.loads(store.path_for(relative).read_text(encoding="utf-8"))
    return {**{key: proof["result"][key] for key in ("errors", "checks", "requirements") if key in proof["result"]},
            "identity": proof["identity"], "original_receipt": relative, "original_digest": digest(proof)}


def leaf_project_context(store: TreeStore, ancestors: list[dict]) -> dict:
    """Give private actors the caller requirements that planners received.

    These operands explain publication audits and their evidence needs.
    They do not accept work or extend the selected leaf's instruction.
    Keep their original receipt and digest beside the supplied facts.
    """
    project = {"goal": store.state["goal"], "ancestor_specifications": ancestors}
    constraints = caller_plan_constraints(store)
    if constraints is not None:
        project["caller_constraints"] = constraints
    check = project_check_context(store)
    if check is not None:
        project["latest_project_check"] = check
    return project


def review(context: RoleContext, store: TreeStore) -> dict:
    """Review acceptance coverage independently before selecting any leaf."""
    if not boundary(context, store):
        return {"status": "refine", "summary": "Project instructions changed before plan review."}
    tree = store.load()
    result = record_plan_constraints(context, store)
    if result is not None:
        if result["errors"]:
            store.state.pop("reviewed_revision", None)
            rejected = PlanReview(status="revise", summary="The configured project contract requires repair: " + " ".join(result["errors"]))
            record_review(store, rejected)
            return {"status": rejected.status, "summary": rejected.summary}
        store.save()
    ordered = review_nodes(store)
    survey_path = f"planning/surveys/{len(store.state['decisions']) + 1:04d}.json"
    survey = {"identity": identity(context, store), "reads": observe_conditions(context, survey_conditions(ordered))}
    # These are planning facts, not completion evidence. Query exact requested
    # inventory names so the reviewer does not invent crafting ratios.
    names = sorted({name.removeprefix("minecraft:") for node in ordered for condition in node.criteria
                    if condition.kind == "inventory" for name in condition.names if not any(c in name for c in "*?[" )})
    survey["recipes"] = recipe_reads(context, names)
    survey["tool_contracts"] = [row for row in context.runtime.tools["minecraft_tool_schemas"]()
                                if row["name"] in {"minecraft_recipe_search", "minecraft_smelt_item", "minecraft_eat_best"}]
    store.write(survey_path, survey)
    store.state["planning_survey"] = survey_path
    store.save()
    common = {
        "identity": identity(context, store), "goal": store.state["goal"], "tree": tree.model_dump(mode="json"),
        # The reviewer needs the same caller-owned coverage as the planner.
        # Separate publication audits cannot be inferred from static cells.
        "plan_constraints": caller_plan_constraints(store),
        "latest_project_check": project_check_context(store),
        "server_rules": json.loads(store.path_for(store.state["server_rules"]).read_text(encoding="utf-8")).get("server_rules", "") if store.state.get("server_rules") else "",
        "leaf_prerequisites": {key: sorted(value) for key, value in validate_tree(tree).items()},
        "accepted_nodes": list(store.state["acceptances"]), "active_recovery": gameplay_history(store)["recovery"],
        "project_initial_observation": json.loads(store.path_for(store.state["initial_observation"]).read_text(encoding="utf-8")),
        "original_observation": json.loads(store.path_for(store.state["planning_observation"]).read_text(encoding="utf-8")),
        "observation_path": store.state["planning_observation"],
         "current_site_survey": {"identity": survey["identity"], "reads": scoped_reads(survey["reads"]),
                                 "recipes": scoped_reads(survey["recipes"]), "tool_contracts": survey["tool_contracts"]},
        "survey_path": survey_path}
    topology = [{"id": node.id, "parent": node.parent, "children": node.children,
                 "depends_on": node.depends_on, "kind": node.kind} for node in tree.nodes]
    historical = accepted_inputs(store, {node.id for node in tree.nodes})
    reviews = [{"node_id": key, "status": "accepted", "scope": "Original independently verified historical attainment.",
                "original_receipt": value["certificate"]["verification"]} for key, value in historical.items()]
    for node in ordered:
        if not boundary(context, store):
            return {"status": "refine", "summary": "Project instructions changed during scoped plan review."}
        selected = dependency_scope(tree, [node.id])
        scoped = {**common, "tree": {"goal_id": tree.goal_id, "revision": tree.revision,
                                    "nodes": [node.model_dump(mode="json")]},
                  "tree_topology": topology, "review_scope": {"node_id": node.id, "kind": node.kind},
                  "dependency_specifications": [row.model_dump(mode="json") for row in tree.nodes
                                                if row.id in selected and row.id != node.id and row.kind == "leaf" and not store.valid(row.id)],
                  "accepted_dependencies": accepted_inputs(store, selected), "preceding_reviews": reviews,
                  "current_site_survey": {**common["current_site_survey"],
                      "reads": scoped_reads(observe_conditions(context, survey_conditions([node])))}}
        result = reason(context, store, "reviewer", PlanReview, scoped)
        if not boundary(context, store):
            return {"status": "refine", "summary": "Project instructions changed during scoped plan review."}
        reviews.append({"node_id": node.id, **result.model_dump(mode="json"),
                        "original_receipt": store.state["decisions"][-1]["path"] + "/output.json"})
        if result.status != "accepted":
            if not result.target_id:
                result.target_id = node.id
            store.state["review_target"] = result.target_id
            break
    if not boundary(context, store):
        return {"status": "refine", "summary": "Project instructions changed during plan review."}
    record_review(store, result)
    return {"status": result.status, "summary": result.summary}


def record_review(store: TreeStore, result: PlanReview) -> None:
    """Keep every review outcome available to later explicit recovery."""
    if result.status == "refine":
        if result.target_id not in {node.id for node in store.load().nodes}:
            raise ValueError("The reviewer named an unknown refinement target.")
        store.state.update(feedback=result.summary, refinement_target=result.target_id)
    elif result.status == "accepted":
        store.state.update(reviewed_revision=store.state["revision"], feedback="", refinement_target="")
        store.state.pop("review_target", None)
    else:
        store.state.update(feedback=result.summary, refinement_target="")
    store.save()


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


def execution_observations(context: RoleContext, store: TreeStore, node: Node, attempt: dict) -> dict:
    """Give one native actor original current reads for its exact predicates.

    The reads use the normal serialized, cancellable world boundary. They
    change no world state and cannot prove ordered effects or accept a task.
    Store the complete source inside the actor's readable inbox. Independent
    verification still obtains new reads after the completion candidate.
    """
    reads = observe_conditions(context, node.criteria)
    proof = {"schema": "pm.minecraft-execution-observations.v1",
             "identity": identity(context, store, node=node, actor_attempt=attempt["id"]),
             "scope": "Current read-only predicates. Required actor effects, ordered travel, durable notes, and independent acceptance remain separate.",
             "reads": reads, "checks": evaluate(node.criteria, reads, certificates=store.state["acceptances"])}
    relative = attempt["path"] + "/inbox/criterion-observations.json"
    store.write(relative, proof)
    return {**proof, "reads": scoped_reads(reads), "original_receipt": "inbox/criterion-observations.json",
            "original_digest": digest(proof)}


def act(context: RoleContext, store: TreeStore) -> dict:
    """Dispatch a fresh native actor child for the exact selected revision."""
    if not boundary(context, store):
        return {"status": "replan", "summary": "Project instructions changed before specialist dispatch."}
    selected = store.state["selection"]
    node = next(node for node in store.load().nodes if node.id == selected["leaf_id"])
    if node.id not in store.eligible():
        raise ValueError("The selected leaf lost its prerequisite acceptance.")
    attempt_id = f"actor-{len(store.state['attempts']) + 1:04d}"
    child_id = context.task_id + "-" + attempt_id
    relative = "attempts/" + attempt_id
    prerequisites = store.requirements(node.id)
    attempt = {**selected, "actor_attempt": attempt_id, "id": attempt_id, "path": relative, "started": time.time(), "state": "running",
               "prerequisites": {key: {"verification": store.state["acceptances"][key]["verification"],
                                       "certificate_digest": digest(store.state["acceptances"][key])}
                                 for key in prerequisites}}
    store.state["attempts"].append(attempt)
    store.state["selection"].update(actor_attempt=attempt_id)
    store.save()
    req = request(context)
    child_environment = FolderEnvironment(store.root / relative, "bash")
    parent_boundary = context.runtime.tools["managed_instructions"]
    parent_revision = store.state["instruction_revision"]
    def instructions(wait=False, close=False):
        current = parent_boundary()
        if current["revision"] != parent_revision:
            raise InstructionsChanged("The project instruction changed. Replan from retained world effects.")
        return {**current, "instruction": node.instruction}
    instructions.authorize = getattr(parent_boundary, "authorize", None)
    instructions.cancelled = getattr(parent_boundary, "cancelled", None)
    endpoint = context.runtime.tools["minecraft_endpoint"]()
    tools = {**context.runtime.tools, "managed_instructions": instructions,
             "minecraft_lease": lambda: BODY_LEASES.acquire(endpoint, child_id)}
    child_stop = LinkedStopEvent(context.runtime.stop_event)
    child_runtime = replace(context.runtime, environment=child_environment, tools=tools, stop_event=child_stop)
    nodes = {row.id: row for row in store.load().nodes}
    prerequisite_context = {}
    for key, certificate in attempt["prerequisites"].items():
        verified = json.loads(store.path_for(certificate["verification"]).read_text(encoding="utf-8"))
        prerequisite_context[key] = {**certificate, "specification": nodes[key].model_dump(mode="json"),
                                     "identity": verified["identity"], "checks": verified["checks"]}
        source_actor = next(row for row in store.state["attempts"] if row["id"] == verified["identity"]["actor_attempt"])
        source_report = yaml.safe_load(store.path_for(source_actor["report"]).read_text(encoding="utf-8"))
        prerequisite_context[key]["actor_summary"] = source_report["summary"]
    ancestors, cursor = [], node.parent
    while cursor is not None:
        ancestors.append(nodes[cursor].model_dump(mode="json"))
        cursor = nodes[cursor].parent
    history = gameplay_history(store)
    store.write(relative + "/inbox/gameplay-history.json", history)
    acceptance = {"criteria": [row.model_dump(mode="json") for row in node.criteria],
                  "current_criterion_observations": execution_observations(context, store, node, attempt),
                  "prerequisites": prerequisite_context,
                  "project": leaf_project_context(store, ancestors),
                  "gameplay_history": scoped_history(store, dependency_scope(store.load(), [node.id]), history, {node.id}),
                  "full_gameplay_history": "inbox/gameplay-history.json",
                  "history_journals": [str(store.path_for(row["path"] + "/scratch/actor-journal.jsonl"))
                                       for row in store.state["attempts"][:-1]]}
    earlier = [row for row in store.state["attempts"][:-1] if row["leaf_id"] == node.id]
    if earlier:
        previous = earlier[-1]
        handoff = store.path_for(previous["path"] + "/outbox/handoff.md")
        acceptance["previous_attempt"] = {"identity": previous, "handoff": handoff.read_text(encoding="utf-8") if handoff.is_file() else "",
                                          "journals": [str(store.path_for(row["path"] + "/scratch/actor-journal.jsonl")) for row in earlier],
                                          "scope": "Past work; current specification and fresh observations take precedence."}
    payload = MinecraftRequest(goal=node.instruction, instruction=node.instruction, body_id=req.body_id,
        specialist=node.specialist, persona_context=req.persona_context, include_image=req.include_image,
        acceptance=acceptance,
        task_reference={**selected, "actor_attempt": attempt_id}).model_dump(mode="json")
    runner = WorkflowRunner(ROOT / "workflows/minecraft/minecraft.workflow.md", store.root / "kernel-children", namespace="minecraft_leaf")
    try:
        result = runner.invoke(WorkflowRequest(child_id, payload), child_runtime)
    except InstructionsChanged as error:
        attempt.update(state="stale", summary=str(error))
        store.save()
        return {"status": "replan", "summary": str(error)}
    except (ValidationError, json.JSONDecodeError) as error:
        # Native output is untrusted. Cleanup has finished before invoke raises.
        # Keep the rejected candidate and world calls; never invent missing data.
        attempt.update(state="invalid", summary=f"{type(error).__name__}: {error}",
                       journal=relative + "/scratch/actor-journal.jsonl")
        store.save()
        return {"status": "replan", "summary": "The native completion contract was rejected. "
                "Obtain fresh world state and preserve prior effects before a new attempt. " + attempt["summary"]}
    context.runtime.check_cancelled()
    if result.status != WorkflowStatus.Completed:
        journal_path = relative + "/scratch/actor-journal.jsonl"
        rows = [json.loads(line) for line in store.path_for(journal_path).read_text(encoding="utf-8").splitlines()] if store.path_for(journal_path).is_file() else []
        errors = [row for row in rows if row["event"] == "uncertain"]
        attempt.update(state="uncertain" if errors else "failed", summary=str(result.kernel_summary), journal=journal_path, errors=errors)
        store.save()
        detail = errors[-1]["error"] if errors else str(result.kernel_summary)
        return {"status": "waiting" if errors else "replan", "summary":
                "The actor stopped without acceptance. Restore the service, then observe retained effects before a new method. " + detail}
    report = result.report
    attempt.update(state=report["outcome"]["status"], summary=report["summary"],
                   session=report["agent_session"], report=relative + "/outbox/report.yaml",
                   journal=relative + "/scratch/actor-journal.jsonl")
    final = report["final_observation"]
    attempt["progress_digest"] = digest(progress_facts(final))
    store.save()
    from workflows.minecraft.steps import build_world
    req.goal = store.state["goal"]
    world = build_world(req, {"result": report["final_observation"]}, None, [])
    world["outcome"] = "progress"
    world["episodes"][0].update(goal=node.instruction, outcome="unverified", source_attempt=attempt_id)
    world["episodes"][0]["gameplay"] = report.get("gameplay", {})
    context.runtime.environment.write_text("outbox/world.yaml", yaml.safe_dump(world, allow_unicode=True, sort_keys=False))
    publish = context.runtime.tools.get("managed_milestone")
    if publish is not None:
        publish(attempt_id, report["summary"], attempt["journal"])
    if not boundary(context, store):
        return {"status": "replan", "summary": "The project changed before actor completion was accepted."}
    routes = {"complete": "candidate", "yield": "progress", "replan": "replan", "blocked": "waiting"}
    prior = [row for row in store.state["attempts"][:-1] if row["leaf_id"] == node.id and row.get("progress_digest")]
    if report["outcome"]["status"] == "yield" and prior and prior[-1]["progress_digest"] == attempt["progress_digest"]:
        return {"status": "replan", "summary": "The repeated leaf attempt made no observed progress. Change the method."}
    return {"status": routes[report["outcome"]["status"]], "summary": report["summary"]}


def verify_node(context: RoleContext, store: TreeStore, node: Node, actor_attempt: str) -> dict:
    """Require original actor provenance and fresh independent world reads."""
    verifier_id = f"verifier-{len(store.state['decisions']) + 1:04d}-{node.id}"
    ident = identity(context, store, node=node, actor_attempt=actor_attempt, verifier_attempt=verifier_id)
    reads = observe_conditions(context, node.criteria)
    checks = evaluate(node.criteria, reads, certificates=store.state["acceptances"])
    relative = f"verification/{verifier_id}.json"
    # Persist original reads before inference so an interruption retains evidence.
    store.write(relative, {"schema": "pm.minecraft-leaf-verification.v1", "identity": ident,
                           "verified": False, "reads": reads, "checks": checks, "review": {"status": "pending"}})
    review_result = reason(context, store, "verifier", VerificationReview, {
        "identity": ident, "specification": node.model_dump(mode="json"), "reads": scoped_reads(reads), "checks": checks,
        "original_receipt": relative})
    complete_ids = {row.id for row in node.criteria}
    verified = (all(checks.values()) and review_result.status == "accepted"
                and set(review_result.criterion_ids) == complete_ids and boundary(context, store))
    files = []
    if node.kind == "leaf":
        attempt = next(row for row in store.state["attempts"] if row["id"] == actor_attempt)
        actor = yaml.safe_load((store.root / attempt["report"]).read_text(encoding="utf-8"))
        journal = [json.loads(line) for line in (store.root / attempt["journal"]).read_text(encoding="utf-8").splitlines()]
        dispatches = {row["seq"]: row for row in journal if row["event"] == "dispatch"}
        results = [row for row in journal if row["event"] == "result"]
        verified = (verified and actor["outcome"]["status"] == "complete" and bool(results)
                    and actor["task_reference"] == {**store.state["selection"], "actor_attempt": actor_attempt}
                    and all(row["seq"] in dispatches and dispatches[row["seq"]]["tool"] == row["tool"]
                            and dispatches[row["seq"]]["arguments"] == row["arguments"] for row in results)
                    and len({row["seq"] for row in results}) == len(results)
                    and len(dispatches) == len(results))
        files = [attempt["report"], attempt["journal"]]
    store.write(relative, {"schema": "pm.minecraft-leaf-verification.v1", "identity": ident,
        "verified": verified, "reads": reads, "checks": checks, "review": review_result.model_dump(mode="json"),
        "outcome": "completion_evidence" if verified else "unmet_completion_conditions"})
    if verified:
        store.accept(node.id, relative, files)
        return {"status": "accepted", "summary": review_result.summary}
    return {"status": "replan" if review_result.status == "accepted" else review_result.status,
            "summary": "Independent verification failed: " + review_result.summary}


def verify(context: RoleContext, store: TreeStore) -> dict:
    """Verify the exact completion candidate before releasing dependencies."""
    selected = store.state["selection"]
    node = next(node for node in store.load().nodes if node.id == selected["leaf_id"])
    return verify_node(context, store, node, selected["actor_attempt"])


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
        context.runtime.environment.write_text("outbox/world.yaml", yaml.safe_dump(world, allow_unicode=True, sort_keys=False))
    report = {"schema": "pm.minecraft-project-report.v1", "accepted": True, "verified": evidence["verified"],
              "summary": "Reviewed plan retained without execution." if req.plan_only else "The project passed independent world verification.",
              "goal": store.state["goal"], "tree_revision": store.state["revision"], "verification": evidence,
              "plan_only": req.plan_only, "attempts": store.state["attempts"], "body_id": req.body_id}
    context.runtime.environment.write_text("outbox/report.yaml", yaml.safe_dump(report, allow_unicode=True, sort_keys=False))
    return {"status": "done", "summary": report["summary"]}


def _run(context: RoleContext) -> dict:
    """Route one declared role and retain its explicit failure or feedback."""
    store = store_for(context)
    if context.phase == "replan":
        return plan(context, store)
    if context.phase == "review":
        result = review(context, store)
        if result["status"] == "accepted" and request(context).plan_only:
            result["status"] = "inspect"
        return result
    operation = {"retain": retain, "plan": plan, "survey": survey, "select": select, "act": act, "verify": verify, "finish": finish}[context.phase]
    result = operation(context, store)
    if result["status"] in {"replan", "waiting", "blocked"}:
        store.state["feedback"] = result["summary"]
        if result["status"] == "replan":
            store.state["require_leaf_change"] = True
        store.save()
    return result


def run(context: RoleContext) -> dict:
    """Run a role without the outer runner replaying it for a file lock.

    Local evidence commits retry only their file operation. Other persistent
    permission errors fail the project. Its original world receipts remain.
    """
    try:
        return _run(context)
    except PermissionError as error:
        raise RuntimeError(f"A11 file access failed in {context.phase}. Do not replay world work: {error}") from error
