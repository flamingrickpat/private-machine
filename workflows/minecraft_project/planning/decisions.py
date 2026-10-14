"""Model decisions of the project roles: decision schemas, the retrieval loop, the token budget, and context shrinking."""
from __future__ import annotations

import hashlib
import json

from pm_workflows.python_role import RoleContext
from pydantic import BaseModel
from pm.context_render import render_context
from pm.context_render.turns import Role, Turn
from pm.llm.prompt_encoding import OpenAIChatEncoder

from workflows.minecraft_project.schemas import Node, TaskTree
from workflows.minecraft_project.tree_store import TreeStore, digest
from workflows.minecraft_project.evidence_selection import catalog, read_source
from workflows.minecraft_project.planning.run_context import Decision, RESOURCES


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
    archive = f"planning/context/{len(store.state['decisions']) + 1:04d}-full-input.json"
    store.write(archive, payload)
    sources = catalog(context, store)
    archive_path = store.path_for(archive)
    sources["decision_context"] = {"path": str(archive_path.resolve()),
                                   "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest()}
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
    payload = decision_context(payload)
    name = f"decisions/{len(store.state['decisions']) + 1:04d}-{role}"
    store.write(name + "/unscoped-context.json", original_payload)
    payload["context_archive"] = {"original_receipt": name + "/unscoped-context.json",
                                  "original_digest": digest(original_payload), "catalog_source": "decision_context"}
    system = ((RESOURCES / f"{role}.md").read_text(encoding="utf-8")
              + "\nTo inspect an original source, set evidence_requests with its catalog source ID and numbered line window or query. "
              "Read evidence_index to find original source IDs when the selected contract does not supply their names. "
              "That response requests retrieval only; no plan or acceptance from it is committed. "
              "After retrieval, return your final decision with evidence_requests empty."
              + "\nThe response_format defines the output schema.")
    messages = [Turn(Role.System, system),
                Turn(Role.User, render_context(payload, schema_path=RESOURCES / "llm_context.json"))]
    generation = {"enable_thinking": False, "temperature": 0.1,
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
    measurement = measure(messages, cancel=context.runtime.stop_event, **generation)
    if measurement.get("measurement_complete") is False:
        raise ValueError("The project input contains unmeasured media. Its text estimate cannot establish the complete input budget.")
    input_characters = sum(len(row.content.text) for row in messages)
    store.write(name + "/input.json", {"identity": payload["identity"], "messages": OpenAIChatEncoder().encode(messages)["messages"], "context": payload,
                                        "generation": generation, "input_characters": input_characters,
                                        "context_budget": measurement})
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


def decision_context(payload: dict) -> dict:
    """Select technical context before measurement, regardless of context capacity.

    The caller archives the full input before inference. Evidence retrieval
    remains available. This function does not change specifications or reads.
    """
    result = scoped_decision_context(compact_plan_context(payload))
    result.pop("persona_context", None)
    return result


def native_decision_context(payload: dict) -> dict:
    """Select an overview for a planner with task and world retrieval tools.

    Current task specifications remain available through task_get. Historical
    method operands and survey outputs remain in the caller's full archive.
    Recovery, recent diagnoses, and failed verification stay visible here.
    """
    result = decision_context(payload)
    tree = result.get("previous_tree")
    if tree:
        result["previous_tree"] = {"goal_id": tree["goal_id"], "revision": tree.get("revision", 1),
            "specifications": "Use task_list and task_get for current draft specifications."}
        result.pop("tree_topology", None)
    result.pop("current_tool_contracts", None)
    initial = result.get("project_initial_observation")
    if initial:
        # The original inventory anchors relative gathering targets. The
        # original position anchors the site after the player moves.
        result["project_initial_observation"] = {"inventory": initial.get("inventory", []),
            "camera": {key: value for key, value in initial.get("camera", {}).items() if key == "feet_position"},
            "player": {key: value for key, value in initial.get("player", {}).items() if key == "dimension"}}
    history = result.pop("gameplay_history", None)
    if history:
        result["execution_context"] = {key: history[key] for key in
            ("recovery", "service_wait", "navigation", "outcome_count", "completion_verified") if key in history}
        result["execution_context"]["latest_diagnoses"] = history.get("diagnoses", [])[-2:]
        result["execution_context"]["history_scope"] = "Full methods, searches, and outcomes remain in context_archive. Read them before repeating a failed method."
    survey = result.get("previous_site_survey")
    if survey:
        result["previous_site_survey"] = {key: value for key, value in survey.items()
                                          if key not in {"reads", "recipes", "tool_contracts", "shared_tool_contracts"}}
        result["previous_site_survey"]["history_scope"] = "Historical outputs remain in context_archive. Use world_read for current facts."
    return result


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
    attempt_started = (result.get("latest_attempt") or {}).get("started")
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

