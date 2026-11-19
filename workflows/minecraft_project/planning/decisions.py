from __future__ import annotations

from dataclasses import replace
import hashlib
from pm_workflows.python_role import RoleContext
from pydantic import ValidationError, TypeAdapter, JsonValue
from pm.constants import Sampler
from pm.context_render.prompt import RendererCapture, RendererInput
from pm.llm.role_model import role_model
from pm.llm.generation import GenerationSettings, ResponseFormat, ResponseSchema
from pm.llm.prompt_encoding import OpenAIChatEncoder
from pm.llm.structured import parse_response, JsonOutputError
from workflows.minecraft_project.schemas import EvidenceDecision, ReviewValidationContext
from workflows.minecraft_project.tree_store import TreeStore, digest
from workflows.minecraft_project.evidence_selection import catalog, read_source
from workflows.minecraft_project.planning.run_context import RESOURCES
from workflows.minecraft_project.planning.context_contracts import (
    DecisionContext, ReviewContext, DecisionRole, ArchiveReference, SourceBinding, SourceRead, EvidenceCatalog,
    SourceCatalogArchive, SourceReadBatch, DecisionInputReceipt, RejectedDecision, RawDecision)
from workflows.minecraft_project.planning.rendering import ProjectDecisionRenderInput, render_project_decision


def decision_schema(contract: type[EvidenceDecision]) -> dict[str, JsonValue]:
    schema = contract.model_json_schema()
    node = schema.get("$defs", {}).get("Node")
    if node is not None:
        node["required"] = [key for key in node["properties"] if key != "revision"]
    return TypeAdapter[dict[str, JsonValue]](dict[str, JsonValue]).validate_python(schema)


def reason[T: EvidenceDecision](context: RoleContext, store: TreeStore, role: DecisionRole,
                               contract: type[T], payload: DecisionContext) -> T:
    runtime = context.runtime
    if runtime is None:
        raise RuntimeError("Project reasoning requires its workflow runtime.")
    selected = payload.model_copy(deep=True)
    archive = f"planning/context/{len(store.state['decisions']) + 1:04d}-full-input.json"
    store.write(archive, payload)
    sources = catalog(context, store)
    archive_path = store.path_for(archive)
    sources["decision_context"] = SourceBinding(path=str(archive_path.resolve()),
        sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest())
    relative = f"planning/context/{len(store.state['decisions']) + 1:04d}-catalog.json"
    source_catalog = SourceCatalogArchive(sources=sources)
    store.write(relative, source_catalog)
    index = store.path_for(relative)
    sources["evidence_index"] = SourceBinding(path=str(index.resolve()), sha256=hashlib.sha256(index.read_bytes()).hexdigest())
    groups: dict[str, int] = {}
    for name in sources:
        group = name.split('/')[0]
        groups[group] = groups.get(group, 0) + 1
    selected.evidence_catalog = EvidenceCatalog(index_source="evidence_index", source_groups=groups,
        named_sources=tuple(name for name in sources if '/' not in name))
    selected.evidence_catalog_receipt = ArchiveReference(original_receipt=relative, original_digest=digest(source_catalog))
    seen: set[str] = set()
    while True:
        result = reason_once(context, store, role, contract, selected)
        if not result.evidence_requests:
            return result
        reads: list[SourceRead] = []
        for query in result.evidence_requests:
            key = digest(query)
            if key in seen:
                raise ValueError("The decision repeated an identical evidence request without progress.")
            seen.add(key)
            reads.append(read_source(sources, query))
        relative = store.state["decisions"][-1]["path"] + "/retrieved.json"
        store.write(relative, SourceReadBatch(reads=tuple(reads)))
        selected.retrieved_evidence = (*selected.retrieved_evidence, *reads)
        runtime.check_cancelled()


def reason_once[T: EvidenceDecision](context: RoleContext, store: TreeStore, role: DecisionRole,
                                    contract: type[T], payload: DecisionContext) -> T:
    runtime = context.runtime
    if runtime is None:
        raise RuntimeError("Project reasoning requires its workflow runtime.")
    runtime.check_cancelled()
    schema = decision_schema(contract)
    name = f"decisions/{len(store.state['decisions']) + 1:04d}-{role.value}"
    store.write(name + "/unscoped-context.json", payload)
    selected = payload.model_copy(deep=True)
    selected.context_archive = ArchiveReference(original_receipt=name + "/unscoped-context.json",
        original_digest=digest(payload), catalog_source="decision_context")
    system = ((RESOURCES / f"{role.value}.md").read_text(encoding="utf-8")
        + "\nUse evidence_requests to inspect an original catalog source and numbered lines or query. "
        "Read evidence_index for exact source names. Retrieval requests do not commit a plan or acceptance. "
        "After retrieval, return your final decision with evidence_requests empty. "
        "The response_format defines the output schema.")
    generation = GenerationSettings(enable_thinking=False, temperature=Sampler.TempInstruct,
        response_format=ResponseFormat(type="json_schema", json_schema=ResponseSchema(
            name=contract.__name__, strict=True, schema_body=schema)))
    session = role_model(runtime)
    settings = session.renderer_settings_for(generation)
    input = ProjectDecisionRenderInput("Minecraft " + role.value, system, (), context=selected)
    with RendererCapture(session.capture_settings("minecraft.decision"), RendererInput("minecraft.decision", input, settings)) as capture:
        prompt = render_project_decision(input, settings)
        capture.save_output(prompt)
    wire_messages = TypeAdapter[tuple[JsonValue, ...]](tuple[JsonValue, ...]).validate_python(OpenAIChatEncoder().encode(prompt.turns)["messages"])
    store.write(name + "/input.json", DecisionInputReceipt(identity=selected.identity, messages=wire_messages,
        context=selected, input_tokens=prompt.estimated_tokens, input_ceiling=prompt.input_ceiling))
    store.state["decisions"].append({"role": role.value, "path": name, "identity": selected.identity.model_dump(mode="json")})
    store.save()
    validation_context = (ReviewValidationContext(node_ids=frozenset(node.id for node in payload.tree_topology))
        if isinstance(payload, ReviewContext) else None)
    attempt = 0
    while True:
        text = session.complete(prompt.turns, settings=generation)
        store.write(name + "/raw.json", RawDecision(text=text))
        runtime.check_cancelled()
        try:
            result = parse_response(text, contract, validation_context=validation_context)
            break
        except (JsonOutputError, ValidationError) as error:
            attempt += 1
            store.write(name + f"/rejection-{attempt:02d}.json", RejectedDecision(text=text, error=str(error)))
            input = replace(input, rejected_answer=text, problem=str(error))
            with RendererCapture(session.capture_settings("minecraft.decision"), RendererInput("minecraft.decision", input, settings)) as capture:
                prompt = render_project_decision(input, settings)
                capture.save_output(prompt)
    store.write(name + "/output.json", result)
    return result
