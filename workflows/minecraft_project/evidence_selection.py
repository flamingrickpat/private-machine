"""Select workflow evidence by task dependencies and retain original sources."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from workflows.minecraft_body.journal import read_actor_journal
from workflows.minecraft_body.call_outcomes import GameplayEvidence
from workflows.minecraft_project.schemas import Condition, Node, Renewal, RetentionChoice, RetentionProposal, TaskTree
from workflows.minecraft_project.tree_store import TreeStore, acceptance_requirements, digest, node_digest


def dependency_scope(tree: TaskTree, targets: list[str]) -> set[str]:
    """Include complete target subtrees and recursive acceptance prerequisites.

    Ancestors remain outside this set. Their complete contracts belong in the
    separate integration input. This avoids pulling every sibling into a leaf.
    """
    nodes = {node.id: node for node in tree.nodes}
    requirements = acceptance_requirements(tree)
    selected = set()
    def include(key):
        if key in selected:
            return
        selected.add(key)
        for dependency in requirements[key]:
            include(dependency)
        for child in nodes[key].children:
            include(child)
    for key in targets:
        include(key)
    return selected


def catalog(context, store: TreeStore) -> dict:
    """Bind readable sources to exact paths and hashes without exposing secrets."""
    sources = {}
    for folder in ("planning", "verification", "attempts"):
        for path in sorted((store.root / folder).rglob("*")):
            if path.is_file() and path.suffix in {".json", ".jsonl", ".yaml", ".md"}:
                if path.name.endswith("-catalog.json"):
                    continue
                name = path.relative_to(store.root).as_posix()
                sources[name] = path
    request_path = store.root / "inbox/request.yaml"
    if request_path.exists():
        sources["project_request"] = request_path
    external = context.runtime.tools.get("minecraft_evidence_sources")
    if external is not None:
        sources.update({name: Path(path) for name, path in external().items()})
    return {name: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in sources.items()}


def accepted_inputs(store: TreeStore, keys: set[str]) -> dict:
    """Expose original identities and criterion results for valid dependencies."""
    result = {}
    for key in sorted(keys):
        if not store.valid(key):
            continue
        record = store.state["acceptances"][key]
        proof = json.loads(store.path_for(record["verification"]).read_text(encoding="utf-8"))
        result[key] = {"certificate": record, "checks": proof["checks"], "identity": proof["identity"],
                       "scope": "Historical attainment. Current world truth needs new reads."}
    return result


def read_source(sources: dict, request) -> dict:
    """Read numbered source lines; reject changed or undeclared original files."""
    binding = sources[request.source.replace("\\", "/")]
    path = Path(binding["path"])
    contents = path.read_bytes()
    if hashlib.sha256(contents).hexdigest() != binding["sha256"]:
        raise ValueError("An original evidence source changed during inference: " + request.source)
    lines = contents.decode("utf-8-sig").splitlines()
    if request.query:
        matches = [index for index, line in enumerate(lines) if request.query.casefold() in line.casefold()]
        start = matches[0] if matches else len(lines)
    else:
        start = request.start_line - 1
    return {"source": request.source, "sha256": binding["sha256"], "total_lines": len(lines),
            "start_line": start + 1, "lines": lines[start:start + request.line_count],
            "query": request.query}


def review_nodes(store: TreeStore) -> list[Node]:
    """Select unfinished contracts and the full goal for current plan review.

    Valid historical attainment already has an independent review of its exact
    specification. Its execution instructions belong to that original time.
    Root integration remains current even when its certificate is retained.
    """
    nodes = store.load().nodes
    return ([node for node in nodes if node.kind == "leaf" and not store.valid(node.id)]
            + [node for node in nodes if node.kind == "story" and not store.valid(node.id)]
            + [node for node in nodes if node.kind == "goal"])


def protect_accepted_specifications(store: TreeStore, proposed: TaskTree) -> None:
    """Reject ordinary repair that replaces or removes valid attained work.

    New requirements need new node identities. A fresh assumption failure can
    revoke old proof before repair. Exact historical retention has its separate
    renewal validation and does not call this ordinary repair guard.
    The root integration can gain new required work under independent review.
    """
    nodes = {node.id: node for node in proposed.nodes}
    original = {node.id: node for node in store.load().nodes}
    changed = [key for key, certificate in store.state["acceptances"].items()
               if original[key].kind != "goal" and store.valid(key)
               and (key not in nodes or node_digest(nodes[key]) != certificate["spec_digest"])]
    if changed:
        raise ValueError("Valid attained specifications must remain exact; add separate current requirements: " + ", ".join(sorted(changed)))


def scoped_history(store: TreeStore, selected: set[str], full: dict, targets: set[str] | None = None) -> dict:
    """Keep relevant methods and all global safety and transport boundaries."""
    from workflows.minecraft_project.steps import scoped_gameplay
    evidence = GameplayEvidence()
    journals = []
    methods = {key for key in selected if not store.valid(key)} | (targets or set())
    for attempt in store.state["attempts"]:
        if attempt["leaf_id"] not in methods:
            continue
        relative = attempt.get("journal", attempt["path"] + "/scratch/actor-journal.jsonl")
        journals.append(relative)
        for row in read_actor_journal(store.path_for(relative)):
            evidence.consume(row, relative)
    result = scoped_gameplay(evidence.snapshot())
    for field in ("recovery", "service_wait", "life", "failed_recoveries"):
        result[field] = full[field]
    result["scope"] = {"node_ids": sorted(selected), "method_node_ids": sorted(methods), "journals": journals,
                       "meaning": "Selected task methods; original global journals remain available through retrieval."}
    return result


def retention_operation(store: TreeStore, choice: RetentionChoice) -> RetentionProposal:
    """Copy superseded valid requirements into fresh exact renewal identities.

    This operation derives schema fields from accepted state, without model
    arithmetic or field copying. It changes no files or certificates.
    """
    nodes = {node.id: node for node in store.load().nodes}
    candidates = {row["specification"]["id"]: row for row in store.acceptance_candidates()}
    renewals = []
    for key in choice.restore_nodes:
        if key not in nodes or key not in candidates:
            raise ValueError("Retention selected an unknown original certificate: " + key)
        if not store.valid(key) or node_digest(nodes[key]) == candidates[key]["certificate"]["spec_digest"]:
            continue
        new_id = key + "_renewal"
        suffix = 2
        while new_id in nodes:
            new_id = key + "_renewal_" + str(suffix)
            suffix += 1
        node = nodes[key].model_copy(update={"id": new_id, "revision": 1}, deep=True)
        renewals.append(Renewal(source_node_id=key, node=node))
        nodes[new_id] = node
    return RetentionProposal(**choice.model_dump(mode="json"), renewals=renewals)


def retention_tree(store: TreeStore, proposal: RetentionProposal) -> TaskTree:
    """Validate a native restoration without changing files or acceptance state.

    Replaced valid leaves need new identities with identical current contracts.
    Parent integration gains their acceptance criteria. Historical proof keeps
    every original specification and dependency hash through normal restoration.
    """
    if proposal.status != "planned" or not proposal.restore_nodes or len(set(proposal.restore_nodes)) != len(proposal.restore_nodes):
        raise ValueError("Retention needs unique explicit restoration IDs.")
    tree = store.load()
    nodes = {node.id: node for node in tree.nodes}
    candidates = {row["specification"]["id"]: row for row in store.acceptance_candidates()}
    superseded = {key for key in proposal.restore_nodes if store.valid(key)
                  and key in candidates and node_digest(nodes[key]) != candidates[key]["certificate"]["spec_digest"]}
    if {row.source_node_id for row in proposal.renewals} != superseded or len(proposal.renewals) != len(superseded):
        raise ValueError("Every superseded valid prerequisite needs exactly one separate current renewal: " + ", ".join(sorted(superseded)))
    proposed = tree.model_copy(deep=True)
    proposed_nodes = {node.id: node for node in proposed.nodes}
    for renewal in proposal.renewals:
        original, node = nodes[renewal.source_node_id], renewal.node
        if original.kind != "leaf" or node.id in proposed_nodes:
            raise ValueError("A renewal needs a fresh leaf identity.")
        expected = original.model_copy(update={"id": node.id, "revision": node.revision})
        if node_digest(expected) != node_digest(node):
            raise ValueError("A current renewal must preserve the complete superseded specification.")
        parent = proposed_nodes[node.parent]
        parent.children.append(node.id)
        criterion_id = "renewed_" + node.id
        if any(row.id == criterion_id for row in parent.criteria):
            raise ValueError("A renewal criterion identity already exists.")
        parent.criteria.append(Condition(id=criterion_id, description="Fresh attainment of the retained current prerequisite.",
                                         kind="accepted", node_id=node.id))
        proposed.nodes.append(node)
        proposed_nodes[node.id] = node
    return store.restore_specifications(proposed, proposal.restore_nodes)
