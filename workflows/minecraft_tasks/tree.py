"""Validate task graphs and retain revisions, attempts, and acceptance receipts."""
from __future__ import annotations

from graphlib import TopologicalSorter, CycleError
import hashlib
import json
from pathlib import Path
import time

from workflows.minecraft_tasks.schemas import Node, TaskTree


def digest(value) -> str:
    """Hash canonical JSON so unchanged specifications retain their identity."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def node_digest(node: Node) -> str:
    """Exclude the revision counter when comparing specification content."""
    return digest(node.model_dump(exclude={"revision"}))


def validate_tree(tree: TaskTree) -> dict[str, set[str]]:
    """Return leaf prerequisites; reject invalid hierarchy or dependency cycles.

    Dependencies on a story require all its leaves. Ancestor dependencies also
    apply to descendants. Dispatch alone never releases a prerequisite.
    Certificate references require a descendant or declared prerequisite.
    """
    nodes = {node.id: node for node in tree.nodes}
    if len(nodes) != len(tree.nodes) or tree.goal_id not in nodes:
        raise ValueError("The tree needs unique IDs and an existing goal.")
    roots = [node.id for node in tree.nodes if node.parent is None]
    if roots != [tree.goal_id] or nodes[tree.goal_id].kind != "goal":
        raise ValueError("The tree needs exactly one root goal.")
    for node in tree.nodes:
        if len(set(node.children)) != len(node.children) or len(set(node.depends_on)) != len(node.depends_on):
            raise ValueError("Duplicate children or dependencies are invalid.")
        if len({condition.id for condition in node.criteria}) != len(node.criteria):
            raise ValueError("Criterion IDs must be unique within a node.")
        if node.kind == "leaf" and node.children or node.kind != "leaf" and not node.children:
            raise ValueError("Leaves have no children; goals and stories need children.")
        for key in node.children + node.depends_on:
            if key not in nodes:
                raise ValueError(f"Missing task dependency or child: {key}")
        for child in node.children:
            if nodes[child].parent != node.id:
                raise ValueError("Parent and child references disagree.")
        if node.parent is not None and (node.parent not in nodes or node.id not in nodes[node.parent].children):
            raise ValueError("A node needs a matching parent reference.")
    try:
        list(TopologicalSorter({key: set(node.children) for key, node in nodes.items()}).static_order())
    except CycleError as error:
        raise ValueError("The task hierarchy has a cycle.") from error
    reached = set()
    def descendants(key):
        reached.add(key)
        node = nodes[key]
        return {key} if node.kind == "leaf" else set().union(*(descendants(child) for child in node.children))
    descendants(tree.goal_id)
    if reached != nodes.keys():
        raise ValueError("All nodes must belong to the root goal.")
    graph = {}
    for key, node in nodes.items():
        if node.kind != "leaf":
            continue
        dependencies = set()
        cursor = key
        while cursor is not None:
            for dependency in nodes[cursor].depends_on:
                dependencies.update(descendants(dependency))
            cursor = nodes[cursor].parent
        graph[key] = dependencies
    try:
        list(TopologicalSorter(graph).static_order())
    except CycleError as error:
        raise ValueError("The task dependencies have a cycle.") from error
    for node in tree.nodes:
        for condition in node.criteria + node.assumptions:
            if condition.kind != "accepted":
                continue
            source = condition.node_id
            if source not in nodes or source == node.id:
                raise ValueError("An acceptance reference needs a different existing node.")
            prerequisites = set()
            cursor = node.id
            while cursor is not None:
                for dependency in nodes[cursor].depends_on:
                    prerequisites.update(descendants(dependency))
                cursor = nodes[cursor].parent
            if node.kind == "leaf":
                allowed = descendants(source) <= graph[node.id]
            else:
                subtree = set()
                def collect(key):
                    for child in nodes[key].children:
                        subtree.add(child)
                        collect(child)
                collect(node.id)
                allowed = source in subtree or descendants(source) <= prerequisites
            if not allowed:
                raise ValueError("Acceptance references must name descendants or declared prerequisites.")
    try:
        list(TopologicalSorter(acceptance_requirements(tree, graph)).static_order())
    except CycleError as error:
        raise ValueError("The acceptance certificates have a cycle.") from error
    return graph


def acceptance_requirements(tree: TaskTree, leaf_graph: dict | None = None) -> dict[str, set[str]]:
    """Guard child outcomes, declared dependencies, and cited certificates.

    A refined story retains its own acceptance contract. Its children alone
    cannot release work that depends on the story's verified outcome.
    """
    if leaf_graph is None:
        leaf_graph = validate_tree(tree)
    nodes = {node.id: node for node in tree.nodes}
    result = {}
    for key, node in nodes.items():
        required = set(leaf_graph[key]) if node.kind == "leaf" else set(node.children)
        required.update(row.node_id for row in node.criteria if row.kind == "accepted")
        cursor = key
        while cursor is not None:
            required.update(nodes[cursor].depends_on)
            required.update(row.node_id for row in nodes[cursor].assumptions if row.kind == "accepted")
            cursor = nodes[cursor].parent
        result[key] = required
    return result


class TreeStore:
    """Own one task folder's specification and evidence index.

    Writes use atomic replacement. Immutable snapshots and attempt directories
    survive cancellation. Managed worker locks prevent concurrent writers.
    A replacement actor observes retained world effects before choosing work.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.path = self.root / "planning/state.json"
        self.state = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {
            "revision": 0, "acceptances": {}, "attempts": [], "decisions": [], "feedback": "", "instruction_revision": -1}

    def write(self, relative: str, value) -> None:
        """Commit JSON atomically; retry a file lock without replaying a role.

        A persistent lock raises RuntimeError. PermissionError would trigger
        the runner's phase retry, which could repeat completed world work.
        """
        path = self.path_for(relative)
        temporary = path.with_suffix(path.suffix + ".tmp")
        contents = json.dumps(value, indent=2, ensure_ascii=False)
        for attempt in range(5):
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary.write_text(contents, encoding="utf-8")
                temporary.replace(path)
                return
            except PermissionError as error:
                if attempt == 4:
                    raise RuntimeError(f"Task evidence commit is locked: {path}. Do not replay this phase.") from error
                time.sleep(0.05 * (attempt + 1))

    def path_for(self, relative: str) -> Path:
        """Reject evidence paths outside the owning task directory."""
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError("The task evidence path escaped its owner.")
        return path

    def save(self) -> None:
        """Commit the current evidence index before another external dispatch."""
        self.write("planning/state.json", self.state)

    def load(self) -> TaskTree:
        """Load the exact accepted specification snapshot."""
        return TaskTree.model_validate(json.loads((self.root / f"planning/specs/{self.state['revision']:04d}.json").read_text(encoding="utf-8")))

    def revise(self, proposed: TaskTree, *, target: str = "") -> TaskTree:
        """Keep stable IDs and invalidate only changed work and its dependents.

        Refinement can replace a selected leaf with a story and new descendants.
        It must preserve the selected acceptance conditions and unrelated nodes.
        Runtime code assigns revisions; model counters have no authority.
        """
        proposed = proposed.model_copy(deep=True)
        validate_tree(proposed)
        old = {node.id: node for node in self.load().nodes} if self.state["revision"] else {}
        new = {node.id: node for node in proposed.nodes}
        if old and (not old.keys() <= new.keys() or proposed.goal_id != self.load().goal_id):
            raise ValueError("A revision cannot remove stable nodes or replace the goal ID.")
        if target:
            selected = old[target]
            permitted = {target}
            def collect(key):
                permitted.add(key)
                for child in new[key].children:
                    collect(child)
            collect(target)
            for key in old.keys() - permitted:
                if node_digest(old[key]) != node_digest(new[key]):
                    raise ValueError("Refinement changed an unrelated node.")
            if [row.model_dump() for row in selected.criteria] != [row.model_dump() for row in new[target].criteria]:
                raise ValueError("Refinement must preserve the selected acceptance conditions.")
        for key, node in new.items():
            if key in old and node.parent != old[key].parent:
                raise ValueError("A revision cannot move an existing node to another parent.")
            node.revision = old[key].revision + (node_digest(old[key]) != node_digest(node)) if key in old else 1
        proposed.revision = self.state["revision"] + 1
        if self.path_for(f"planning/specs/{proposed.revision:04d}.json").exists():
            raise ValueError("A retained specification snapshot already owns this revision.")
        self.write(f"planning/specs/{proposed.revision:04d}.json", proposed.model_dump(mode="json"))
        self.state["revision"] = proposed.revision
        self.state.pop("reviewed_revision", None)
        self.invalidate()
        self.save()
        return proposed

    def valid(self, key: str, visiting: set | None = None) -> bool:
        """Recompute provenance and prerequisite guards for one acceptance."""
        record = self.state["acceptances"].get(key)
        if not record:
            return False
        nodes = {node.id: node for node in self.load().nodes}
        if key not in nodes or record["spec_digest"] != node_digest(nodes[key]):
            return False
        expected_dependencies = self.requirements(key)
        if set(record["dependencies"]) != expected_dependencies:
            return False
        visiting = (visiting or set()) | {key}
        for relative, expected in record["files"].items():
            path = (self.root / relative).resolve()
            if not path.is_relative_to(self.root.resolve()) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                return False
        for dependency, expected in record["dependencies"].items():
            if dependency in visiting or not self.valid(dependency, visiting):
                return False
            if digest(self.state["acceptances"][dependency]) != expected:
                return False
        return True

    def invalidate(self) -> list[str]:
        """Remove invalid acceptance from the active index; retain original files."""
        invalid = [key for key in self.state["acceptances"] if not self.valid(key)]
        for key in invalid:
            del self.state["acceptances"][key]
        return invalid

    def eligible(self) -> list[str]:
        """Return unfinished leaves whose prerequisites have verified evidence."""
        graph = validate_tree(self.load())
        return [key for key in TopologicalSorter(graph).static_order()
                if not self.valid(key) and all(self.valid(dependency) for dependency in self.requirements(key))]

    def requirements(self, key: str) -> set[str]:
        """Return every certificate that must remain valid for this node."""
        return acceptance_requirements(self.load())[key]

    def ready_aggregates(self) -> list[str]:
        """Verify completed stories before dependents can consume their stock."""
        tree = self.load()
        nodes = {node.id: node for node in tree.nodes}
        graph = acceptance_requirements(tree)
        return [key for key in TopologicalSorter(graph).static_order()
                if nodes[key].kind != "leaf" and not self.valid(key)
                and all(self.valid(dependency) for dependency in graph[key])]

    def accept(self, key: str, verification: str, files: list[str]) -> None:
        """Record acceptance only from a matching independent verifier receipt."""
        tree = self.load()
        node = next(node for node in tree.nodes if node.id == key)
        report = json.loads(self.path_for(verification).read_text(encoding="utf-8"))
        identity = report["identity"]
        if not report["verified"] or identity["leaf_id"] != key or identity["tree_revision"] != tree.revision or identity["spec_revision"] != node.revision:
            raise ValueError("The verifier did not accept the current specification.")
        from workflows.minecraft_tasks.verification import evaluate
        if (not all(evaluate(node.criteria, report.get("reads", []), certificates=self.state["acceptances"]).values())
            or report.get("review", {}).get("status") != "accepted"
            or set(report["review"].get("criterion_ids", [])) != {row.id for row in node.criteria}):
            raise ValueError("The verifier has no sufficient independent world evidence.")
        dependencies = self.requirements(key)
        if not all(self.valid(dependency) for dependency in dependencies):
            raise ValueError("A prerequisite has no valid acceptance.")
        paths = set(files) | {verification}
        self.state["acceptances"][key] = {"spec_digest": node_digest(node), "verification": verification,
            "identity": identity, "dependencies": {dependency: digest(self.state["acceptances"][dependency]) for dependency in dependencies},
            "files": {relative: hashlib.sha256(self.path_for(relative).read_bytes()).hexdigest() for relative in paths}}
        self.save()
