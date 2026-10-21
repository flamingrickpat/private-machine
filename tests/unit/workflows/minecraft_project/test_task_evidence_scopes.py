"""Check scoped workflow boundaries against retained real gameplay captures."""
from __future__ import annotations

from pm.records.copying import pickle_copy

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tests.unit.workflows.minecraft_project import test_gameplay as captured
from workflows.minecraft_project.evidence_selection import dependency_scope, protect_accepted_specifications, retention_operation, retention_tree, read_source, review_nodes
from workflows.minecraft_project.schemas import EvidenceRead, Renewal, RetentionChoice, RetentionProposal
from workflows.minecraft_project.tree_store import node_digest


class EvidenceScopeTests(unittest.TestCase):
    def test_captured_publication_requirements_reach_shared_plan_context(self):
        import json
        from workflows.minecraft_project.planning.constraints import caller_plan_constraints
        from workflows.minecraft_project.tree_store import TreeStore, digest
        capture = json.loads((Path(__file__).resolve().parents[4] / 'test_data/captures/alpha/a13-caller-plan-context.json').read_text(encoding='utf-8'))
        original_input = json.loads(capture['review_input']['messages'][1]['content'])
        self.assertNotIn('plan_constraints', original_input)
        self.assertEqual(capture['native_review']['status'], 'revise')
        with TemporaryDirectory() as folder:
            store = TreeStore(Path(folder))
            for name, contents in capture['files'].items():
                path = store.path_for(name)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(contents.encode('utf-8'))
            store.state['planning_constraints'] = capture['contract_path']
            before = pickle_copy(store.state)
            result = caller_plan_constraints(store)
            original = json.loads(capture['files'][capture['contract_path']])
            for key, value in original['result'].items():
                self.assertEqual(result[key], value)
            self.assertEqual(result['original_receipt'], capture['contract_path'])
            self.assertEqual(result['original_digest'], digest(original))
            self.assertEqual(store.state, before)
            self.assertIn('checked before project publication', result['requirements']['durable_home_and_route'])

    def test_captured_hosted_retention_preserves_exact_recursive_proof(self):
        import base64
        import json
        import zlib
        from workflows.minecraft_project.tree_store import TreeStore
        capture = json.loads((Path(__file__).resolve().parents[4] / "test_data/captures/alpha/a13-scoped-retention.json").read_text())
        files = json.loads(zlib.decompress(base64.b64decode(capture["files_zlib_base64"])))
        with TemporaryDirectory() as folder:
            store = TreeStore(Path(folder))
            for name, contents in files.items():
                path = store.path_for(name)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(contents.encode("utf-8"))
            for revision, tree in capture["specifications"].items():
                store.write("planning/specs/" + revision + ".json", tree)
            store.state = pickle_copy(capture["state"])
            store.save()
            proposal = RetentionProposal.model_validate(capture["native_proposal"])
            proposed = retention_tree(store, proposal)
            store.revise(proposed)
            for key in proposal.restore_nodes:
                self.assertTrue(store.valid(key))
            for renewal in proposal.renewals:
                self.assertFalse(store.valid(renewal.node.id))
            for key, original in capture["state"]["acceptances"].items():
                if key not in proposal.restore_nodes:
                    self.assertEqual(store.state["acceptances"][key], original)
            # M3-48 actually tried to rewrite already certified mining.
            # Replay that native patch through the ordinary repair boundary.
            from workflows.minecraft_project.schemas import RevisionProposal
            from workflows.minecraft_project.planning.proposals import apply_node_updates
            conflict = json.loads((Path(__file__).resolve().parents[4] / "test_data/captures/alpha/a13-historical-review-conflict.json").read_text(encoding="utf-8"))
            patch = RevisionProposal.model_validate(conflict["native_patch"])
            self.assertTrue(store.valid(conflict["native_review"]["target_id"]))
            self.assertNotIn(conflict["native_review"]["target_id"], {node.id for node in review_nodes(store)})
            before = store.path.read_bytes()
            with self.assertRaises(ValueError):
                protect_accepted_specifications(store, apply_node_updates(store.load(), patch.updates))
            self.assertEqual(store.path.read_bytes(), before)

    def retained_case(self, root):
        """Load original files and certificates from the real recovery capture."""
        store, capture = captured.GameplayTests().retained_certificate_case(root)
        store.archive_acceptances(capture["before"], "captured-original-receipt.json")
        return store, capture

    def proposal(self, store):
        """Build a boundary input; this test does not simulate model inference."""
        candidates = store.acceptance_candidates()
        nodes = {node.id: node for node in store.load().nodes}
        renewals = [Renewal(source_node_id=row["specification"]["id"],
                           node=nodes[row["specification"]["id"]].model_copy(update={"id": "new_current_requirement"}))
                    for row in candidates if store.valid(row["specification"]["id"])
                    and node_digest(nodes[row["specification"]["id"]]) != row["certificate"]["spec_digest"]]
        return RetentionProposal(status="planned", summary="Boundary input from the original recovery capture.",
                                 restore_nodes=[row["specification"]["id"] for row in candidates], renewals=renewals)

    def test_historical_restoration_keeps_current_recovery_unaccepted(self):
        with TemporaryDirectory() as folder:
            store, capture = self.retained_case(Path(folder))
            before = store.path.read_bytes()
            proposal = self.proposal(store)
            proposed = retention_tree(store, proposal)
            self.assertEqual(store.path.read_bytes(), before)
            store.revise(proposed)
            for key, record in capture["before"]["acceptances"].items():
                self.assertEqual(store.state["acceptances"][key], record)
                self.assertTrue(store.valid(key))
            self.assertFalse(store.valid(proposal.renewals[0].node.id))
            self.assertTrue(any(condition.node_id == proposal.renewals[0].node.id
                                for node in store.load().nodes for condition in node.criteria))

    def test_missing_dependency_or_current_requirement_cannot_change_state(self):
        with TemporaryDirectory() as folder:
            store, _ = self.retained_case(Path(folder))
            proposal = self.proposal(store)
            before = store.path.read_bytes()
            original = pickle_copy(store.state)
            for altered in [proposal.model_copy(update={"renewals": []}),
                            proposal.model_copy(update={"restore_nodes": [key for key in proposal.restore_nodes
                                if key != proposal.renewals[0].source_node_id]})]:
                with self.assertRaises(ValueError):
                    retention_tree(store, altered)
                self.assertEqual(store.path.read_bytes(), before)
                self.assertEqual(store.state, original)

    def test_dependency_selection_is_independent_of_scenario_node_names(self):
        with TemporaryDirectory() as folder:
            store, _ = self.retained_case(Path(folder))
            tree = store.load()
            target = next(row["specification"]["id"] for row in store.acceptance_candidates()
                          if len(row["certificate"]["dependencies"]) > 1 and not store.valid(row["specification"]["id"]))
            original = dependency_scope(tree, [target])
            names = {node.id: "arbitrary_" + str(index) for index, node in enumerate(tree.nodes)}
            changed = pickle_copy(tree)
            changed.goal_id = names[tree.goal_id]
            for node in changed.nodes:
                node.id = names[node.id]
                node.parent = names[node.parent] if node.parent else None
                node.children = [names[key] for key in node.children]
                node.depends_on = [names[key] for key in node.depends_on]
                for condition in node.criteria + node.assumptions:
                    if condition.kind == "accepted":
                        condition.node_id = names[condition.node_id]
            self.assertEqual(dependency_scope(changed, [names[target]]), {names[key] for key in original})
            self.assertNotIn(tree.goal_id, original)

    def test_retrieval_rejects_changed_original_evidence(self):
        import hashlib
        with TemporaryDirectory() as folder:
            store, _ = self.retained_case(Path(folder))
            record = next(iter(store.state["acceptances"].values()))
            path = store.path_for(record["verification"])
            sources = {"original": {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}
            result = read_source(sources, EvidenceRead(source="original", query="identity", line_count=5))
            self.assertTrue(result["lines"])
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaises(ValueError):
                read_source(sources, EvidenceRead(source="original"))

    def test_completed_historical_work_stays_outside_current_work_review(self):
        with TemporaryDirectory() as folder:
            store, _ = self.retained_case(Path(folder))
            proposal = self.proposal(store)
            store.revise(retention_tree(store, proposal))
            selected = {node.id for node in review_nodes(store)}
            self.assertIn(store.load().goal_id, selected)
            self.assertIn(proposal.renewals[0].node.id, selected)
            self.assertFalse(selected.intersection(key for key in store.state["acceptances"] if key != store.load().goal_id))
            # A changed original source removes validity and requires review.
            key = next(key for key in store.state["acceptances"] if key != store.load().goal_id)
            record = store.state["acceptances"][key]
            path = store.path_for(record["verification"])
            path.write_bytes(path.read_bytes() + b"\n")
            self.assertIn(key, {node.id for node in review_nodes(store)})

    def test_ordinary_repair_cannot_rewrite_valid_original_attainment(self):
        with TemporaryDirectory() as folder:
            store, _ = self.retained_case(Path(folder))
            store.revise(retention_tree(store, self.proposal(store)))
            original = store.path.read_bytes()
            tree = store.load()
            protect_accepted_specifications(store, tree)
            key = next(key for key in store.state["acceptances"] if key != tree.goal_id)
            changed = pickle_copy(tree)
            node = next(node for node in changed.nodes if node.id == key)
            node.instruction += " Repeat the original material work using current inventory."
            with self.assertRaises(ValueError):
                protect_accepted_specifications(store, changed)
            self.assertEqual(store.path.read_bytes(), original)

    def test_renewal_fields_come_from_actual_validity_and_exact_current_contract(self):
        with TemporaryDirectory() as folder:
            store, _ = self.retained_case(Path(folder))
            selected = self.proposal(store)
            choice = RetentionChoice(status="planned", summary=selected.summary, restore_nodes=selected.restore_nodes)
            before = store.path.read_bytes()
            operation = retention_operation(store, choice)
            self.assertEqual(len(operation.renewals), 1)
            renewal = operation.renewals[0]
            original = next(node for node in store.load().nodes if node.id == renewal.source_node_id)
            self.assertEqual(node_digest(renewal.node.model_copy(update={"id": original.id})), node_digest(original))
            self.assertEqual(store.path.read_bytes(), before)
            store.revise(retention_tree(store, operation))
            for key in choice.restore_nodes:
                self.assertTrue(store.valid(key))
            self.assertFalse(store.valid(renewal.node.id))
