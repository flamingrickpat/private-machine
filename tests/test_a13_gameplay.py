"""Check recovery boundaries with original real-server receipts."""
from __future__ import annotations

import copy
import base64
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import threading
import zlib

from runtime.minecraft_outcomes import GameplayEvidence, classify
from workflows.minecraft_tasks.schemas import Node, TaskTree
from workflows.minecraft_tasks.tree import TreeStore

CAPTURE = Path(__file__).resolve().parents[1] / "test_data/captures/alpha/a13-gameplay-results.json"


class GameplayTests(unittest.TestCase):
    def retained_certificate_case(self, root):
        """Rebuild original captured files without changing their byte hashes."""
        capture = json.loads((CAPTURE.parent / "a13-recovery-certificate-preservation.json").read_text(encoding="utf-8"))
        files = json.loads(zlib.decompress(base64.b64decode(capture["files_zlib_base64"])))
        for relative, contents in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents.encode("utf-8"))
        store = TreeStore(root)
        for revision, specification in capture["specifications"].items():
            store.write(f"planning/specs/{int(revision):04d}.json", specification)
        store.state = copy.deepcopy(capture["after"])
        store.save()
        return store, capture

    def test_real_recovery_revision_restores_exact_consumed_material_certificates(self):
        with TemporaryDirectory() as folder:
            store, capture = self.retained_certificate_case(Path(folder))
            self.assertEqual(set(capture["before"]["acceptances"]) - set(store.state["acceptances"]),
                             {"iron_mine", "iron_smelt", "iron_pickaxe"})
            store.archive_acceptances(capture["before"], "retained-before-receipt.json")
            candidates = {row["certificate"]["identity"]["leaf_id"]: row for row in store.acceptance_candidates()}
            self.assertEqual(set(candidates), {"recovery_exit", "iron_mine", "iron_smelt", "iron_pickaxe"})
            store.restore_acceptances()
            self.assertNotIn("iron_mine", store.state["acceptances"])
            proposed = store.load()
            original = TaskTree.model_validate(capture["specifications"][str(capture["before"]["revision"])])
            recovery = next(node for node in original.nodes if node.id == "recovery_exit")
            proposed.nodes = [recovery if node.id == recovery.id else node for node in proposed.nodes]
            store.revise(proposed)
            self.assertEqual(store.state["acceptances"], capture["before"]["acceptances"])
            self.assertTrue(all(store.valid(key) for key in store.state["acceptances"]))
            self.assertEqual(store.acceptance_candidates(), [])

    def test_real_archived_certificate_rejects_changed_original_file(self):
        with TemporaryDirectory() as folder:
            store, capture = self.retained_certificate_case(Path(folder))
            store.archive_acceptances(capture["before"], "retained-before-receipt.json")
            record = capture["before"]["acceptances"]["iron_mine"]
            path = store.path_for(next(path for path in record["files"] if path.endswith(".jsonl")))
            path.write_bytes(path.read_bytes() + b"\n")
            proposed = store.load()
            original = TaskTree.model_validate(capture["specifications"][str(capture["before"]["revision"])])
            recovery = next(node for node in original.nodes if node.id == "recovery_exit")
            proposed.nodes = [recovery if node.id == recovery.id else node for node in proposed.nodes]
            store.revise(proposed)
            self.assertTrue(store.valid("recovery_exit"))
            self.assertFalse(any(store.valid(key) for key in ("iron_mine", "iron_smelt", "iron_pickaxe")))

    def test_real_archive_import_rejects_a_changed_dependency_receipt(self):
        with TemporaryDirectory() as folder:
            store, capture = self.retained_certificate_case(Path(folder))
            record = capture["before"]["acceptances"]["recovery_exit"]
            path = store.path_for(record["verification"])
            path.write_bytes(path.read_bytes() + b"\n")
            store.archive_acceptances(capture["before"], "retained-before-receipt.json")
            keys = {row["certificate"]["identity"]["leaf_id"] for row in store.acceptance_candidates()}
            self.assertFalse(keys & {"recovery_exit", "iron_mine", "iron_smelt", "iron_pickaxe"})

    def test_real_certificate_revocation_cannot_be_undone_by_specification_repair(self):
        with TemporaryDirectory() as folder:
            store, capture = self.retained_certificate_case(Path(folder))
            store.state = copy.deepcopy(capture["before"])
            store.save()
            store.archive_acceptances(store.state, "original-state.json")
            store.revoke("recovery_exit")
            store.invalidate()
            store.restore_acceptances()
            self.assertFalse(any(store.valid(key) for key in ("recovery_exit", "iron_mine", "iron_smelt", "iron_pickaxe")))

    def test_native_restore_operation_requires_exact_historical_dependencies(self):
        from workflows.minecraft_tasks.schemas import RevisionProposal
        with TemporaryDirectory() as folder:
            store, capture = self.retained_certificate_case(Path(folder))
            store.archive_acceptances(capture["before"], "retained-before-receipt.json")
            before = copy.deepcopy(store.state)
            with self.assertRaisesRegex(ValueError, "recovery_exit"):
                store.restore_specifications(store.load(), ["iron_mine"])
            self.assertEqual(store.state, before)
            request = RevisionProposal(status="planned", summary="Restore original accepted work.",
                                       restore_nodes=["recovery_exit", "iron_mine", "iron_smelt", "iron_pickaxe"])
            proposed = store.restore_specifications(store.load(), request.restore_nodes)
            store.revise(proposed)
            self.assertEqual(store.state["acceptances"], capture["before"]["acceptances"])

    def test_native_restoration_rejects_unknown_and_revoked_candidates(self):
        from workflows.minecraft_tasks.schemas import RevisionProposal
        with TemporaryDirectory() as folder:
            store, capture = self.retained_certificate_case(Path(folder))
            store.archive_acceptances(capture["before"], "retained-before-receipt.json")
            with self.assertRaisesRegex(ValueError, "No retained restoration"):
                store.restore_specifications(store.load(), ["unknown"])
            with self.assertRaises(ValueError):
                RevisionProposal(status="planned", summary="Duplicate restoration.", restore_nodes=["iron_mine", "iron_mine"])
            with self.assertRaises(ValueError):
                RevisionProposal(status="survey", summary="A survey cannot restore proof.", restore_nodes=["iron_mine"])

    def test_native_restoration_checks_recursive_dependencies_before_commit(self):
        with TemporaryDirectory() as folder:
            store, capture = self.retained_certificate_case(Path(folder))
            store.archive_acceptances(capture["before"], "retained-before-receipt.json")
            before = copy.deepcopy(store.state)
            with self.assertRaisesRegex(ValueError, "recovery_exit"):
                store.restore_specifications(store.load(), ["iron_mine", "iron_smelt", "iron_pickaxe"])
            with self.assertRaisesRegex(ValueError, "iron_smelt"):
                store.restore_specifications(store.load(), ["iron_pickaxe", "recovery_exit"])
            self.assertEqual(store.state, before)
            proposed = store.restore_specifications(
                store.load(), ["iron_pickaxe", "iron_smelt", "iron_mine", "recovery_exit"])
            store.revise(proposed)
            self.assertEqual(store.state["acceptances"], capture["before"]["acceptances"])
            self.assertTrue(all(store.valid(key) for key in store.state["acceptances"]))

    def test_restored_real_actor_proof_uses_original_certificate_revision(self):
        from workflows.minecraft_tasks.verification import accepted_actor_matches
        captured = json.loads((CAPTURE.parent / "a13-restored-publication-identity.json").read_text(encoding="utf-8"))
        with TemporaryDirectory() as folder:
            store, capture = self.retained_certificate_case(Path(folder))
            store.state["attempts"] = copy.deepcopy(captured["attempts"])
            store.archive_acceptances(capture["before"], "retained-before-receipt.json")
            proposed = store.restore_specifications(
                store.load(), ["recovery_exit", "iron_mine", "iron_smelt", "iron_pickaxe"])
            tree = store.revise(proposed)
            nodes = {node.id: node for node in tree.nodes}
            for key in ("recovery_exit", "iron_mine", "iron_smelt", "iron_pickaxe"):
                record = store.state["acceptances"][key]
                report = json.loads(store.path_for(record["verification"]).read_text(encoding="utf-8"))
                self.assertTrue(accepted_actor_matches(store, nodes[key], report), key)
            self.assertNotEqual(nodes["recovery_exit"].revision,
                                store.state["acceptances"]["recovery_exit"]["identity"]["spec_revision"])
            record = store.state["acceptances"]["iron_pickaxe"]
            report = json.loads(store.path_for(record["verification"]).read_text(encoding="utf-8"))
            actor = next(row for row in store.state["attempts"] if row["id"] == record["identity"]["actor_attempt"])
            actor["spec_revision"] += 1
            self.assertFalse(accepted_actor_matches(store, nodes["iron_pickaxe"], report))
            actor["spec_revision"] -= 1
            report["identity"]["tree_revision"] += 1
            self.assertFalse(accepted_actor_matches(store, nodes["iron_pickaxe"], report))
            report["identity"]["tree_revision"] -= 1
            path = store.path_for(actor["journal"])
            path.write_bytes(path.read_bytes() + b"\n")
            self.assertFalse(accepted_actor_matches(store, nodes["iron_pickaxe"], report))

    def test_actual_furnace_missing_reply_preserves_real_recovered_supplies(self):
        capture = json.loads((CAPTURE.parent / "a13-window-inventory.json").read_text(encoding="utf-8"))
        row = next(row for row in capture["rows"] if row["seq"] == 23)
        self.assertEqual(row["arguments"]["input_count"], 2)
        self.assertEqual(row["arguments"]["fuel_count"], 2)
        self.assertEqual(row["result"]["input_consumed"], 0)
        self.assertEqual(row["result"]["recovered_output"]["count"], 1)
        delta = {change["item"]: change["delta"] for change in row["result"]["inventory_delta"]["changes"]}
        self.assertEqual(delta, {"raw_iron": 2, "iron_ingot": 1})
        outcome = classify(row)
        self.assertEqual(outcome["kind"], "partial_progress")
        self.assertTrue(outcome["made_progress"])
        self.assertFalse(outcome["completion_verified"])

    def test_actual_furnace_window_confirms_retained_input_without_inventory_gain(self):
        capture = json.loads((CAPTURE.parent / "a13-furnace-storage-target.json").read_text(encoding="utf-8"))
        rows = {(row["seq"], row["event"]): row for row in capture["rows"]}
        window = rows[(21, "result")]["result"]["window"]
        self.assertEqual(window["kind"], "furnace")
        self.assertEqual(window["slots"], [{"slot": 0, "item": "raw_iron", "count": 2},
            {"slot": 1, "item": None, "count": 0}, {"slot": 2, "item": "iron_ingot", "count": 1}])
        self.assertFalse(rows[(22, "result")]["ok"])
        self.assertEqual(rows[(24, "uncertain")]["tool"], "minecraft_chest_withdraw")
        history = GameplayEvidence()
        history.consume(rows[(24, "uncertain")], capture["sources"]["journal"]["path"])
        self.assertIsNotNone(history.service_wait)

    def test_uncertain_furnace_context_survives_a_later_actor_with_only_recorded_calls(self):
        from runtime.minecraft_actor import uncertain_world_effects
        capture = json.loads((CAPTURE.parent / "a13-uncertain-furnace-context.json").read_text(encoding="utf-8"))
        history = [{"source": capture["sources"]["previous_journal"]["path"], "calls": capture["previous_rows"]},
                   {"source": capture["sources"]["journal"]["path"], "calls": capture["current_rows"]}]
        original = capture["reconciliation"]["retained_actions"]
        self.assertEqual(len(original), 25)
        self.assertEqual(original[-1]["outcome"], "uncertain")
        self.assertTrue(all(row["outcome"] == "recorded" for row in original[:-1]))
        self.assertEqual(uncertain_world_effects(history[1:]), [])
        pending = uncertain_world_effects(history)
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["seq"], 45)
        self.assertEqual(pending[0]["arguments"], {"input": "raw_iron", "input_count": 3, "fuel": "birch_log"})
        self.assertEqual(pending[0]["candidate_furnaces"][-1]["position"], {"x": 30, "y": 102, "z": 5})
        self.assertEqual(uncertain_world_effects(history + history), pending)
        self.assertLess(len(json.dumps(pending)), 2000)
        # A projected later receipt must explicitly account for the old slots.
        later = {"seq": 99, "event": "result", "time": capture["current_rows"][-1]["time"] + 1,
                 "tool": "minecraft_smelt_item", "ok": False, "result": {"input_item": "raw_iron",
                 "recovered_output": {"name": "iron_ingot", "count": 1}, "furnace_before": {
                 "position": {"x": 30, "y": 102, "z": 5}, "slots": [
                     {"slot": 0, "item": "raw_iron", "count": 2}, {"slot": 2, "item": "iron_ingot", "count": 1}]}}}
        self.assertEqual(uncertain_world_effects(history + [{"source": "projected-next", "calls": [later]}]), [])
        later["result"]["furnace_before"]["slots"][0]["count"] = 1
        self.assertEqual(uncertain_world_effects(history + [{"source": "projected-next", "calls": [later]}]), pending)

    def test_partial_reconciled_smelt_returns_raw_without_counting_it_as_mined(self):
        from dev.work_items.alpha_a13_minecraft_gate import iron_chain
        capture = json.loads((CAPTURE.parent / "a13-smelt-timeout.json").read_text(encoding="utf-8"))
        saved = capture["saved_furnaces"]["furnaces"][0]
        self.assertEqual(saved["BurnTime"], 0)
        slots = [{"slot": item["Slot"], "item": item["id"].split(":")[1], "count": item["Count"]} for item in saved["Items"]]
        position = {key: saved[key] for key in ("x", "y", "z")}
        furnace = {"position": position, "slots": slots}
        # Subsequent calls project another one-log shortage, a completed final
        # smelt, and crafting. They test provenance, not actual M3 completion.
        partial = {"event": "result", "tool": "minecraft_smelt_item", "ok": False, "result": {
            "input_item": "raw_iron", "input_consumed": 1, "furnace_before": furnace,
            "recovered_output": {"name": "iron_ingot", "count": 1}, "output": {"name": "iron_ingot", "count": 1},
            "inventory_delta": {"changes": [{"item": "raw_iron", "delta": 1}, {"item": "iron_ingot", "delta": 2}]}}}
        final = {"event": "result", "tool": "minecraft_smelt_item", "ok": True, "result": {
            "input_item": "raw_iron", "input_consumed": 1, "output": {"name": "iron_ingot", "count": 1},
            "inventory_delta": {"changes": [{"item": "raw_iron", "delta": -1}, {"item": "iron_ingot", "delta": 1}]}}}
        craft = {"event": "result", "tool": "minecraft_craft_item", "ok": True, "result": {
            "resolved_item": "iron_pickaxe", "inventory_delta": {"changes": [
                {"item": "iron_pickaxe", "delta": 1}, {"item": "iron_ingot", "delta": -3}]}}}
        rows = capture["mining_rows"] + capture["rows"] + [partial, final, craft]
        rows = [row for row in rows if row["event"] in {"result", "uncertain"}]
        with TemporaryDirectory() as directory:
            store = TreeStore(Path(directory))
            store.state["attempts"] = [{"path": "attempts/actor-0001"}]
            store.save()
            path = store.path_for("attempts/actor-0001/scratch/actor-journal.jsonl")
            path.parent.mkdir(parents=True)
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            result = iron_chain(store.root, {"inventory": []})
            self.assertTrue(result["verified"], result)
            self.assertEqual(result["mined_raw_iron"], 3)
            self.assertEqual(result["smelted_ingots"], 3)
            # The same output without its original uncertain dispatch is invalid.
            path.write_text("\n".join(json.dumps(row) for row in rows if row["event"] != "uncertain"), encoding="utf-8")
            self.assertFalse(iron_chain(store.root, {"inventory": []})["verified"])

    def test_uncertain_native_smelt_requires_matching_retained_furnace_contents(self):
        from dev.work_items.alpha_a13_minecraft_gate import recovered_smelt_matches
        capture = json.loads((CAPTURE.parent / "a13-smelt-timeout.json").read_text(encoding="utf-8"))
        pending = next(row for row in capture["rows"] if row["event"] == "uncertain")
        placements = [cell["position"] for row in capture["rows"] if row["event"] == "result"
                      for cell in row["result"].get("placed", [])]
        # These slots project the captured one-log shortage. Fresh live slots
        # must still establish the actual recovery before acceptance can pass.
        furnace = {"position": placements[-1], "slots": [{"slot": 0, "item": "raw_iron", "count": 2},
                   {"slot": 2, "item": "iron_ingot", "count": 1}]}
        self.assertTrue(recovered_smelt_matches(pending, placements, furnace, 1))
        self.assertFalse(recovered_smelt_matches(pending, [], furnace, 1))
        self.assertFalse(recovered_smelt_matches(pending, placements, furnace, 2))
        furnace["slots"][0]["count"] = 1
        self.assertFalse(recovered_smelt_matches(pending, placements, furnace, 1))

    def test_captured_successful_equipment_noops_pause_after_two_calls(self):
        capture = json.loads((CAPTURE.parent / "a13-equipment-no-progress.json").read_text(encoding="utf-8"))
        rows = {row["seq"]: row for row in capture["rows"]}
        history = GameplayEvidence()
        source = capture["sources"]["journal"]["path"]
        for seq in (348, 349):
            self.assertTrue(rows[seq]["ok"])
            self.assertEqual(classify(rows[seq])["reason"], "equipment_unchanged")
            self.assertFalse(classify(rows[seq])["made_progress"])
            history.consume(rows[seq], source)
        rejected = history.guard(rows[350]["tool"], rows[350]["arguments"])
        self.assertEqual(rejected["reason"], "method_paused")
        self.assertIn("cannot gather or craft", rejected["equipment_guidance"])
        self.assertIsNone(history.guard("minecraft_craft_item", {"item": "crafting_table"}))
        improved = copy.deepcopy(rows[348])
        improved["result"]["equipped"] = {"name": "stone_pickaxe", "durability": 131, "max_durability": 131}
        self.assertTrue(classify(improved)["made_progress"])
        self.assertEqual(classify(improved)["quantities"]["equipment_changed"], 1)

    def test_captured_repeated_equipment_context_keeps_every_original_sequence(self):
        from workflows.minecraft_tasks.steps import scoped_gameplay, scoped_decision_context
        capture = json.loads((CAPTURE.parent / "a13-equipment-no-progress.json").read_text(encoding="utf-8"))
        history = GameplayEvidence()
        source = capture["sources"]["journal"]["path"]
        for row in capture["rows"]:
            if row["tool"] == "minecraft_equip_best_tool":
                history.consume(row, source)
        raw = history.snapshot()
        scoped = scoped_gameplay(raw)
        decoded = self.unpack_planning_history(scoped)
        self.assertEqual(decoded["recent_outcomes"], raw["recent_outcomes"])
        self.assertTrue(any(row.get("changes", {}).get("seq") for row in scoped["recent_outcomes"]))
        selected = scoped_decision_context({"gameplay_history": scoped})["gameplay_history"]
        self.assertEqual(self.unpack_planning_history(selected)["recent_outcomes"], raw["recent_outcomes"])
        self.assertLess(len(json.dumps(scoped["recent_outcomes"])), len(json.dumps(raw["recent_outcomes"])) / 2)
        self.assertEqual(classify(capture["rows"][1])["reason"], "unsafe_support")

    def test_captured_latest_overflow_references_old_survey_and_preserves_search_coverage(self):
        from workflows.minecraft_tasks.steps import scoped_decision_context, decision_schema
        from workflows.minecraft_tasks.schemas import RevisionProposal
        capture = json.loads((CAPTURE.parent / "a13-historical-survey-budget.json").read_text(encoding="utf-8"))
        payload = capture["unscoped_context"]
        before = copy.deepcopy(payload)
        scoped = scoped_decision_context(payload)
        old = payload["requested_world_survey"]
        current = scoped["requested_world_survey"]
        for key in ("identity", "queries", "recipe_queries", "original_receipt", "original_digest"):
            self.assertEqual(current[key], old[key])
        self.assertEqual(current["archived_read_count"], len(old["reads"]))
        self.assertNotIn("reads", current)
        self.assertNotIn("recipes", current)
        self.assertEqual(self.unpack_planning_history(scoped["gameplay_history"])["searches"],
                         self.unpack_planning_history(payload["gameplay_history"])["searches"])
        guide = (Path(__file__).resolve().parents[1] / "workflows/minecraft_tasks/resources/planner.md").read_text(encoding="utf-8")
        system = guide + "\nJSON schema:\n" + json.dumps(decision_schema(RevisionProposal), separators=(",", ":"))
        self.assertEqual(capture["input"]["input_characters"], 124188)
        self.assertLess(len(system) + len(json.dumps(scoped, ensure_ascii=False, separators=(",", ":"))), capture["input"]["input_characters"])
        self.assertEqual(payload, before)
        fresh = copy.deepcopy(payload)
        fresh["requested_world_survey"]["identity"]["tree_revision"] = fresh["identity"]["tree_revision"]
        self.assertEqual(scoped_decision_context(fresh)["requested_world_survey"], fresh["requested_world_survey"])

    def test_captured_house_progress_context_retains_every_search_and_current_fact(self):
        from workflows.minecraft_tasks.steps import scoped_decision_context, decision_schema
        from workflows.minecraft_tasks.schemas import RevisionProposal
        capture = json.loads((CAPTURE.parent / "a13-planner-growth.json").read_text(encoding="utf-8"))
        payload = capture["unscoped_context"]
        before = copy.deepcopy(payload)
        selected = scoped_decision_context(payload)
        original = self.unpack_planning_history(capture["rejected_context"]["gameplay_history"])
        decoded = self.unpack_planning_history(selected["gameplay_history"])
        for key in ("methods", "method_states", "searches", "recent_outcomes", "recovery", "life", "navigation",
                    "failed_recoveries", "service_wait", "archived_methods", "archived_diagnoses", "diagnoses"):
            self.assertEqual(decoded[key], original[key], key)
        self.assertEqual(len(decoded["searches"]), 73)
        self.assertEqual(len(selected["accepted_nodes"]), 11)
        for key in ("previous_tree", "accepted_nodes", "observation", "latest_attempt", "server_rules", "plan_constraints"):
            self.assertEqual(selected[key], payload[key], key)
        old = payload["previous_site_survey"]
        current = selected["previous_site_survey"]
        for key in ("identity", "original_receipt", "original_digest"):
            self.assertEqual(current[key], old[key])
        for prior, row in zip(old["recipes"], current["recipes"]):
            self.assertLess(prior["time"], payload["latest_attempt"]["started"])
            self.assertNotIn("result", row)
            for key in ("tool", "arguments", "time", "ok", "original_digest", "observation_available"):
                self.assertEqual(row[key], prior[key])
        guide = (Path(__file__).resolve().parents[1] / "workflows/minecraft_tasks/resources/planner.md").read_text(encoding="utf-8")
        system = guide + "\nJSON schema:\n" + json.dumps(decision_schema(RevisionProposal), separators=(",", ":"))
        self.assertEqual(capture["original_input_characters"], 130147)
        self.assertLess(len(system) + len(json.dumps(selected, ensure_ascii=False, separators=(",", ":"))), capture["original_input_characters"])
        self.assertEqual(payload, before)

    def test_current_recipes_and_unusual_search_shapes_stay_explicit(self):
        from workflows.minecraft_tasks.steps import scoped_decision_context
        capture = json.loads((CAPTURE.parent / "a13-planner-growth.json").read_text(encoding="utf-8"))
        payload = copy.deepcopy(capture["unscoped_context"])
        for row in payload["previous_site_survey"]["recipes"]:
            row["time"] = payload["latest_attempt"]["started"]
        payload["gameplay_history"]["searches"][0]["coverage"] = {"center": None, "bounds": None, "legacy": True}
        expected = self.unpack_planning_history(payload["gameplay_history"])["searches"]
        selected = scoped_decision_context(payload)
        self.assertEqual(selected["previous_site_survey"]["recipes"], payload["previous_site_survey"]["recipes"])
        self.assertNotIn("search_coverage_vector_fields", selected["gameplay_history"])
        self.assertEqual(self.unpack_planning_history(selected["gameplay_history"])["searches"], expected)

    def test_captured_wrong_tool_cannot_be_bypassed_with_new_coordinates(self):
        capture = json.loads((CAPTURE.parent / "a13-harvesting-body-space.json").read_text(encoding="utf-8"))
        history = GameplayEvidence()
        source = capture["sources"]["journal"]["path"]
        rows = {row["seq"]: row for row in capture["rows"]}
        for seq in (309, 311, 315, 317):
            history.consume(rows[seq], source)
        action = rows[321]
        rejected = history.guard(action["tool"], action["arguments"])
        self.assertEqual(rejected["reason"], "harvesting_prerequisite")
        self.assertIn("accepted historical", rejected["message"])
        self.assertIsNone(rows[311]["result"]["can_harvest_with_held"])
        self.assertFalse(rows[309]["result"]["nearest"]["harvestable_with_held"])
        self.assertIsNone(history.guard("minecraft_craft_item", {"item": "crafting_table"}))
        # Project the repaired inspector metadata at the next target. A
        # diagnosis cannot grant a dig when the actual tool is unsuitable.
        inspect = copy.deepcopy(rows[311])
        inspect.update(seq=400, time=action["time"] + 1)
        inspect["arguments"]["position"] = action["arguments"]["position"]
        inspect["result"].update(position=action["arguments"]["position"], can_harvest_with_held=False)
        history.consume(inspect, source)
        history.permit = {"next_tool": action["tool"], "next_arguments": action["arguments"]}
        self.assertEqual(history.guard(action["tool"], action["arguments"])["reason"], "harvesting_prerequisite")
        inspect.update(seq=401, time=inspect["time"] + 1)
        inspect["result"]["can_harvest_with_held"] = True
        history.consume(inspect, source)
        self.assertIsNone(history.guard(action["tool"], action["arguments"]))
        self.assertEqual(history.snapshot()["harvesting"]["ineffective"], 2)

    def test_captured_suffocation_keeps_death_and_inventory_loss(self):
        capture = json.loads((CAPTURE.parent / "a13-harvesting-body-space.json").read_text(encoding="utf-8"))
        rows = {row["seq"]: row for row in capture["rows"]}
        self.assertIn("suffocated in a wall", capture["death_message"])
        self.assertEqual(rows[388]["result"]["death_count"], 1)
        self.assertEqual(rows[388]["result"]["inventory"], [])
        history = GameplayEvidence()
        source = capture["sources"]["journal"]["path"]
        history.consume(rows[362], source)
        history.consume(rows[388], source)
        self.assertEqual(history.guard("minecraft_build", rows[377]["arguments"])["reason"], "death_requires_replan")

    def test_captured_staircase_death_records_missing_items_and_forces_replan(self):
        capture = json.loads((CAPTURE.parent / "a13-staircase-death.json").read_text(encoding="utf-8"))
        history = GameplayEvidence()
        source = capture["sources"]["journal"]["path"]
        for row in capture["rows"]:
            history.consume(row, source)
        failed = history.current_failed_recovery()
        self.assertEqual(failed["reason"], "player_died")
        self.assertEqual(sum(row["missing"] for row in failed["inventory_missing_after_death"]), 252)
        self.assertEqual(next(row["missing"] for row in failed["inventory_missing_after_death"] if row["item"] == "raw_iron"), 2)
        self.assertIn("do not establish consumption", failed["inventory_scope"])
        self.assertEqual(history.guard("minecraft_craft_item", {"item": "iron_pickaxe"})["reason"], "death_requires_replan")
        self.assertEqual(capture["rows"][1]["result"]["end_position"]["y"], 0)
        self.assertEqual(history.observation["player"]["health"], 20)
        self.assertFalse(history.recovery_proved())

    def test_captured_death_counters_on_projected_action_receipt_stop_before_next_observation(self):
        capture = json.loads((CAPTURE.parent / "a13-staircase-death.json").read_text(encoding="utf-8"))
        before, action, after = copy.deepcopy(capture["rows"])
        # Add the new lifecycle fields from the original two body observations.
        # This checks the receipt boundary, not a retroactive repaired action.
        action["result"].update(body_session=before["result"]["body_session"],
                                death_count_before=before["result"]["death_count"],
                                death_count_after=after["result"]["death_count"])
        history = GameplayEvidence()
        source = capture["sources"]["journal"]["path"]
        history.consume(before, source)
        history.consume(action, source)
        self.assertEqual(classify(action)["kind"], "failed_recovery")
        self.assertFalse(classify(action)["made_progress"])
        self.assertEqual(classify(action)["quantities"]["blocks_dug"], 22)
        self.assertEqual(history.guard("minecraft_mine_block", {"position": {"x": 38, "y": 0, "z": 1}})["reason"], "death_requires_replan")
        self.assertEqual(sum(row["missing"] for row in history.current_failed_recovery()["inventory_missing_after_death"]), 252)
        history.consume(after, source)
        self.assertEqual(len(history.failed_recoveries), 1)
        self.assertFalse(history.recovery_proved())

    def test_captured_current_rules_distinguish_food_from_health_recovery(self):
        capture = json.loads((CAPTURE.parent / "a13-healing-context.json").read_text(encoding="utf-8"))
        history = GameplayEvidence()
        source = capture["sources"]["journal"]["path"]
        for row in capture["rows"]:
            history.consume(row, source)
        guidance = history.recovery_guidance()["health_recovery"]
        self.assertEqual(guidance["health"], 5)
        self.assertEqual(guidance["food"], 18)
        self.assertIs(guidance["natural_regeneration"], False)
        self.assertIn("golden_apple regeneration", guidance["next_step"])
        self.assertIn("absorption is not restored health", guidance["next_step"])
        self.assertIsNotNone(history.recovery)
        self.assertIsNone(history.permit)
        self.assertTrue(any("find and eat food to restore health" in row["diagnosis"]
                            for row in capture["native_diagnoses"]))
        # Another actor needs its own current info read before this guidance.
        history.consume({"event": "epoch_acquired", "time": history.reads[-1]["time"] + 1}, "next-actor")
        self.assertIsNone(history.recovery_guidance()["health_recovery"])

    def test_captured_actor_refusals_keep_diagnosis_required_without_repeating_history(self):
        capture = json.loads((CAPTURE.parent / "a13-actor-refusal-context.json").read_text(encoding="utf-8"))
        history = GameplayEvidence()
        history.methods = {history.method_key(row["tool"], row["arguments"]): copy.deepcopy(row)
                           for row in capture["paused_methods"]}
        source = capture["sources"]["journal"]["path"]
        # Use the actual first fifteen reads before the repeated equipment refusals.
        # Older paused methods retain their actual operands and preconditions.
        for row in capture["rows"]:
            if row.get("seq", 0) <= 15 and row["event"] != "diagnosis":
                history.consume(row, source)
        paused = copy.deepcopy(history.paused_methods())
        denied = next(row for row in capture["rejections"] if row["tool"] == "minecraft_equip"
                      and row["arguments"] == {"item": "moss_block"})
        refusal = history.guard(denied["tool"], denied["arguments"])
        self.assertEqual(refusal["reason"], denied["reason"])
        self.assertIsNone(history.permit)
        self.assertLess(len(json.dumps(refusal)), 5000)
        self.assertGreater(len(json.dumps(capture["original_refusal"]["content"])), 10000)
        self.assertEqual(refusal["paused_methods"], [])
        call = refusal["diagnosis_call"]
        self.assertEqual(call["tool"], "actor_diagnose")
        self.assertNotIn("diagnosis", call["arguments"])
        self.assertTrue(call["arguments"]["recovery_action"])
        usable = {row["seq"] for row in history.reads}
        self.assertTrue(set(call["arguments"]["evidence_sequences"]).issubset(usable))
        self.assertNotIn(1, call["arguments"]["evidence_sequences"])
        self.assertEqual(call["arguments"]["evidence_sequences"], [13, 14, 15])
        # This boundary diagnosis is test text, not a native recovery claim.
        correction = history.diagnosis("Equip the owned moss to prepare a supported recovery step.", **call["arguments"])
        self.assertIsNone(history.permit)
        history.consume({**correction, "time": history.reads[-1]["time"] + 1}, source)
        self.assertIsNone(history.guard(denied["tool"], denied["arguments"]))
        self.assertIsNone(history.permit)
        self.assertEqual(history.guard(denied["tool"], denied["arguments"])["reason"], "recovery_required")
        self.assertEqual(history.paused_methods(), paused)
        self.assertEqual(history.decision_snapshot()["paused_methods"], paused)
        self.assertEqual(len(history.snapshot()["methods"]), len(history.methods))
        with self.assertRaises(ValueError):
            history.diagnosis("Missing evidence cannot authorize a recovery action.",
                              **{**call["arguments"], "evidence_sequences": [9999]})
        with self.assertRaises(ValueError):
            history.diagnosis("Optional work cannot end recovery.",
                              **{**call["arguments"], "recovery_action": False})

    def test_captured_alive_claim_cannot_establish_required_health(self):
        from workflows.minecraft_tasks.schemas import Condition
        from workflows.minecraft_tasks.verification import evaluate
        capture = json.loads((CAPTURE.parent / "a13-health-predicate.json").read_text(encoding="utf-8"))
        rejected = next(c for n in capture["proposal"]["updates"] for c in n["criteria"] if c["id"] == "rec_health")
        alive = Condition.model_validate(rejected)
        health = alive.model_copy(update={"kind": "health", "minimum": 11})
        read = {"criterion_id": alive.id, "tool": "minecraft_observe", "arguments": {"include_image": False},
                "ok": True, "result": capture["observation"]}
        self.assertEqual(capture["observation"]["player"]["health"], 5)
        self.assertTrue(evaluate([alive], [read])[alive.id])
        self.assertFalse(evaluate([health], [read])[health.id])
        self.assertIn("natural regen off", capture["server_info"]["server_rules"])
        # These projected thresholds check the primitive, not a healed player.
        self.assertTrue(evaluate([health.model_copy(update={"minimum": 5})], [read])[health.id])
        self.assertFalse(evaluate([health.model_copy(update={"minimum": 6})], [read])[health.id])
        for value in (None, True, "11", -1, 21):
            invalid = copy.deepcopy(read)
            invalid["result"]["player"]["health"] = value
            self.assertFalse(evaluate([health], [invalid])[health.id])
        wrong_dimension = copy.deepcopy(read)
        wrong_dimension["result"]["player"]["dimension"] = "the_nether"
        self.assertFalse(evaluate([health.model_copy(update={"minimum": 5})], [wrong_dimension])[health.id])
        for operands in ({"minimum": 21}, {"names": ["stone"]}, {"position": {"x": 12, "y": 31, "z": 1}}):
            with self.assertRaises(ValueError):
                Condition.model_validate({**health.model_dump(), **operands})

    def test_captured_long_mining_context_retains_decision_facts_within_budget(self):
        from workflows.minecraft_tasks.steps import scoped_decision_context, decision_schema
        from workflows.minecraft_tasks.schemas import RevisionProposal
        capture = json.loads((CAPTURE.parent / "a13-decision-scope.json").read_text(encoding="utf-8"))
        original = capture["input"]
        payload = json.loads(original["messages"][1]["content"])
        payload["latest_verification"]["verified"] = capture["verification"]["verified"]
        before = copy.deepcopy(payload)
        scoped = scoped_decision_context(payload)
        prior = self.unpack_planning_history(payload["gameplay_history"])
        current = self.unpack_planning_history(scoped["gameplay_history"])
        keyed = lambda rows: {(row["tool"], json.dumps(row["arguments"], sort_keys=True)): row for row in rows}
        prior_methods, current_methods = keyed(prior["methods"]), keyed(current["methods"])
        for key, row in prior_methods.items():
            if row["ineffective"] >= 2:
                self.assertIn(key, current_methods)
        for key, row in current_methods.items():
            old = prior_methods[key]
            for field in ("tool", "arguments", "ineffective", "last"):
                self.assertEqual(row[field], old[field])
            self.assertEqual(current["method_states"][row["state_index"]], prior["method_states"][old["state_index"]])
        for field in ("searches", "recovery", "life", "navigation", "recent_outcomes", "failed_recoveries", "outcome_count"):
            self.assertEqual(current[field], prior[field])
        self.assertEqual(len(prior["methods"]), len(current["methods"]) + sum(group["method_count"] for group in current["archived_methods"].values()))
        self.assertEqual(current["archived_methods"]["minecraft_mine_block"]["inventory_changes"]["raw_iron"]["gained"], 2)
        self.assertEqual(payload, before)
        self.assertEqual(scoped["previous_tree"], payload["previous_tree"])
        self.assertEqual(scoped["latest_verification"]["specification"]["from_node"], "site_survey")
        self.assertNotIn("reads", scoped["latest_verification"])
        guide = (Path(__file__).resolve().parents[1] / "workflows/minecraft_tasks/resources/planner.md").read_text(encoding="utf-8")
        system = guide + "\nJSON schema:\n" + json.dumps(decision_schema(RevisionProposal), separators=(",", ":"))
        self.assertEqual(original["input_characters"], 274405)
        self.assertLess(len(system) + len(json.dumps(scoped, ensure_ascii=False, separators=(",", ":"))), 118000)

    def test_captured_failed_verification_keeps_all_original_reads_when_scoped(self):
        from workflows.minecraft_tasks.steps import scoped_decision_context
        capture = json.loads((CAPTURE.parent / "a13-resumed-context.json").read_text(encoding="utf-8"))
        payload = json.loads(capture["documents"]["planner_input"]["messages"][1]["content"])
        payload["latest_verification"]["verified"] = False
        scoped = scoped_decision_context(payload)
        self.assertEqual(scoped["latest_verification"], payload["latest_verification"])

    def setUp(self):
        self.capture = json.loads(CAPTURE.read_text(encoding="utf-8"))

    def test_captured_fine_control_acknowledgement_cannot_prove_motion(self):
        capture = json.loads((CAPTURE.parent / "a13-fine-motion.json").read_text(encoding="utf-8"))
        by_seq = {row["seq"]: row for row in capture["rows"]}
        still = by_seq[104]
        self.assertTrue(still["ok"])
        self.assertEqual(still["result"]["pose_before"], still["result"]["pose_after"])
        outcome = classify(still)
        self.assertEqual(outcome["kind"], "unsuccessful_action")
        self.assertEqual(outcome["reason"], "no_motion")
        self.assertEqual(outcome["quantities"]["moved_blocks"], 0)
        self.assertFalse(outcome["made_progress"])
        step = classify(by_seq[68])
        self.assertGreater(step["quantities"]["moved_blocks"], 0)
        self.assertLess(step["quantities"]["moved_blocks"], 0.5)
        self.assertTrue(step["made_progress"])
        self.assertEqual(step["kind"], "partial_progress")

    def test_captured_zero_motion_inputs_pause_changed_navigation(self):
        capture = json.loads((CAPTURE.parent / "a13-fine-motion.json").read_text(encoding="utf-8"))
        by_seq = {row["seq"]: row for row in capture["rows"]}
        history = GameplayEvidence()
        source = capture["sources"]["journal"]["path"]
        # Replay only original no-motion rows to check the two-failure boundary.
        # This selection is not a claim that the original live calls were consecutive.
        for seq in (67, 104):
            history.consume(by_seq[seq], source)
        self.assertEqual(history.navigation["ineffective"], 2)
        for tool, arguments in (("minecraft_fine_control", {"left": True}),
                                ("minecraft_walk_to_exact", {"x": 16, "y": 86, "z": 1})):
            self.assertEqual(history.guard(tool, arguments)["reason"], "method_paused")
        self.assertIsNone(history.guard("minecraft_fine_control", {}))
        self.assertIsNone(history.guard("minecraft_observe", {"include_image": False}))
        # Input release is a contract projection, not another world action.
        released = copy.deepcopy(by_seq[104])
        released["arguments"] = {}
        self.assertFalse(classify(released)["unsuccessful"])
        history.consume(by_seq[114], source)
        self.assertEqual(history.navigation["ineffective"], 0)

    def test_actual_approved_m3_plan_still_lacks_required_house_and_root_checks(self):
        from dev.work_items.alpha_a13_minecraft_gate import m3_plan_constraints
        from workflows.minecraft_tasks.schemas import TaskTree
        capture = json.loads((CAPTURE.parent / "a13-m3-contract-gap.json").read_text(encoding="utf-8"))
        self.assertEqual(capture["documents"]["review"]["status"], "accepted")
        tree = TaskTree.model_validate(capture["documents"]["tree"])
        before = tree.model_dump(mode="json")
        result = m3_plan_constraints(tree)
        errors = " ".join(result["errors"])
        self.assertIn("construction leaves: walls lacks 1 exact cells", errors)
        self.assertIn("(28, 105, 0)", errors)
        self.assertIn("construction leaves: floor lacks 25 exact cells", errors)
        self.assertIn("root fresh checks: roof lacks 25 exact cells", errors)
        self.assertIn("Root acceptance must directly cite an accepted mining leaf", errors)
        self.assertIn("Root acceptance must directly cite an accepted smelting leaf", errors)
        self.assertIn("current inventory", errors)
        self.assertEqual(len(result["requirements"]["house"]["cells"]["walls"]), 30)
        self.assertEqual(tree.model_dump(mode="json"), before)

    def test_m3_checks_survive_actual_workflow_role_grants(self):
        from pm_workflows.kernel import Kernel
        missing = json.loads((CAPTURE.parent / "a13-plan-contract-capability.json").read_text(encoding="utf-8"))
        self.assertNotIn("minecraft_plan_constraints", missing["planner_tools"])
        self.assertIn("minecraft_plan_constraints", missing["reviewer_tools"])
        self.assertNotIn("route_test", missing["plan_constraints"]["requirements"])
        from pm_workflows.runtime import WorkflowRuntime
        from dev.work_items.alpha_a11_minecraft_gate import project_contract_manifest
        from dev.work_items.alpha_a13_minecraft_gate import m3_plan_constraints, m3_project_check
        from workflows.minecraft.module_tools import build_minecraft_module_tools
        from workflows.minecraft_tasks.schemas import TaskTree
        from working_target import FolderEnvironment
        project = Path(__file__).resolve().parents[1]
        original = project / "workflows/minecraft_tasks/minecraft_tasks.workflow.md"
        before = original.read_bytes()
        capture = json.loads((CAPTURE.parent / "a13-m3-contract-gap.json").read_text(encoding="utf-8"))
        with TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = project_contract_manifest(original, root / original.name, plan=True, completion=True, generation=True)
            tools = build_minecraft_module_tools()("ad5f4b4094a14bf6bcfe9caeead2770a", None)
            # Only test capability narrowing here. No model or world tool runs.
            tools.update(managed_instructions=None, minecraft_plan_constraints=m3_plan_constraints,
                         minecraft_project_check=m3_project_check, minecraft_decision_generation=lambda: {"reasoning": {"enabled": False}})
            kernel = Kernel(manifest, root / "workspace", "m3-role-grants", base_dir=project,
                            kernel_data_root=root / "kernel", runtime=WorkflowRuntime(
                                environment=FolderEnvironment(root / "target", bash_executable="bash"),
                                stop_event=threading.Event(), tools=tools))
            reviewer = kernel._runtime_for_role(kernel.manifest.roles["reviewer"])
            planner = kernel._runtime_for_role(kernel.manifest.roles["planner"])
            self.assertIs(planner.tools["minecraft_plan_constraints"], m3_plan_constraints)
            final = kernel._runtime_for_role(kernel.manifest.roles["final_verifier"])
            self.assertNotIn("minecraft_project_check", reviewer.tools)
            self.assertNotIn("minecraft_plan_constraints", final.tools)
            self.assertEqual(reviewer.tools["minecraft_decision_generation"](), {"reasoning": {"enabled": False}})
            result = reviewer.tools["minecraft_plan_constraints"](TaskTree.model_validate(capture["documents"]["tree"]))
            self.assertTrue(result["errors"])
            self.assertIs(final.tools["minecraft_project_check"], m3_project_check)
        self.assertEqual(original.read_bytes(), before)

    def test_captured_native_m3_repair_covers_goal_without_replacing_accepted_work(self):
        from dev.work_items.alpha_a13_minecraft_gate import m3_plan_constraints
        from workflows.minecraft_tasks.schemas import TaskTree
        before = json.loads((CAPTURE.parent / "a13-m3-contract-gap.json").read_text(encoding="utf-8"))["documents"]["tree"]
        capture = json.loads((CAPTURE.parent / "a13-native-m3-contract-repair.json").read_text(encoding="utf-8"))
        after = capture["documents"]["tree"]
        self.assertTrue(all(capture["documents"]["preservation_checks"].values()))
        self.assertEqual(m3_plan_constraints(TaskTree.model_validate(after))["errors"], [])
        self.assertEqual(capture["documents"]["constraint"]["result"]["errors"], [])
        specs = lambda tree: {n["id"]: {k: v for k, v in n.items() if k != "revision"} for n in tree["nodes"]}
        old, new = specs(before), specs(after)
        self.assertEqual(set(old), set(new))
        for key in ("craft_table", "prep_wood", "stone_pickaxe"):
            self.assertEqual(old[key], new[key])

    def test_captured_region_air_count_requires_all_variants_and_complete_coverage(self):
        from workflows.minecraft_tasks.schemas import Condition
        from workflows.minecraft_tasks.verification import evaluate
        capture = json.loads((CAPTURE.parent / "a13-region-air-count.json").read_text(encoding="utf-8"))["documents"]
        conditions = [Condition.model_validate(c) for c in capture["node"]["criteria"]]
        air = next(c for c in conditions if c.id == "ss_air_clear")
        foundation = next(c for c in conditions if c.id == "ss_foundation")
        reads = capture["verification"]["reads"]
        before = copy.deepcopy(reads)
        row = next(r for r in reads if r["criterion_id"] == air.id)
        self.assertEqual(row["result"]["blocks"], [])
        self.assertEqual(row["result"]["air_cells"], 75)
        self.assertEqual(row["result"]["unseen_cells"], 0)
        # This caller-condition projection tests the real aggregate response.
        # It does not turn the original failed native attempt into acceptance.
        all_variants = air.model_copy(update={"names": ["air", "cave_air", "void_air"]})
        self.assertTrue(evaluate([all_variants], reads)[air.id])
        self.assertFalse(evaluate([air], reads)[air.id])
        self.assertEqual(next(r for r in reads if r["criterion_id"] == foundation.id)["result"]["unseen_cells"], 1)
        self.assertFalse(evaluate([foundation], reads)[foundation.id])
        self.assertEqual(reads, before)

    def test_captured_region_air_counter_rejects_invalid_counter_projections(self):
        from workflows.minecraft_tasks.schemas import Condition
        from workflows.minecraft_tasks.verification import evaluate
        capture = json.loads((CAPTURE.parent / "a13-region-air-count.json").read_text(encoding="utf-8"))["documents"]
        air = next(Condition.model_validate(c) for c in capture["node"]["criteria"] if c["id"] == "ss_air_clear")
        all_variants = air.model_copy(update={"names": ["air", "cave_air", "void_air"]})
        original = next(r for r in capture["verification"]["reads"] if r["criterion_id"] == air.id)
        for counter in (None, True, -1, 76):
            row = copy.deepcopy(original)
            row["result"]["air_cells"] = counter
            with self.subTest(counter=counter):
                self.assertFalse(evaluate([all_variants], [row])[air.id])

    def with_captured_life(self, row):
        """Derive current-schema inputs from original geometry and real life fields.

        The original captures remain unchanged. This projection tests geometry
        with valid lifecycle operands; it cannot establish new native acceptance.
        """
        result = copy.deepcopy(row)
        if row.get("event") == "result" and row.get("tool") == "minecraft_observe" and row["ok"]:
            probe = json.loads((CAPTURE.parent / "a13-respawn-boundary.json").read_text(encoding="utf-8"))["probe"]
            baseline = probe["rows"][0]["result"]
            result["result"].update({key: baseline[key] for key in ("body_session", "death_count")})
        return result

    def test_initial_invalid_plan_repair_does_not_load_revision_zero(self):
        from workflows.minecraft_tasks.schemas import PlanProposal
        from workflows.minecraft_tasks.steps import rejected_context
        from workflows.minecraft_tasks.tree import digest
        capture = json.loads((CAPTURE.parent / "a13-initial-plan-rejection.json").read_text(encoding="utf-8"))
        raw, state = capture["documents"]["raw"], capture["documents"]["state"]
        self.assertEqual(state["revision"], 0)
        with self.assertRaisesRegex(ValueError, "cannot repeat a position"):
            PlanProposal.model_validate(json.loads(raw["text"]))
        with TemporaryDirectory() as folder:
            store = TreeStore(Path(folder))
            source = state["rejected_proposal"]
            store.write(source, raw)
            before = store.path_for(source).read_bytes()
            result = rejected_context(store, source)
            self.assertEqual(result["kind"], "invalid_model_output")
            self.assertEqual(result["original_digest"], digest(raw))
            self.assertEqual(result["candidate"], json.loads(raw["text"]))
            self.assertIn("cannot repeat a position", result["validation_errors"][0]["msg"])
            self.assertEqual(store.state["revision"], 0)
            self.assertFalse(store.path_for("planning/specs/0000.json").exists())
            self.assertEqual(store.path_for(source).read_bytes(), before)

    def test_initial_repair_retains_first_actual_geometry_after_later_rejection(self):
        from workflows.minecraft_tasks.steps import rejected_context
        from workflows.minecraft_tasks.tree import digest
        capture = json.loads((CAPTURE.parent / "a13-initial-repair-context.json").read_text(encoding="utf-8"))
        documents = capture["documents"]
        state = documents["state"]
        self.assertEqual(state["revision"], 0)
        with TemporaryDirectory() as folder:
            store = TreeStore(Path(folder))
            store.state = copy.deepcopy(state)
            first_source = state["decisions"][0]["path"] + "/raw.json"
            source = state["rejected_proposal"]
            self.assertNotEqual(first_source, source)
            store.write(first_source, documents["first_raw"])
            store.write(source, documents["latest_raw"])
            originals = {path: store.path_for(path).read_bytes() for path in (first_source, source)}
            result = rejected_context(store, source)
            first = result["first_rejected_proposal"]
            self.assertEqual(first["original_receipt"], first_source)
            self.assertEqual(first["original_digest"], digest(documents["first_raw"]))
            self.assertEqual(first["candidate"], json.loads(documents["first_raw"]["text"]))
            self.assertIn("cannot repeat a position", first["validation_errors"][0]["msg"])
            self.assertEqual(result["original_digest"], digest(documents["latest_raw"]))
            self.assertFalse(store.path_for("planning/specs/0000.json").exists())
            self.assertEqual(store.state["acceptances"], {})
            for path, original in originals.items():
                self.assertEqual(store.path_for(path).read_bytes(), original)

    def test_initial_semantic_rejection_supplies_complete_tree_and_exact_errors(self):
        from collections import Counter
        from workflows.minecraft_tasks.schemas import PlanProposal, TaskTree
        from workflows.minecraft_tasks.steps import scoped_rejection
        from workflows.minecraft_tasks.tree import validate_tree
        capture = json.loads((CAPTURE.parent / "a13-initial-tree-repair.json").read_text(encoding="utf-8"))
        original = capture["documents"]["rejected_tree"]
        tree = TaskTree.model_validate(original)
        with self.assertRaisesRegex(ValueError, "Node iron_pickaxe, condition ip_mine, references iron_mine"):
            validate_tree(tree)
        result = scoped_rejection(None, original, capture["sources"]["rejected_tree"]["path"])
        self.assertEqual(result["kind"], "invalid_initial_tree")
        self.assertNotIn("changed_nodes", result)
        self.assertEqual(TaskTree.model_validate(result["candidate_tree"]).model_dump(), tree.model_dump())
        self.assertEqual(len(result["candidate_tree"]["nodes"]), 16)
        fragment = TaskTree.model_validate(capture["documents"]["root_only_tree"])
        with self.assertRaisesRegex(ValueError, "Node wooden_home_iron_pickaxe, field children"):
            validate_tree(fragment)
        raw = json.loads((CAPTURE.parent / "a13-initial-plan-rejection.json").read_text(encoding="utf-8"))["documents"]["raw"]
        candidate = json.loads(raw["text"])
        condition = candidate["tree"]["nodes"][7]["criteria"][0]
        duplicate_cells = sorted(cell for cell, count in Counter((p["x"], p["y"], p["z"]) for p in condition["positions"]).items() if count > 1)
        self.assertTrue(duplicate_cells)
        with self.assertRaises(ValueError) as failure:
            PlanProposal.model_validate(candidate)
        self.assertIn(str(duplicate_cells), str(failure.exception))

    def test_initial_candidate_keeps_actual_complete_tree_before_root_fragment(self):
        from workflows.minecraft_tasks.steps import initial_candidate
        from workflows.minecraft_tasks.tree import validate_tree
        capture = json.loads((CAPTURE.parent / "a13-initial-tree-repair.json").read_text(encoding="utf-8"))
        with TemporaryDirectory() as folder:
            store = TreeStore(Path(folder))
            complete_source, fragment_source = "planning/rejected/0001.json", "planning/rejected/0002.json"
            store.write(complete_source, capture["documents"]["rejected_tree"])
            store.write(fragment_source, capture["documents"]["root_only_tree"])
            store.state["rejections"] = 2
            before = {source: store.path_for(source).read_bytes() for source in (complete_source, fragment_source)}
            source, tree = initial_candidate(store)
            self.assertEqual(source, complete_source)
            self.assertEqual(tree.model_dump(mode="json"), capture["documents"]["rejected_tree"])
            with self.assertRaisesRegex(ValueError, "Node iron_pickaxe, condition ip_mine"):
                validate_tree(tree)
            self.assertEqual(store.state["revision"], 0)
            self.assertEqual(store.state["acceptances"], {})
            self.assertFalse(store.path_for("planning/specs/0001.json").exists())
            for source, data in before.items():
                self.assertEqual(store.path_for(source).read_bytes(), data)
            store.state["revision"] = 1
            self.assertIsNone(initial_candidate(store))

    def test_actual_incomplete_node_update_is_rejected_by_generation_schema(self):
        from jsonschema.validators import validator_for
        from workflows.minecraft_tasks.schemas import RevisionProposal, TaskTree
        from workflows.minecraft_tasks.steps import apply_node_updates, decision_schema
        from workflows.minecraft_tasks.tree import validate_tree
        capture = json.loads((CAPTURE.parent / "a13-incomplete-node-update.json").read_text(encoding="utf-8"))
        raw = json.loads(capture["documents"]["response"]["choices"][0]["message"]["content"])
        self.assertNotIn("parent", raw["updates"][0])
        self.assertNotIn("depends_on", raw["updates"][0])
        original_schema = RevisionProposal.model_json_schema()
        schema = decision_schema(RevisionProposal)
        errors = list(validator_for(schema)(schema).iter_errors(raw))
        self.assertTrue(any(error.validator == "required" and "parent" in error.message for error in errors))
        self.assertTrue(any(error.validator == "required" and "depends_on" in error.message for error in errors))
        self.assertEqual(RevisionProposal.model_json_schema(), original_schema)
        proposal = RevisionProposal.model_validate(raw)
        base = TaskTree.model_validate(capture["documents"]["base_tree"])
        with self.assertRaisesRegex(ValueError, "exactly one root goal"):
            validate_tree(apply_node_updates(base, proposal.updates))

    def test_actual_model_candidate_repair_keeps_all_other_node_operands(self):
        from jsonschema.validators import validator_for
        from workflows.minecraft_tasks.schemas import RevisionProposal, TaskTree
        from workflows.minecraft_tasks.steps import apply_node_updates, decision_schema
        from workflows.minecraft_tasks.tree import node_digest, validate_tree
        capture = json.loads((CAPTURE.parent / "a13-incomplete-node-update.json").read_text(encoding="utf-8"))
        raw = json.loads(capture["documents"]["repair_response"]["choices"][0]["message"]["content"])
        schema = decision_schema(RevisionProposal)
        self.assertEqual(list(validator_for(schema)(schema).iter_errors(raw)), [])
        proposal = RevisionProposal.model_validate(raw)
        self.assertEqual([node.id for node in proposal.updates], ["iron_pickaxe"])
        base = TaskTree.model_validate(capture["documents"]["base_tree"])
        original = base.model_dump()
        merged = apply_node_updates(base, proposal.updates)
        validate_tree(merged)
        self.assertEqual(base.model_dump(), original)
        old = {node.id: node_digest(node) for node in base.nodes}
        for node in merged.nodes:
            if node.id != "iron_pickaxe":
                self.assertEqual(node_digest(node), old[node.id])
        leaf = next(node for node in merged.nodes if node.id == "iron_pickaxe")
        self.assertIn("iron_mine", leaf.depends_on)
        self.assertIn("iron_smelt", leaf.depends_on)
        self.assertEqual(len(merged.nodes), 16)

    def test_actual_invalid_memory_operands_and_valid_calls_use_public_schema(self):
        from runtime.minecraft_actor import argument_errors
        capture = json.loads((CAPTURE.parent / "a13-invalid-arguments.json").read_text(encoding="utf-8"))
        invalid = next(row for entry in capture["journals"] for row in entry["rows"]
                       if row.get("event") == "uncertain")
        self.assertIn("literal_error", invalid["error"])
        errors = argument_errors(capture["schema"], invalid["arguments"])
        self.assertEqual([error["path"] for error in errors], ["$.kind"])
        valid = [row for entry in capture["journals"] for row in entry["rows"]
                 if row.get("event") == "result" and row["tool"] == "minecraft_remember"]
        self.assertEqual(len(valid), 2)
        for row in valid:
            self.assertEqual(argument_errors(capture["schema"], row["arguments"]), [])

    def test_local_operand_rejection_does_not_create_uncertain_effects(self):
        capture = json.loads((CAPTURE.parent / "a13-invalid-arguments.json").read_text(encoding="utf-8"))
        entry = next(entry for entry in capture["journals"] if entry["case"] == "navigation")
        observation = next(row for row in entry["rows"] if row.get("event") == "result")
        invalid = next(row for row in entry["rows"] if row.get("event") == "uncertain")
        evidence = GameplayEvidence()
        evidence.consume(observation, entry["source"])
        outcomes = copy.deepcopy(evidence.outcomes)
        # Derive the local rejection event from the actual malformed call.
        # The original upstream uncertainty remains in the captured fixture.
        evidence.consume({**invalid, "event": "arguments_rejected"}, entry["source"])
        self.assertIsNone(evidence.service_wait)
        self.assertEqual(evidence.outcomes, outcomes)
        self.assertIsNone(evidence.guard("minecraft_observe", {"include_image": False}))

    def test_captured_default_argument_diagnosis_releases_one_equivalent_call(self):
        capture = json.loads((CAPTURE.parent / "a13-default-arguments.json").read_text(encoding="utf-8"))
        evidence = GameplayEvidence(capture["tool_schemas"])
        source = capture["source"]
        for row in capture["rows"]:
            if row.get("event") == "diagnosis" and row["next_tool"] == "minecraft_craft_item":
                diagnosis = row
                break
            evidence.consume(row, source)
        arguments = {"item": "wooden_pickaxe"}
        self.assertEqual(diagnosis["next_arguments"], {**arguments, "repetitions": 1})
        self.assertEqual(evidence.guard("minecraft_craft_item", arguments)["reason"], "method_paused")
        record = evidence.diagnosis(diagnosis["diagnosis"], diagnosis["evidence_sequences"],
                                   diagnosis["next_tool"], diagnosis["next_arguments"], diagnosis["recovery_action"])
        evidence.consume({**record, "time": diagnosis["time"]}, source)
        self.assertIsNone(evidence.guard("minecraft_craft_item", arguments))
        self.assertEqual(evidence.guard("minecraft_craft_item", arguments)["reason"], "method_paused")
        failures = [row for row in evidence.outcomes if row["tool"] == "minecraft_craft_item"
                    and row["arguments"].get("item") == "wooden_pickaxe"]
        self.assertTrue(all("repetitions" not in row["arguments"] for row in failures))

    def test_captured_defaults_preserve_changed_repetitions_and_original_receipts(self):
        capture = json.loads((CAPTURE.parent / "a13-default-arguments.json").read_text(encoding="utf-8"))
        evidence = GameplayEvidence(capture["tool_schemas"])
        arguments = {"item": "wooden_pickaxe"}
        self.assertEqual(evidence.method_key("minecraft_craft_item", arguments),
                         evidence.method_key("minecraft_craft_item", {**arguments, "repetitions": 1}))
        self.assertNotEqual(evidence.method_key("minecraft_craft_item", arguments),
                            evidence.method_key("minecraft_craft_item", {**arguments, "repetitions": 2}))
        for row in capture["rows"]:
            evidence.consume(row, capture["source"])
        self.assertEqual(evidence.service_wait["seq"], 31)
        self.assertEqual(evidence.guard("minecraft_equip", {"item": "birch_log"})["reason"], "service_wait")
        self.assertIn("updateSlot:0", capture["body_error"])

    def test_actual_respawn_cannot_prove_survival_or_unlock_mutation(self):
        from runtime.minecraft_outcomes import safe_landing
        capture = json.loads((CAPTURE.parent / "a13-respawn-boundary.json").read_text(encoding="utf-8"))["probe"]
        evidence = GameplayEvidence()
        source = capture["gameplay"]["failed_recoveries"][0]["source"]
        for row in capture["rows"]:
            evidence.consume(row, source)
        self.assertEqual(evidence.life["death_count"], 1)
        self.assertEqual(len(evidence.failed_recoveries), 1)
        self.assertFalse(evidence.failed_recoveries[0]["completion_verified"])
        self.assertGreaterEqual(evidence.stable_observations, 2)
        self.assertTrue(safe_landing(*(row["result"] for row in capture["rows"][2:5])))
        self.assertFalse(evidence.recovery_proved())
        self.assertEqual(evidence.guard("minecraft_craft_item", {"item": "stick"})["reason"], "death_requires_replan")
        self.assertIsNone(evidence.guard("minecraft_observe", {"include_image": False}))
        with self.assertRaisesRegex(ValueError, "player died"):
            evidence.diagnosis("Respawn restored full vitals.", [7], "minecraft_craft_item", {"item": "stick"}, True)

    def test_original_death_without_life_metadata_cannot_prove_recovery(self):
        capture = json.loads((CAPTURE.parent / "a13-respawn-boundary.json").read_text(encoding="utf-8"))["original_failed_recovery"]
        self.assertFalse(capture["native_food_recovery"])
        self.assertEqual(len(capture["death_lines"]), 1)
        evidence = GameplayEvidence()
        evidence.consume(capture["last_predeath_observation"], capture["original_journal"])
        evidence.consume(capture["first_respawn_observation"], capture["original_journal"])
        self.assertEqual(evidence.observation["player"]["food"], 20)
        self.assertIsNone(evidence.life)
        self.assertFalse(evidence.recovery_proved())
        self.assertIsNotNone(evidence.recovery)

    def test_observed_death_counter_cannot_decrease_in_one_body_session(self):
        capture = json.loads((CAPTURE.parent / "a13-respawn-boundary.json").read_text(encoding="utf-8"))["probe"]
        evidence = GameplayEvidence()
        source = capture["gameplay"]["failed_recoveries"][0]["source"]
        after = capture["rows"][1]
        evidence.consume(after, source)
        # Derive an invalid counter from the original post-death observation.
        invalid = copy.deepcopy(after)
        invalid["time"] += 1
        invalid["result"]["death_count"] = 0
        with self.assertRaisesRegex(ValueError, "cannot decrease"):
            evidence.consume(invalid, source)

    def test_captured_alive_body_does_not_prove_food_recovery(self):
        from workflows.minecraft_tasks.schemas import Condition
        from workflows.minecraft_tasks.verification import evaluate
        capture = json.loads((CAPTURE.parent / "a13-review-context.json").read_text(encoding="utf-8"))
        food = Condition(id="food", description="Food has recovered above the low-supplies threshold.", kind="food", minimum=7)
        alive = Condition(id="alive", description="The body is alive.", kind="alive")
        observed = capture["low_food_observation"]
        self.assertEqual(observed["player"]["food"], 5)
        reads = [{"criterion_id": condition.id, "tool": "minecraft_observe", "arguments": {"include_image": False},
                  "ok": True, "result": observed} for condition in (food, alive)]
        self.assertEqual(evaluate([food, alive], reads), {"food": False, "alive": True})
        recovered = {**reads[0], "result": capture["prior_food_observation"]["result"]}
        self.assertEqual(recovered["result"]["player"]["food"], 9)
        self.assertEqual(evaluate([food], [recovered]), {"food": True})
        # Derive invalid reads from the original capture to test fail-closed checks.
        # These inputs do not establish a native recovery or another live gate.
        for replacement in ({}, {"food": 30}, {"food": True}, {"food": 9, "dimension": "the_nether"}):
            invalid = {**recovered, "result": {**recovered["result"], "player": replacement}}
            self.assertEqual(evaluate([food], [invalid]), {"food": False})

    def test_food_condition_rejects_item_geometry_and_out_of_range_thresholds(self):
        from workflows.minecraft_tasks.schemas import Condition
        base = {"id": "food", "description": "Enough food for normal work.", "kind": "food", "minimum": 7}
        self.assertEqual(Condition.model_validate(base).minimum, 7)
        for operands in ({"minimum": 21}, {"minimum": -1}, {"names": ["bread"]},
                         {"position": {"x": 32, "y": 100, "z": 2}}, {"node_id": "recover"}):
            with self.assertRaises(ValueError):
                Condition.model_validate({**base, **operands})

    def test_original_waiting_review_already_received_recipe_operands(self):
        from workflows.minecraft_tasks.steps import scoped_reads
        capture = json.loads((CAPTURE.parent / "a13-review-context.json").read_text(encoding="utf-8"))
        payload = json.loads(capture["original_input"]["messages"][1]["content"])
        recipes = scoped_reads(capture["original_survey"]["recipes"])
        self.assertEqual(payload["current_site_survey"]["recipes"], recipes)
        by_pattern = {row["arguments"]["pattern"]: row["result"] for row in recipes}
        self.assertEqual(by_pattern["wooden_door"]["reason"], "not_found")
        self.assertTrue(by_pattern["iron_ingot"]["matches"])
        self.assertTrue(by_pattern["iron_pickaxe"]["matches"])
        self.assertEqual({row["name"] for row in capture["tool_contracts"]},
                         {"minecraft_recipe_search", "minecraft_smelt_item", "minecraft_eat_best"})

    def test_recipe_only_survey_retains_explicit_refinement_queries(self):
        from workflows.minecraft_tasks.schemas import PlanProposal, RefinementProposal
        from workflows.minecraft_tasks.steps import refinement_plan
        from workflows.minecraft_tasks.schemas import TaskTree
        capture = json.loads((CAPTURE.parent / "a13-review-context.json").read_text(encoding="utf-8"))
        original = json.loads(capture["original_input"]["messages"][1]["content"])
        patterns = [row["arguments"]["pattern"] for row in original["current_site_survey"]["recipes"]]
        # Select original lookup operands to test the read-only proposal boundary.
        # No fixed model response or new world observation supplies acceptance.
        proposal = PlanProposal(status="survey", summary="Read original recipe operands.", recipe_queries=patterns)
        self.assertEqual(proposal.queries, [])
        refined = RefinementProposal(status="survey", summary=proposal.summary, replacement=None, recipe_queries=patterns)
        mapped = refinement_plan(TaskTree.model_validate(original["tree"]), "recover", refined)
        self.assertEqual(mapped.recipe_queries, patterns)
        for invalid in ({"status": "waiting", "recipe_queries": patterns}, {"status": "survey", "recipe_queries": [""]}):
            with self.assertRaises(ValueError):
                PlanProposal.model_validate({"summary": "Invalid lookup scope.", **invalid})

    def test_captured_waiting_plan_gets_fresh_review_without_erasing_partial_work(self):
        from workflows.minecraft_tasks.steps import restart_review
        capture = json.loads((CAPTURE.parent / "a13-review-context.json").read_text(encoding="utf-8"))
        original = capture["waiting_planner"]["planning_state"]
        with TemporaryDirectory() as folder:
            store = TreeStore(Path(folder))
            store.state = copy.deepcopy(original)
            self.assertTrue(restart_review(store, "plan", original["instruction_revision"], ""))
            for key in ("revision", "acceptances", "decisions", "attempts", "feedback"):
                self.assertEqual(store.state[key], original[key])
            self.assertEqual(store.state["recovery_observation"], original["planning_observation"])
            for phase, revision, answer in (("replan", 0, ""), ("plan", 1, ""), ("plan", 0, "Use birch wood.")):
                self.assertFalse(restart_review(store, phase, revision, answer))
            store.state["require_leaf_change"] = True
            self.assertFalse(restart_review(store, "plan", 0, ""))
            # Derive an interrupted boundary from the original attempt.
            store.state["attempts"][-1]["state"] = "running"
            self.assertFalse(restart_review(store, "plan", 0, ""))
            self.assertEqual(store.state["attempts"][-1]["state"], "interrupted")

    def test_captured_removed_nodes_repair_uses_authoritative_tree_and_explicit_updates(self):
        from workflows.minecraft_tasks.schemas import RevisionProposal, TaskTree
        from workflows.minecraft_tasks.steps import scoped_rejection, compact_plan_context, scoped_decision_context, decision_schema
        capture = json.loads((CAPTURE.parent / "a13-review-context.json").read_text(encoding="utf-8"))["removed_nodes_repair"]
        original = capture["original_input"]
        payload = json.loads(original["messages"][1]["content"])
        previous = TaskTree.model_validate(payload["previous_tree"])
        proposed = capture["original_proposal"]
        rejected = TaskTree.model_validate(proposed["tree"])
        scoped = scoped_rejection(previous, rejected.model_dump(mode="json"), payload["rejected_proposal"]["original_receipt"])
        self.assertEqual(scoped["kind"], "removed_stable_nodes")
        self.assertEqual(scoped["removed_nodes"], ["p_craft", "p_gather", "pick", "s_site", "site", "w_craft", "w_gather", "wood"])
        self.assertEqual(scoped["original_digest"], payload["rejected_proposal"]["original_digest"])
        with self.assertRaises(ValueError):
            RevisionProposal.model_validate(proposed)
        changed = next(node for node in rejected.nodes if node.id == "recover")
        update = RevisionProposal(status="planned", summary="Use one explicit captured node update.", updates=[changed])
        self.assertEqual(update.updates[0].id, "recover")
        payload["rejected_proposal"] = scoped
        compact = scoped_decision_context(compact_plan_context(payload))
        guide = (Path(__file__).resolve().parents[1] / "workflows/minecraft_tasks/resources/planner.md").read_text(encoding="utf-8")
        system = guide + "\nJSON schema:\n" + json.dumps(decision_schema(RevisionProposal), separators=(",", ":"))
        characters = len(system) + len(json.dumps(compact, ensure_ascii=False, separators=(",", ":")))
        self.assertGreater(original["input_characters"], 123000)
        self.assertLess(characters, original["input_characters"])

    def navigation_capture(self, through):
        capture = json.loads((CAPTURE.parent / "a13-navigation-stall.json").read_text(encoding="utf-8"))
        history = GameplayEvidence()
        for row in capture["rows"]:
            if row.get("seq", 0) > through:
                break
            history.consume(row, capture["source"])
        return history, capture

    def unpack_planning_history(self, scoped):
        """Decode named columns independently to compare captured facts."""
        def sources(value):
            if isinstance(value, list):
                return [sources(row) for row in value]
            if not isinstance(value, dict):
                return value
            result = {key: sources(row) for key, row in value.items()}
            if "source_index" in result:
                result["source"] = scoped["source_paths"][result.pop("source_index")]
            return result
        result = sources(scoped)
        if "inventory_items" in scoped:
            states = []
            for state in scoped["method_states"]:
                if "method_state_fields" in scoped:
                    state = dict(zip(scoped["method_state_fields"], state))
                restored = {key: value for key, value in state.items() if key not in {"inventory_indexes", "held_item_index"}}
                restored["inventory"] = [scoped["inventory_items"][index] for index in state["inventory_indexes"]]
                held = state["held_item_index"]
                restored["held_item"] = scoped["inventory_items"][held] if held is not None else None
                states.append(restored)
            result["method_states"] = states
        methods = []
        for row in scoped["methods"]:
            method = dict(zip(scoped["method_fields"], row))
            if "tool_index" in method:
                method["tool"] = scoped["tool_names"][method.pop("tool_index")]
            last = dict(zip(scoped["outcome_fields"], method["last"]))
            additional = last.pop("additional_fields")
            last["kind"] = scoped["outcome_kinds"][last.pop("kind_index")]
            method["last"] = sources({"tool": method["tool"], "arguments": method["arguments"], **last, **additional})
            methods.append(method)
        result["methods"] = methods
        result["recent_outcomes"] = [{**methods[row["method_index"]]["last"], **row.get("changes", {})} if "method_index" in row else row
                                     for row in result["recent_outcomes"]]
        searches = ([sources(dict(zip(scoped["search_fields"], row))) for row in scoped["searches"]]
                    if "search_fields" in scoped else result["searches"])
        result["searches"] = [sources({**scoped["search_defaults"], **row}) for row in searches]
        if "search_coverages" in scoped:
            coverages = [dict(zip(scoped["search_coverage_field_sets"][index], values)) if index is not None else values
                         for index, values in scoped["search_coverages"]]
            if "search_coverage_vector_fields" in scoped:
                axes = scoped["search_coverage_vector_fields"]
                coverages = [{**row, "center": dict(zip(axes, row["center"])),
                              "bounds": {**row["bounds"], **{key: dict(zip(axes, row["bounds"][key])) for key in ("min", "max")}}}
                             if isinstance(row, dict) else row for row in coverages]
            for row in result["searches"]:
                row["coverage"] = coverages[row.pop("coverage_index")]
        return result

    def test_captured_resumed_planning_input_preserves_all_recent_outcomes_within_budget(self):
        from workflows.minecraft_tasks.steps import scoped_gameplay, decision_schema
        from workflows.minecraft_tasks.schemas import RevisionProposal
        capture = json.loads((CAPTURE.parent / "a13-resumed-context.json").read_text(encoding="utf-8"))
        original = capture["documents"]["planner_input"]
        payload = json.loads(original["messages"][1]["content"])
        before = copy.deepcopy(payload)
        expanded = self.unpack_planning_history(payload["gameplay_history"])
        history = {**expanded, "methods": [{"tool": row["tool"], "arguments": row["arguments"],
            "ineffective": row["ineffective"], "state": payload["gameplay_history"]["method_states"][row["state_index"]],
            "last": row["last"]} for row in expanded["methods"]]}
        scoped = scoped_gameplay(history)
        restored = self.unpack_planning_history(scoped)
        self.assertEqual(restored["recent_outcomes"], expanded["recent_outcomes"])
        for prior, current in zip(expanded["methods"], restored["methods"]):
            for key in ("tool", "arguments", "ineffective", "last"):
                self.assertEqual(prior[key], current[key])
        for key in ("searches", "recovery", "navigation", "service_wait", "life", "failed_recoveries", "outcome_count", "diagnoses"):
            self.assertEqual(restored[key], expanded[key])
        self.assertTrue(any(row.get("changes", {}).get("seq") for row in scoped["recent_outcomes"]))
        payload["gameplay_history"] = scoped
        guide = (Path(__file__).resolve().parents[1] / "workflows/minecraft_tasks/resources/planner.md").read_text(encoding="utf-8")
        system = guide + "\nJSON schema:\n" + json.dumps(decision_schema(RevisionProposal), separators=(",", ":"))
        characters = len(system) + len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        self.assertEqual(original["input_characters"], 123694)
        self.assertLess(characters, original["input_characters"])
        self.assertEqual(json.loads(original["messages"][1]["content"]), before)

    def test_captured_current_leaf_operands_override_historical_success_claim(self):
        from workflows.minecraft.actor import leaf_execution_context
        from workflows.minecraft.schemas import MinecraftRequest
        capture = json.loads((CAPTURE.parent / "a13-resumed-context.json").read_text(encoding="utf-8"))
        request = MinecraftRequest.model_validate(capture["documents"]["leaf_request"])
        context = leaf_execution_context(request)
        foundation = next(c for c in context["criteria"] if c["id"] == "ss_foundation")
        self.assertEqual(foundation["kind"], "block")
        self.assertEqual(len(foundation["positions"]), 25)
        self.assertIn({"x": 30, "y": 102, "z": 3}, foundation["positions"])
        self.assertIn("fill it", context["instruction"])
        self.assertIn("15 minimum", capture["documents"]["actor_task_card"])
        self.assertNotIn("previous_attempt", context)
        self.assertEqual(context["identity"]["actor_attempt"], "actor-0011")

    def test_captured_observation_loop_keeps_current_reads_beside_leaf_operands(self):
        from workflows.minecraft.actor import leaf_execution_context
        from workflows.minecraft.schemas import MinecraftRequest
        capture = json.loads((CAPTURE.parent / "a13-native-observation-loop.json").read_text(encoding="utf-8"))
        request = MinecraftRequest.model_validate(capture["leaf_request"])
        context = leaf_execution_context(request)
        observations = context["current_criterion_observations"]
        self.assertEqual(observations, request.acceptance["current_criterion_observations"])
        self.assertEqual(observations["identity"], capture["full_observations"]["identity"])
        self.assertTrue(all(observations["checks"].values()))
        self.assertEqual(capture["native_generation"]["reasoning"], {"enabled": False})
        self.assertEqual(capture["completed_calls"], 888)
        self.assertNotIn("prerequisites", context)

    def test_captured_caller_requirements_reach_private_leaf_context(self):
        from workflows.minecraft.actor import leaf_execution_context
        from workflows.minecraft.schemas import MinecraftRequest
        from workflows.minecraft_tasks.steps import leaf_project_context
        capture = json.loads((CAPTURE.parent / "a13-caller-actor-context.json").read_text(encoding="utf-8"))
        original = capture["actor_request"]
        self.assertNotIn("caller_constraints", original["acceptance"]["project"])
        self.assertFalse(capture["publication_checks"]["saved_home"])
        self.assertFalse(capture["publication_checks"]["tested_saved_route"])
        with TemporaryDirectory() as folder:
            store = TreeStore(Path(folder))
            store.state["goal"] = original["acceptance"]["project"]["goal"]
            relative = capture["caller_constraints"]["original_receipt"]
            store.state["planning_constraints"] = relative
            store.write(relative, capture["constraint_proof"])
            project = leaf_project_context(store, original["acceptance"]["project"]["ancestor_specifications"])
        self.assertEqual(project["caller_constraints"], capture["caller_constraints"])
        updated = copy.deepcopy(original)
        updated["acceptance"]["project"] = project
        context = leaf_execution_context(MinecraftRequest.model_validate(updated))
        self.assertEqual(context["caller_constraints"], capture["caller_constraints"])
        self.assertEqual(context["criteria"], original["acceptance"]["criteria"])
        self.assertEqual(context["instruction"], original["goal"])
        self.assertNotIn("prerequisites", context)

    def test_captured_geometry_remains_in_scoped_observations(self):
        from workflows.minecraft_tasks.steps import scoped_reads
        capture = json.loads((CAPTURE.parent / "a13-block-geometry.json").read_text(encoding="utf-8"))
        original = [row for row in capture["confirmed_probe"]["calls"] if row["tool"] == "minecraft_inspect_block"]
        scoped = scoped_reads(original)
        for source, result in zip(original, scoped):
            self.assertEqual(result["result"]["properties"], source["result"]["properties"])
            self.assertEqual(result["result"]["collision_shapes"], source["result"]["collision_shapes"])
            self.assertTrue(result["observation_available"])
        furnace = next(row["result"] for row in scoped if row["result"]["block_name"] == "furnace")
        self.assertIs(furnace["properties"]["lit"], False)
        self.assertEqual(furnace["properties"]["facing"], "north")
        self.assertEqual(furnace["collision_shapes"], [[0, 0, 0, 1, 1, 1]])
        self.assertEqual(capture["failed_initial_assumption"]["status"], "failed")

    def test_captured_food_repair_budget_shares_only_identical_context(self):
        from workflows.minecraft_tasks.steps import compact_plan_context, scoped_gameplay, decision_schema
        from workflows.minecraft_tasks.schemas import RevisionProposal
        capture = json.loads((CAPTURE.parent / "a13-review-context.json").read_text(encoding="utf-8"))
        original = capture["repair_budget"]["original_input"]
        payload = json.loads(original["messages"][1]["content"])
        compact = compact_plan_context(payload)
        self.assertTrue(compact["previous_tree"])
        self.assertFalse(compact["refinement_target"])
        self.assertEqual(compact["latest_plan_review"], payload["latest_plan_review"])
        self.assertNotEqual(compact["feedback"], payload["feedback"])
        old = payload["previous_site_survey"]["tool_contracts"]
        remaining = compact["previous_site_survey"]["tool_contracts"]
        for row in old:
            self.assertIn(row, remaining + compact["current_tool_contracts"])
        for key in ("identity", "original_receipt", "original_digest", "recipes"):
            self.assertEqual(compact["previous_site_survey"][key], payload["previous_site_survey"][key])
        history = payload["gameplay_history"]
        expanded = self.unpack_planning_history(history)
        raw = {key: value for key, value in expanded.items() if key not in {
            "methods", "method_states", "method_fields", "outcome_fields", "source_paths", "outcome_kinds", "search_defaults", "tool_names"}}
        raw["methods"] = [{"tool": row["tool"], "arguments": row["arguments"], "ineffective": row["ineffective"],
                           "state": history["method_states"][row["state_index"]], "last": row["last"]}
                          for row in expanded["methods"]]
        compact["gameplay_history"] = scoped_gameplay(raw)
        self.assertEqual(self.unpack_planning_history(compact["gameplay_history"])["methods"], expanded["methods"])
        guide = (Path(__file__).resolve().parents[1] / "workflows/minecraft_tasks/resources/planner.md").read_text(encoding="utf-8")
        # This captured repair has an accepted tree and no refinement target.
        # Current production therefore sends the explicit revision contract.
        system = guide + "\nJSON schema:\n" + json.dumps(decision_schema(RevisionProposal), separators=(",", ":"))
        characters = len(system) + len(json.dumps(compact, ensure_ascii=False, separators=(",", ":")))
        self.assertGreater(original["input_characters"], 122000)
        self.assertLess(characters, original["input_characters"])
        differing = copy.deepcopy(payload)
        differing["feedback"] = "A later actual read invalidated the old review."
        differing["previous_site_survey"]["tool_contracts"][0]["description"] += " Original historical wording."
        projected = compact_plan_context(differing)
        self.assertEqual(projected["feedback"], differing["feedback"])
        self.assertIn(differing["previous_site_survey"]["tool_contracts"][0], projected["previous_site_survey"]["tool_contracts"])

    def test_captured_timeout_retains_measured_partial_travel(self):
        history, _ = self.navigation_capture(5)
        outcome = history.outcomes[-1]
        self.assertTrue(outcome["unsuccessful"])
        self.assertTrue(outcome["made_progress"])
        self.assertEqual(outcome["kind"], "partial_progress")
        self.assertGreater(outcome["quantities"]["moved_blocks"], 28)
        self.assertFalse(outcome["completion_verified"])
        self.assertEqual(history.navigation["ineffective"], 0)

    def test_captured_stalled_walks_pause_changed_destinations_and_walkers(self):
        history, capture = self.navigation_capture(8)
        next_call = next(row for row in capture["rows"] if row.get("seq") == 56 and row["event"] == "dispatch")
        self.assertEqual(history.navigation["ineffective"], 2)
        self.assertEqual(len(history.methods), 3)
        for tool in ("minecraft_walk_to_surface", "minecraft_walk_to_visible", "minecraft_walk_to_exact"):
            rejected = history.guard(tool, next_call["arguments"])
            self.assertEqual(rejected["reason"], "method_paused")
            self.assertEqual(rejected["navigation"]["ineffective"], 2)
        settling = history.outcomes[-2]
        self.assertLess(settling["quantities"]["moved_blocks"], 0.5)
        self.assertFalse(settling["made_progress"])

    def test_captured_navigation_diagnosis_requires_fresh_body_evidence(self):
        history, capture = self.navigation_capture(10)
        inspection = next(row for row in capture["rows"] if row.get("seq") == 20 and row["event"] == "result")
        # Test the diagnosis boundary with original body and floor reads.
        # This selected-input check does not establish another native action.
        history.consume(inspection, capture["source"])
        arguments = next(row["arguments"] for row in capture["rows"] if row.get("seq") == 42 and row["event"] == "dispatch")
        with self.assertRaisesRegex(ValueError, "Stalled walking needs a fresh body observation"):
            history.diagnosis("The local walk stalled; inspect another approach.", [20], "minecraft_walk_to_surface", arguments, False)
        record = history.diagnosis("Fresh body evidence retains the stalled location and supplies; change the approach.",
                                   [10, 20], "minecraft_walk_to_surface", arguments, False)
        self.assertEqual(record["next_arguments"], arguments)
        self.assertEqual(record["evidence_sequences"], [10, 20])

    def test_captured_successful_travel_clears_navigation_stall(self):
        before, _ = self.navigation_capture(46)
        after, _ = self.navigation_capture(47)
        self.assertEqual(before.navigation["ineffective"], 1)
        self.assertEqual(after.navigation["ineffective"], 0)
        self.assertGreater(after.outcomes[-1]["quantities"]["moved_blocks"], 7)

    def test_old_captured_body_read_cannot_clear_new_navigation_stall(self):
        history, capture = self.navigation_capture(8)
        position = history.position.copy()
        original = next(row for row in capture["rows"] if row.get("seq") == 1 and row["event"] == "result")
        history.consume(original, "older-planning-observation")
        self.assertEqual(history.position, position)
        self.assertEqual(history.navigation["ineffective"], 2)

    def test_original_m3_planning_history_fits_without_losing_methods_or_partial_work(self):
        from workflows.minecraft_tasks.steps import scoped_gameplay
        capture = json.loads((CAPTURE.parent / "a13-planning-budget.json").read_text(encoding="utf-8"))
        original = capture["original_input"]
        payload = json.loads(original["messages"][1]["content"])
        history = payload["gameplay_history"]
        self.assertGreater(original["input_characters"], 120000)
        scoped = scoped_gameplay(history)
        expanded = self.unpack_planning_history(scoped)
        self.assertEqual(len(scoped["methods"]), len(history["methods"]))
        for before, after in zip(history["methods"], expanded["methods"]):
            state = expanded["method_states"][after["state_index"]]
            for key in ("position", "dimension"):
                self.assertEqual(state[key], before["state"][key])
            for key in ("inventory", "held_item"):
                items = before["state"][key]
                projected = state[key]
                if key == "held_item":
                    items, projected = [items] if items else [], [projected] if projected else []
                facts = lambda rows: [(item["name"], item["count"], item.get("durability"), item.get("max_durability"), item.get("slot")) for item in rows]
                self.assertEqual(facts(items), facts(projected))
            self.assertEqual(after["last"], before["last"])
            self.assertEqual(after["ineffective"], before["ineffective"])
        for key in ("searches", "recovery", "service_wait", "recent_outcomes", "outcome_count", "diagnoses"):
            self.assertEqual(expanded[key], history[key])
        payload["gameplay_history"] = scoped
        characters = len(original["messages"][0]["content"]) + len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        self.assertLess(characters, original["input_characters"])

    def test_accumulated_captured_m3_history_fits_with_every_method_and_quantity(self):
        from workflows.minecraft_tasks.steps import scoped_gameplay
        capture = json.loads((CAPTURE.parent / "a13-planning-budget.json").read_text(encoding="utf-8"))
        original = capture["accumulated_input"]["original_input"]
        payload = json.loads(original["messages"][1]["content"])
        history = payload["gameplay_history"]
        raw = {key: value for key, value in history.items() if key != "method_states"}
        raw["methods"] = [{"tool": row["tool"], "arguments": row["arguments"], "ineffective": row["ineffective"],
                           "state": history["method_states"][row["state_index"]],
                           "last": {"tool": row["tool"], "arguments": row["arguments"], **row["last"]}}
                          for row in history["methods"]]
        scoped = scoped_gameplay(raw)
        expanded = self.unpack_planning_history(scoped)
        self.assertEqual(len(expanded["methods"]), 146)
        for before, after in zip(raw["methods"], expanded["methods"]):
            self.assertEqual(after["last"], before["last"])
            self.assertEqual(after["arguments"], before["arguments"])
            self.assertEqual(after["tool"], before["tool"])
            self.assertEqual(after["ineffective"], before["ineffective"])
            self.assertEqual(expanded["method_states"][after["state_index"]], before["state"])
        for key in ("searches", "recovery", "navigation", "service_wait", "recent_outcomes", "outcome_count", "diagnoses"):
            self.assertEqual(expanded[key], history[key])
        payload["gameplay_history"] = scoped
        characters = len(original["messages"][0]["content"]) + len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        self.assertGreater(original["input_characters"], 154000)
        self.assertLess(characters, original["input_characters"])

    def test_original_rejected_m3_revision_keeps_exact_changed_conditions_within_budget(self):
        from workflows.minecraft_tasks.schemas import TaskTree
        from workflows.minecraft_tasks.steps import scoped_rejected_tree
        from workflows.minecraft_tasks.tree import node_digest
        capture = json.loads((CAPTURE.parent / "a13-planning-budget.json").read_text(encoding="utf-8"))
        original = capture["repair_input"]["original_input"]
        payload = json.loads(original["messages"][1]["content"])
        previous = TaskTree.model_validate(payload["previous_tree"])
        rejected = TaskTree.model_validate(payload["rejected_proposal"])
        self.assertGreater(original["input_characters"], 120000)
        scoped = scoped_rejected_tree(previous, rejected, capture["repair_input"]["source"])
        self.assertEqual(scoped["removed_nodes"], ["h_build"])
        old = {node.id: node for node in previous.nodes}
        reconstructed = {key: node for key, node in old.items() if key not in scoped["removed_nodes"]}
        for value in scoped["changed_nodes"]:
            value = copy.deepcopy(value)
            for field in ("criteria", "assumptions"):
                for index, row in enumerate(value.get(field, [])):
                    if "from_node" in row:
                        source = old[row["from_node"]]
                        condition = next(c for c in source.criteria + source.assumptions if c.id == row["condition_id"])
                        value[field][index] = condition.model_dump(mode="json")
            node = Node.model_validate(value)
            reconstructed[node.id] = node
        self.assertEqual({key: node_digest(node) for key, node in reconstructed.items()},
                         {node.id: node_digest(node) for node in rejected.nodes})
        payload["rejected_proposal"] = scoped
        characters = len(original["messages"][0]["content"]) + len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        self.assertLess(characters, original["input_characters"])
        with TemporaryDirectory() as directory:
            store = TreeStore(Path(directory))
            store.revise(previous)
            with self.assertRaisesRegex(ValueError, "remove stable nodes: h_build"):
                store.revise(rejected)

    def test_original_invalid_m3_json_stays_raw_and_supplies_the_actual_error_location(self):
        from workflows.minecraft_tasks.schemas import TaskTree
        from workflows.minecraft_tasks.steps import scoped_rejection
        capture = json.loads((CAPTURE.parent / "a13-planning-budget.json").read_text(encoding="utf-8"))
        payload = json.loads(capture["repair_input"]["original_input"]["messages"][1]["content"])
        raw = capture["invalid_raw"]
        with self.assertRaises(json.JSONDecodeError) as failure:
            json.loads(raw["document"]["text"])
        error = failure.exception
        scoped = scoped_rejection(TaskTree.model_validate(payload["previous_tree"]), raw["document"], raw["source"])
        self.assertEqual(scoped["kind"], "invalid_model_output")
        self.assertEqual(scoped["error"], str(error))
        self.assertLessEqual(len(scoped["excerpt"]), 2000)
        self.assertLessEqual(scoped["excerpt_start"], error.pos)
        self.assertGreater(scoped["excerpt_start"] + len(scoped["excerpt"]), error.pos)
        self.assertEqual(scoped["excerpt"], raw["document"]["text"][scoped["excerpt_start"]:scoped["excerpt_start"] + 2000])
        self.assertNotIn("changed_nodes", scoped)

    def test_original_unrelated_m3_refinement_names_the_affected_node_and_retains_the_tree(self):
        from workflows.minecraft_tasks.schemas import TaskTree
        capture = json.loads((CAPTURE.parent / "a13-planning-budget.json").read_text(encoding="utf-8"))
        payload = json.loads(capture["repair_input"]["original_input"]["messages"][1]["content"])
        with TemporaryDirectory() as directory:
            store = TreeStore(Path(directory))
            store.revise(TaskTree.model_validate(payload["previous_tree"]))
            before = store.load().model_dump(mode="json")
            with self.assertRaisesRegex(ValueError, "unrelated node '(h_entry|home)'"):
                store.revise(TaskTree.model_validate(capture["unrelated_revision"]["tree"]), target="h_build")
            self.assertEqual(store.load().model_dump(mode="json"), before)

    def test_original_recovery_proposal_can_update_explicit_nodes_without_copying_accepted_work(self):
        from workflows.minecraft_tasks.schemas import PlanProposal, TaskTree
        from workflows.minecraft_tasks.steps import apply_node_updates
        from workflows.minecraft_tasks.tree import node_digest
        capture = json.loads((CAPTURE.parent / "a13-planning-budget.json").read_text(encoding="utf-8"))
        payload = json.loads(capture["repair_input"]["original_input"]["messages"][1]["content"])
        previous = TaskTree.model_validate(payload["previous_tree"])
        proposed = TaskTree.model_validate(capture["latest_copy_failure"]["tree"])
        old = {node.id: node for node in previous.nodes}
        updates = [node for node in proposed.nodes if node.id not in old or node_digest(node) != node_digest(old[node.id])]
        contract = PlanProposal(status="planned", summary="Captured explicit node changes.", updates=updates)
        merged = apply_node_updates(previous, contract.updates)
        self.assertEqual({node.id: node_digest(node) for node in merged.nodes},
                         {node.id: node_digest(node) for node in proposed.nodes})
        self.assertEqual(next(node for node in merged.nodes if node.id == "w_gather").model_dump(), old["w_gather"].model_dump())
        with TemporaryDirectory() as directory:
            store = TreeStore(Path(directory))
            store.revise(previous)
            store.revise(merged)
            self.assertIn("recovery", {node.id for node in store.load().nodes})

    def test_original_active_recovery_supersedes_optional_refinement_restrictions(self):
        from workflows.minecraft_tasks.schemas import TaskTree
        from workflows.minecraft_tasks.steps import refinement_scope
        capture = json.loads((CAPTURE.parent / "a13-planning-budget.json").read_text(encoding="utf-8"))
        payload = json.loads(capture["latest_copy_failure"]["input"]["messages"][1]["content"])
        self.assertTrue(payload["gameplay_history"]["recovery"])
        with TemporaryDirectory() as directory:
            store = TreeStore(Path(directory))
            store.revise(TaskTree.model_validate(payload["previous_tree"]))
            store.state.update(instruction_revision=1, refinement_target="h_build")
            self.assertEqual(refinement_scope(store, 1, payload["gameplay_history"]), "")
            self.assertEqual(refinement_scope(store, 1, {"recovery": None}), "h_build")
            self.assertEqual(refinement_scope(store, 2, {"recovery": None}), "")

    def test_node_update_contract_cannot_combine_full_tree_duplicate_ids_or_a_survey(self):
        from pydantic import ValidationError
        from workflows.minecraft_tasks.schemas import PlanProposal, TaskTree
        capture = json.loads((CAPTURE.parent / "a13-planning-budget.json").read_text(encoding="utf-8"))
        tree = TaskTree.model_validate(capture["latest_copy_failure"]["tree"])
        node = next(node for node in tree.nodes if node.id == "recovery")
        for arguments in ({"tree": tree, "updates": [node]}, {"updates": [node, node]},
                          {"status": "survey", "updates": [node]}):
            with self.subTest(arguments=list(arguments)):
                with self.assertRaises(ValidationError):
                    PlanProposal.model_validate({"status": "planned", "summary": "Reject ambiguous updates.", **arguments})

    def test_original_blocked_recovery_review_survives_without_accepting_the_unsafe_leaf(self):
        from workflows.minecraft_tasks.schemas import PlanReview, TaskTree
        from workflows.minecraft_tasks.steps import record_review
        capture = json.loads((CAPTURE.parent / "a13-planning-budget.json").read_text(encoding="utf-8"))["blocked_review"]
        result = PlanReview.model_validate(capture["result"])
        self.assertEqual(result.status, "blocked")
        self.assertIn("crafting_table", result.summary)
        with TemporaryDirectory() as directory:
            store = TreeStore(Path(directory))
            store.revise(TaskTree.model_validate(capture["tree"]))
            store.state.update(feedback="Older JSON error.", refinement_target="h_build")
            record_review(store, result)
            restored = TreeStore(Path(directory))
            self.assertEqual(restored.state["feedback"], result.summary)
            self.assertEqual(restored.state["refinement_target"], "")
            self.assertNotIn("reviewed_revision", restored.state)
            self.assertFalse(restored.state["acceptances"])

    def test_original_ready_recovery_preserves_the_optional_refinement_boundary(self):
        from workflows.minecraft_tasks.schemas import TaskTree
        from workflows.minecraft_tasks.steps import refinement_scope
        capture = json.loads((CAPTURE.parent / "a13-planning-budget.json").read_text(encoding="utf-8"))["ready_recovery_refinement"]
        history = json.loads(capture["input"]["messages"][1]["content"])["gameplay_history"]
        self.assertTrue(history["recovery"])
        self.assertEqual(capture["review"]["target_id"], "h_build")
        with TemporaryDirectory() as directory:
            store = TreeStore(Path(directory))
            store.revise(TaskTree.model_validate(capture["tree"]))
            store.state.update(instruction_revision=1, refinement_target="h_build")
            self.assertIn("recover", store.eligible())
            self.assertEqual(refinement_scope(store, 1, history), "h_build")

    def test_original_missing_target_update_cannot_satisfy_the_explicit_refinement_contract(self):
        from pydantic import ValidationError
        from workflows.minecraft_tasks.schemas import RefinementProposal
        capture = json.loads((CAPTURE.parent / "a13-planning-budget.json").read_text(encoding="utf-8"))["missing_target_update"]
        self.assertNotIn("h_build", {node["id"] for node in capture["proposal"]["updates"]})
        with self.assertRaisesRegex(ValidationError, "replacement"):
            RefinementProposal.model_validate(capture["proposal"])

    def test_explicit_refinement_cannot_replace_the_wrong_existing_node(self):
        from workflows.minecraft_tasks.schemas import RefinementProposal, TaskTree
        from workflows.minecraft_tasks.steps import refinement_plan
        capture = json.loads((CAPTURE.parent / "a13-planning-budget.json").read_text(encoding="utf-8"))
        previous = TaskTree.model_validate(capture["ready_recovery_refinement"]["tree"])
        full = TaskTree.model_validate(capture["missing_target_update"]["full_split"]["tree"])
        home = next(node for node in full.nodes if node.id == "home")
        target = next(node for node in full.nodes if node.id == "h_build")
        with self.assertRaisesRegex(ValueError, "retain target ID 'h_build'"):
            refinement_plan(previous, "h_build", RefinementProposal(status="planned", summary="Wrong target.", replacement=home))
        with self.assertRaisesRegex(ValueError, "unrelated nodes: home"):
            refinement_plan(previous, "h_build", RefinementProposal(status="planned", summary="Unrelated replacement.", replacement=target, additions=[home]))

    def test_original_explicit_split_still_cannot_weaken_house_world_predicates(self):
        from workflows.minecraft_tasks.schemas import RefinementProposal, TaskTree
        from workflows.minecraft_tasks.steps import apply_node_updates, refinement_plan
        capture = json.loads((CAPTURE.parent / "a13-planning-budget.json").read_text(encoding="utf-8"))
        previous = TaskTree.model_validate(capture["ready_recovery_refinement"]["tree"])
        full = TaskTree.model_validate(capture["missing_target_update"]["full_split"]["tree"])
        ids = {node.id for node in previous.nodes}
        replacement = next(node for node in full.nodes if node.id == "h_build")
        proposal = RefinementProposal(status="planned", summary="Original captured split.", replacement=replacement,
                                      additions=[node for node in full.nodes if node.id not in ids])
        planned = refinement_plan(previous, "h_build", proposal)
        merged = apply_node_updates(previous, planned.updates)
        with TemporaryDirectory() as directory:
            store = TreeStore(Path(directory))
            store.revise(previous)
            with self.assertRaisesRegex(ValueError, "preserve the selected acceptance conditions"):
                store.revise(merged, target="h_build")

    def test_original_partial_build_retains_three_cells_without_completion(self):
        result = classify(self.capture["partial"]["receipt"])
        self.assertEqual(result["kind"], "partial_progress")
        self.assertTrue(result["unsuccessful"])
        self.assertEqual(result["quantities"]["placed_count"], 3)
        self.assertFalse(result["completion_verified"])

    def test_original_pillar_quantities_establish_progress_and_zero_output(self):
        capture = json.loads((CAPTURE.parent / "a13-pillar-and-fall.json").read_text(encoding="utf-8"))
        for row in capture["pillars"]:
            with self.subTest(seq=row["seq"]):
                result = classify(row)
                self.assertEqual(result["quantities"]["blocks_placed"], row["result"]["blocks_placed"])
                self.assertEqual(result["quantities"]["climbed"], row["result"]["climbed"])
                self.assertEqual(result["made_progress"], row["result"]["blocks_placed"] > 0)
                self.assertFalse(result["completion_verified"])

    def test_original_negative_coordinates_supply_exact_body_cells(self):
        capture = json.loads((CAPTURE.parent / "a13-pillar-and-fall.json").read_text(encoding="utf-8"))
        evidence = GameplayEvidence()
        for row in capture["observations"]:
            evidence.consume(row, capture["source"])
        cells = evidence.recovery_guidance()["current_body_cells"]
        self.assertEqual(cells["floor"], {"x": 9, "y": 112, "z": -3})
        self.assertEqual(cells["feet"], {"x": 9, "y": 113, "z": -3})
        self.assertEqual(cells["head"], {"x": 9, "y": 114, "z": -3})

    def test_original_lava_edge_requires_retreat_despite_full_air_and_health(self):
        capture = json.loads((CAPTURE.parent / "a13-lava-edge-escape.json").read_text(encoding="utf-8"))
        evidence = GameplayEvidence()
        for row in capture["history_rows"]:
            evidence.consume(self.with_captured_life(row), capture["history_source"])
        self.assertEqual(evidence.recovery["lava_cells"], [{"x": -9, "y": 116, "z": 0}])
        self.assertEqual(evidence.observation["player"]["health"], 20)
        self.assertEqual(evidence.observation["player"]["oxygen"], 20)
        self.assertEqual(evidence.guard("minecraft_craft_item", {"item": "iron_pickaxe"})["reason"], "recovery_required")
        for row in capture["rows"]:
            evidence.consume(self.with_captured_life(row), capture["source"])
            if row.get("event") == "result" and row["seq"] == 2:
                self.assertGreaterEqual(evidence.stable_observations, 2)
                self.assertIsNotNone(evidence.recovery)
        self.assertIsNone(evidence.recovery)
        self.assertEqual(evidence.observation["camera"]["block_position"], {"x": -10, "y": 117, "z": 1})

    def test_positive_partial_output_does_not_count_as_no_progress(self):
        entry = self.capture["partial"]
        evidence = GameplayEvidence()
        for _ in range(2):
            evidence.consume(entry["receipt"], entry["source"])
        self.assertEqual(next(iter(evidence.methods.values()))["ineffective"], 0)
        self.assertIsNone(evidence.guard(entry["receipt"]["tool"], entry["receipt"]["arguments"]))
        self.assertTrue(evidence.outcomes[-1]["unsuccessful"])
        self.assertTrue(evidence.outcomes[-1]["made_progress"])

    def test_original_game_failures_do_not_become_transport_failures(self):
        for name in ("navigation", "reach", "supplies"):
            with self.subTest(name=name):
                self.assertEqual(classify(self.capture[name]["receipt"])["kind"], "unsuccessful_action")

    def test_local_negative_search_retains_bounds_and_unknown_buried_cells(self):
        entry = self.capture["search"]
        evidence = GameplayEvidence()
        evidence.consume(entry["observation"], entry["source"])
        evidence.consume(entry["receipt"], entry["source"])
        search = evidence.snapshot()["searches"][0]
        self.assertEqual(search["coverage"], entry["receipt"]["result"]["coverage"])
        self.assertEqual(search["dimension"], "overworld")
        self.assertEqual(search["observed_at"], entry["receipt"]["time"])
        self.assertEqual(search["total_matches"], 0)
        self.assertTrue(search["available"])
        self.assertIn("buried", search["uncertainty"])

    def test_same_failed_method_stays_paused_after_reconstruction(self):
        entry = self.capture["navigation"]
        evidence = GameplayEvidence()
        for _ in range(2):
            evidence.consume(entry["receipt"], entry["source"])
        rejected = evidence.guard(entry["receipt"]["tool"], entry["receipt"]["arguments"])
        self.assertEqual(rejected["reason"], "method_paused")
        self.assertEqual(len(evidence.outcomes), 2)
        self.assertIsNone(evidence.guard("minecraft_inspect_block", {"position": {"x": 0, "y": 0, "z": 0}}))

    def test_explanation_without_a_new_read_cannot_unlock_a_method(self):
        entry = self.capture["navigation"]
        evidence = GameplayEvidence()
        evidence.consume(entry["receipt"], entry["source"])
        evidence.consume(entry["receipt"], entry["source"])
        with self.assertRaisesRegex(ValueError, "fresh usable reads"):
            evidence.diagnosis("The target needs support.", [entry["observation"]["seq"]],
                               entry["receipt"]["tool"], entry["receipt"]["arguments"], False)

    def test_unchanged_tool_and_supply_state_cannot_unlock_failed_method(self):
        entry = self.capture["navigation"]
        evidence = GameplayEvidence()
        evidence.consume(entry["observation"], entry["source"])
        evidence.consume(entry["receipt"], entry["source"])
        evidence.consume(entry["receipt"], entry["source"])
        fresh = {**entry["observation"], "seq": 500, "time": entry["receipt"]["time"] + 1}
        evidence.consume(fresh, entry["source"])
        with self.assertRaisesRegex(ValueError, "remain paused"):
            evidence.diagnosis("A new explanation.", [500], entry["receipt"]["tool"], entry["receipt"]["arguments"], False)

    def test_older_planning_observation_cannot_replace_current_body_state(self):
        entry = self.capture["search"]
        evidence = GameplayEvidence()
        evidence.consume({**entry["observation"], "time": 200}, entry["source"])
        current = copy.deepcopy(evidence.observation)
        older = copy.deepcopy(entry["observation"])
        older["time"] = 100
        older["result"]["player"]["health"] = 0
        evidence.consume(older, "old-planning-file")
        self.assertEqual(evidence.observation, current)
        self.assertIsNone(evidence.recovery)

    def test_deferral_retains_stable_leaf_and_never_accepts_it(self):
        from tests.test_a11_minecraft_tasks import tree
        with TemporaryDirectory() as directory:
            store = TreeStore(Path(directory))
            original = tree()
            store.revise(original)
            revised = original.model_copy(deep=True)
            leaf = next(node for node in revised.nodes if node.kind == "leaf")
            leaf.deferred, leaf.defer_reason = True, "The observed local route lacks support."
            store.revise(revised)
            self.assertNotIn(leaf.id, store.eligible())
            self.assertFalse(store.valid(leaf.id))
            self.assertEqual(store.load().nodes[1].id, original.nodes[1].id)

    def test_deferral_requires_a_leaf_and_reason(self):
        with self.assertRaisesRegex(ValueError, "explicit reason"):
            Node(id="leaf", kind="leaf", instruction="Keep the route.", deferred=True,
                 criteria=[{"id": "alive", "description": "Living body.", "kind": "alive"}])

    def test_milestone_commits_once_and_preserves_publication_identity(self):
        from runtime.managed_runs import ManagedRunReceipt, ManagedRunSupervisor
        from tests.test_design_contracts import small_ghost
        with TemporaryDirectory() as directory:
            events = []
            supervisor = ManagedRunSupervisor(small_ghost(), runtime_root=Path(directory),
                                              workflows={}, event_sink=events.append, clock=lambda: 1)
            supervisor._persist(ManagedRunReceipt("run", "action", "minecraft_project", "exchange", "running",
                                                 "working", str(Path(directory) / "task")))
            supervisor.milestone("run", 1, "actor-0001", "Three wall cells remain retained.", "attempts/actor-0001/scratch/actor-journal.jsonl")
            supervisor.milestone("run", 1, "actor-0001", "Duplicate publication.", "same-journal")
            self.assertEqual(len(events), 1)
            original = events[0]["event_id"]
            supervisor.replay_events()
            self.assertEqual(events[-1]["event_id"], original)
            supervisor.acknowledge_event("run", original)
            supervisor.replay_events()
            self.assertEqual(len(events), 2)
            with self.assertRaisesRegex(Exception, "no longer owns"):
                supervisor.milestone("run", 2, "actor-0002", "Stale progress.", "other-journal")

    def test_milestone_sink_can_inspect_receipt_from_another_thread(self):
        from runtime.managed_runs import ManagedRunReceipt, ManagedRunSupervisor
        from tests.test_design_contracts import small_ghost
        with TemporaryDirectory() as directory:
            supervisor = ManagedRunSupervisor(small_ghost(), runtime_root=Path(directory), workflows={}, clock=lambda: 1)
            supervisor._persist(ManagedRunReceipt("run", "action", "minecraft_project", "exchange", "running",
                                                 "working", str(Path(directory) / "task")))
            completed = threading.Event()
            def sink(message):
                reader = threading.Thread(target=lambda: (supervisor.status("run"), completed.set()))
                reader.start()
                reader.join(1)
                self.assertTrue(completed.is_set())
            supervisor.event_sink = sink
            supervisor.milestone("run", 1, "actor-0001", "Observed progress.", "original-journal")

    def test_original_blocked_path_uses_state_field(self):
        capture = json.loads((CAPTURE.parent / "a13-water-and-path.json").read_text(encoding="utf-8"))
        outcome = classify(capture["path"])
        self.assertEqual(outcome["kind"], "unsuccessful_action")
        self.assertEqual(outcome["reason"], "blocked")

    def test_original_wrong_tool_cannot_invent_mining_progress(self):
        capture = json.loads((CAPTURE.parent / "a13-water-and-path.json").read_text(encoding="utf-8"))
        receipt = {"tool": "minecraft_mine_block", "ok": capture["wrong_tool"]["ok"], "result": capture["wrong_tool"]["result"]}
        outcome = classify(receipt)
        self.assertEqual(outcome["kind"], "unsuccessful_action")
        self.assertTrue(outcome["unsuccessful"])
        self.assertEqual(outcome["reason"], "wrong_harvesting_tool")

    def test_original_low_air_keeps_recovery_active_across_epoch_change(self):
        capture = json.loads((CAPTURE.parent / "a13-water-and-path.json").read_text(encoding="utf-8"))
        evidence = GameplayEvidence()
        evidence.consume(capture["water"], capture["sources"]["water"])
        self.assertIn("water_air", evidence.recovery["causes"])
        evidence.consume({"event": "epoch_acquired", "time": capture["water"]["time"] + 1}, "replacement-actor")
        self.assertIsNotNone(evidence.recovery)
        self.assertIsNone(evidence.permit)
        self.assertEqual(evidence.guard("minecraft_craft_item", {"item": "iron_pickaxe"})["reason"], "recovery_required")

    def test_original_shallow_water_requires_dry_recovery_despite_full_air(self):
        capture = json.loads((CAPTURE.parent / "a13-shallow-water.json").read_text(encoding="utf-8"))
        evidence = GameplayEvidence()
        evidence.consume(capture["observation"], capture["source"])
        self.assertTrue(evidence.observation["player"]["in_water"])
        self.assertEqual(evidence.observation["player"]["oxygen"], 20)
        self.assertEqual(evidence.recovery["causes"], ["water"])
        self.assertEqual(evidence.guard("minecraft_craft_item", {"item": "iron_pickaxe"})["reason"], "recovery_required")
        self.assertIsNone(evidence.guard("minecraft_observe", {"include_image": False}))

    def test_original_prefixed_diagnosis_binds_the_actual_paused_dispatch(self):
        from runtime.minecraft_actor import canonical_actor_tool
        capture = json.loads((CAPTURE.parent / "a13-prefixed-diagnosis.json").read_text(encoding="utf-8"))
        evidence = GameplayEvidence()
        for row in capture["rows"]:
            evidence.consume(row, capture["source"])
        original = capture["diagnosis"]
        name = canonical_actor_tool(original["next_tool"], set(capture["allowed_tools"]))
        self.assertEqual(evidence.guard(name, original["next_arguments"])["reason"], "method_paused")
        record = evidence.diagnosis(original["diagnosis"], original["evidence_sequences"], name,
                                    original["next_arguments"], original["recovery_action"])
        evidence.consume({**record, "time": original["time"]}, capture["source"])
        self.assertEqual(record["next_tool"], "minecraft_craft_item")
        self.assertIsNone(evidence.guard(name, original["next_arguments"]))

    def test_diagnosis_cannot_bind_unknown_or_excluded_actor_tools(self):
        from runtime.minecraft_actor import canonical_actor_tool
        capture = json.loads((CAPTURE.parent / "a13-prefixed-diagnosis.json").read_text(encoding="utf-8"))
        for name in ("other_actor_minecraft_craft_item", "minecraft_actor_actor_diagnose",
                     "minecraft_actor_minecraft_execute_typescript", "minecraft_suicide"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Unknown actor action"):
                canonical_actor_tool(name, set(capture["allowed_tools"]))

    def test_original_failed_escape_requires_dry_destination_body_space(self):
        capture = json.loads((CAPTURE.parent / "a13-failed-water-escape.json").read_text(encoding="utf-8"))
        evidence = GameplayEvidence()
        for source, rows in ((capture["history_source"], capture["history_rows"]),
                             (capture["source"], capture["rows"])):
            for row in rows:
                evidence.consume(row, source)
        self.assertIsNotNone(evidence.recovery)
        for height in (102, 103):
            with self.subTest(height=height), self.assertRaisesRegex(ValueError, "dry feet/head"):
                evidence.diagnosis("Approach the inspected bank.", [13], "minecraft_walk_to_exact",
                                   {"x": -2, "y": height, "z": -27}, True)
        self.assertIsNone(evidence.permit)

    def test_original_failure_history_stays_on_disk_outside_repeated_guidance(self):
        capture = json.loads((CAPTURE.parent / "a13-failed-water-escape.json").read_text(encoding="utf-8"))
        evidence = GameplayEvidence()
        for row in capture["history_rows"]:
            evidence.consume(row, capture["history_source"])
        rejected = evidence.guard("minecraft_craft_item", {"item": "iron_pickaxe", "repetitions": 1})
        self.assertEqual(rejected["reason"], "recovery_required")
        self.assertLess(len(json.dumps(rejected)), 6000)
        self.assertGreater(len(json.dumps(evidence.snapshot())), len(json.dumps(rejected)))
        self.assertTrue(evidence.snapshot()["methods"][0]["state"])
        self.assertEqual(rejected["evidence"], "scratch/decision-evidence.json")
        self.assertEqual(rejected["full_history"], "scratch/gameplay-evidence.json")

    def test_original_escape_accepts_dry_grass_and_exact_neighbor_geometry(self):
        capture = json.loads((CAPTURE.parent / "a13-dry-grass-escape.json").read_text(encoding="utf-8"))
        evidence = GameplayEvidence()
        for row in capture["rows"]:
            evidence.consume(self.with_captured_life(row), capture["source"])
            if row.get("seq") == 6:
                self.assertIsNotNone(evidence.recovery)
        self.assertEqual(evidence.observation["player"]["oxygen"], 20)
        self.assertGreaterEqual(evidence.stable_observations, 2)
        self.assertIsNone(evidence.recovery)

    def test_missing_geometry_and_liquid_cannot_establish_safe_landing(self):
        from runtime.minecraft_outcomes import safe_landing
        self.assertFalse(safe_landing({"is_solid": True, "block_name": "stone"}, {}, {}))
        self.assertFalse(safe_landing({"is_solid": True, "block_name": "stone"},
                                      {"is_solid": False, "block_name": "water"},
                                      {"is_solid": False, "block_name": "air"}))

    def test_typed_service_failure_waits_without_pausing_observation(self):
        evidence = GameplayEvidence()
        row = {"event": "result", "tool": "minecraft_observe", "arguments": {}, "seq": 1,
               "time": 1, "ok": False, "result": {"reason": "not_connected"}}
        for _ in range(3):
            evidence.consume(row, "service-journal")
        self.assertEqual(evidence.outcomes[-1]["kind"], "transport_failure")
        self.assertIsNotNone(evidence.service_wait)
        self.assertIsNone(evidence.guard("minecraft_observe", {}))
        self.assertEqual(evidence.guard("minecraft_mine_block", {})["reason"], "service_wait")
        observed = {**self.capture["search"]["observation"], "time": 2}
        evidence.consume(observed, "service-journal")
        self.assertIsNone(evidence.service_wait)

    def test_exact_block_contract_rejects_inventory_globs(self):
        from workflows.minecraft_tasks.schemas import Condition
        with self.assertRaisesRegex(ValueError, "exact block names"):
            Condition(id="floor", description="A known floor cell.", kind="block", names=["*_planks"],
                      position={"x": 0, "y": 0, "z": 0})
        condition = Condition(id="stock", description="Retained wood stock.", kind="inventory", names=["*_planks"])
        self.assertEqual(condition.names, ["*_planks"])


if __name__ == "__main__":
    unittest.main()
