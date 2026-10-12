"""Topology and routing acceptance for every real application manifest.

Extends ``pm_workflows.dryrun`` instead of adding a second graph engine. Every
workflow folder is discovered from disk. Each manifest is checked for missing
files, unknown destinations, reachable terminal success, bounded cycles, and
declared waiting exits. Seeded simulations are deterministic and stay inside
the static worst-case ceiling.

The generic bus and runtime must contain no workflow-name special case.
"""
from __future__ import annotations

import unittest
from pathlib import Path

from pm_workflows.dryrun import (
    DISPATCH_KINDS,
    TERMINAL_STOP,
    TERMINAL_SUCCESS,
    check_workflow,
)
from pm.runs.workflow_module import PROJECT_ROOT

WORKFLOW_ROOT = PROJECT_ROOT / "workflows"
TERMINALS = {TERMINAL_SUCCESS, TERMINAL_STOP}

# A waiting phase is bounded by the ambient default, not by a real work cap.
# Its exit is the correlated answer plus cancellation; that is declared via the
# human phase's ``next`` and the kernel's cancellation path, so the warning is
# accepted rather than blocking.
KNOWN_WAITING_WARNINGS = {
    ("local_task_question", "ask"),
}

# A project revisits these phases for dynamic leaves and observed replanning.
# Keep the kernel's existing backstop. It does not cap native actor tool calls.
KNOWN_PROJECT_REPEAT_PHASES = {"plan", "survey", "review", "select", "act", "verify", "replan", "wait", "finish"}


def manifests() -> list[Path]:
    return sorted(WORKFLOW_ROOT.rglob("*.workflow.md"))


class TopologyTests(unittest.TestCase):
    def test_every_manifest_is_discovered(self):
        found = {path.name for path in manifests()}
        self.assertIn("main_cognition.workflow.md", found)
        self.assertIn("dialogue_story.workflow.md", found)
        self.assertIn("dialogue_assistant.workflow.md", found)
        self.assertIn("self_management.workflow.md", found)
        self.assertIn("steward.workflow.md", found)
        self.assertIn("memory_consolidation.workflow.md", found)
        self.assertIn("local_task.workflow.md", found)

    def test_every_manifest_has_no_blocking_error(self):
        for path in manifests():
            with self.subTest(manifest=str(path)):
                report = check_workflow(
                    path, PROJECT_ROOT, monte_carlo_runs=200, seed=7,
                )
                self.assertEqual(report.errors, [], f"{path} has blocking errors")

    def test_every_edge_destination_is_declared(self):
        for path in manifests():
            report = check_workflow(path, PROJECT_ROOT, skip_monte_carlo=True)
            with self.subTest(manifest=str(path)):
                for node in report.nodes.values():
                    for edge in node.edges:
                        if edge.target in TERMINALS:
                            continue
                        self.assertIn(
                            edge.target, report.nodes,
                            f"{path}: {node.name} -> unknown {edge.target}",
                        )

    def test_every_manifest_can_reach_a_terminal(self):
        from collections import deque
        for path in manifests():
            report = check_workflow(path, PROJECT_ROOT, skip_monte_carlo=True)
            with self.subTest(manifest=str(path)):
                start = next(iter(report.nodes))
                seen = {start}
                queue = deque([start])
                terminal_edges = []
                while queue:
                    name = queue.popleft()
                    node = report.nodes[name]
                    for edge in node.edges:
                        if edge.target in TERMINALS:
                            terminal_edges.append((name, edge))
                            continue
                        if edge.target not in seen:
                            seen.add(edge.target)
                            queue.append(edge.target)
                self.assertTrue(
                    terminal_edges,
                    f"{path}: no path from the first phase to a terminal",
                )
                # A declared blocked outcome is an explicit failure terminal.
                # The runner requires done before it can publish a success report.
                for phase, edge in terminal_edges:
                    if edge.target == TERMINAL_STOP:
                        self.assertIn(
                            edge.outcome, {"done", "invalid", "pass", "answered", "blocked"},
                            f"{path}: {phase} reaches <stop> on unexpected outcome {edge.outcome}",
                        )

    def test_every_cycle_is_bounded_and_has_an_exit(self):
        for path in manifests():
            report = check_workflow(path, PROJECT_ROOT, skip_monte_carlo=True)
            with self.subTest(manifest=str(path)):
                for cycle in report.cycles:
                    for phase, bound in cycle.bound_per_phase.items():
                        self.assertGreaterEqual(bound, 1)
                        self.assertLessEqual(
                            bound, 999,
                            f"{path}: cycle {cycle.phases} has no finite cap",
                        )
                        # Every cyclic phase keeps at least one non-terminal
                        # route, so the cycle can always leave rather than
                        # being a permanent trap.
                        exits = [
                            edge for edge in report.nodes[phase].edges
                            if edge.target not in TERMINALS
                        ]
                        self.assertTrue(
                            exits,
                            f"{path}: cyclic phase {phase} has no exit",
                        )

    def test_waiting_phase_declares_an_exit(self):
        path = WORKFLOW_ROOT / "local_task" / "local_task_question.workflow.md"
        report = check_workflow(path, PROJECT_ROOT, skip_monte_carlo=True)
        human = next(node for node in report.nodes.values() if node.kind == "human")
        targets = {edge.target for edge in human.edges}
        self.assertTrue(targets - TERMINALS, "human phase has no non-terminal exit")

    def test_warnings_are_only_the_accepted_waiting_phase(self):
        for path in manifests():
            report = check_workflow(path, PROJECT_ROOT, skip_monte_carlo=True)
            with self.subTest(manifest=str(path)):
                for warning in report.warnings:
                    accepted = any(
                        path.stem == workflow + ".workflow" and f"phase '{phase}'" in warning
                        for workflow, phase in KNOWN_WAITING_WARNINGS
                    )
                    accepted = accepted or (path.name == "minecraft_project.workflow.md" and any(
                        warning.startswith(f"phase '{phase}' can be dispatched up to 999 times (per task)")
                        for phase in KNOWN_PROJECT_REPEAT_PHASES))
                    self.assertTrue(accepted, f"{path}: unexpected warning: {warning}")

    def test_seeded_simulation_is_deterministic_and_bounded(self):
        path = WORKFLOW_ROOT / "main_cognition" / "main_cognition.workflow.md"
        first = check_workflow(path, PROJECT_ROOT, monte_carlo_runs=500, seed=11)
        second = check_workflow(path, PROJECT_ROOT, monte_carlo_runs=500, seed=11)
        self.assertEqual(first.monte_carlo.summary(), second.monte_carlo.summary())
        # Every sampled dispatch count is finite and non-negative.
        for count in first.monte_carlo.dispatch_counts:
            self.assertGreaterEqual(count, 0)
        # Sampled outcomes are a subset of the declared simulator outcomes.
        self.assertLessEqual(
            set(first.monte_carlo.outcomes),
            {"success", "stopped", "exhausted"},
        )
        # The optimistic path is never longer than the simulated median.
        best = first.best["zero_items"].dispatches
        self.assertLessEqual(best, first.monte_carlo.summary()["dispatches_median"])

    def test_generic_bus_and_runtime_have_no_workflow_special_case(self):
        names = [
            "dialogue_story", "dialogue_assistant", "self_management",
            "steward", "memory_consolidation", "local_task",
        ]
        generic = [
            PROJECT_ROOT / "pm" / "bus" / "bus.py",
            PROJECT_ROOT / "pm" / "bus" / "envelopes.py",
            PROJECT_ROOT / "pm" / "host" / "application.py",
            PROJECT_ROOT / "pm" / "runs" / "workflow_runner.py",
        ]
        for path in generic:
            text = path.read_text(encoding="utf-8")
            for name in names:
                self.assertNotIn(
                    name, text,
                    f"{path.name} contains a workflow-name special case for {name!r}",
                )


if __name__ == "__main__":
    unittest.main()
