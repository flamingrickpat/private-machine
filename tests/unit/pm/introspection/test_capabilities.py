"""Capability overlay contracts.

The overlay keeps intended, implemented, enabled, and observed evidence
apart. A tool name alone never becomes an observed capability.
"""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pm.introspection.capabilities import (
    CapabilityContext,
    CapabilityDeclaration,
    CapabilityState,
    build_overlay,
    default_capability_declarations,
)
from pm.introspection.grants import SourceGrant, SourceInspector


def _inspector(tmp: str) -> SourceInspector:
    root = Path(tmp) / "repo"
    (root / "workflows").mkdir(parents=True, exist_ok=True)
    (root / "docs").mkdir(exist_ok=True)
    (root / "workflows" / "feature.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    (root / "docs" / "feature.md").write_text("# Feature\n", encoding="utf-8")
    grant = SourceGrant("test", roots=(root,))
    return SourceInspector(grant, repo_revision="rev1", configuration_revision="cfg1")


class OverlayStateTests(unittest.TestCase):
    def test_states_remain_distinct(self):
        with TemporaryDirectory() as tmp:
            inspector = _inspector(tmp)
            declaration = CapabilityDeclaration(
                name="feature", summary="A feature.",
                intended_refs=("docs/feature.md",),
                implemented_refs=("workflows/feature.py",),
            )
            overlay = build_overlay([declaration], CapabilityContext(inspector=inspector))
            finding = overlay.findings["feature"]
            self.assertTrue(finding.has(CapabilityState.Intended))
            self.assertTrue(finding.has(CapabilityState.Implemented))
            self.assertFalse(finding.has(CapabilityState.Enabled))
            self.assertFalse(finding.has(CapabilityState.Observed))
            self.assertEqual(finding.strongest, CapabilityState.Implemented)

    def test_enabled_without_observed_execution_stays_enabled(self):
        with TemporaryDirectory() as tmp:
            inspector = _inspector(tmp)
            declaration = CapabilityDeclaration(
                name="screenshot", summary="A schema exists.",
                implemented_refs=("workflows/feature.py",),
            )
            overlay = build_overlay([declaration], CapabilityContext(
                inspector=inspector, enabled_names=frozenset({"screenshot"})))
            finding = overlay.findings["screenshot"]
            self.assertTrue(finding.has(CapabilityState.Enabled))
            self.assertFalse(finding.has(CapabilityState.Observed))
            self.assertEqual(finding.strongest, CapabilityState.Enabled)

    def test_a_receipt_raises_the_finding_to_observed(self):
        with TemporaryDirectory() as tmp:
            inspector = _inspector(tmp)

            def observed(context):
                return list(context.observed_receipts.get("feature", []))

            declaration = CapabilityDeclaration(
                name="feature", summary="A feature.",
                implemented_refs=("workflows/feature.py",), observed=observed)
            context = CapabilityContext(
                inspector=inspector, enabled_names=frozenset({"feature"}),
                observed_receipts={"feature": [{"reference": "run:1", "detail": "returned an image",
                                                "revision": "r9"}]},
            )
            finding = build_overlay([declaration], context).findings["feature"]
            self.assertEqual(finding.strongest, CapabilityState.Observed)
            self.assertEqual(finding.evidence["observed"][0]["reference"], "run:1")

    def test_missing_source_becomes_an_explicit_limit(self):
        with TemporaryDirectory() as tmp:
            inspector = _inspector(tmp)
            declaration = CapabilityDeclaration(
                name="ghost", summary="Not built yet.",
                implemented_refs=("workflows/missing.py",))
            finding = build_overlay([declaration], CapabilityContext(inspector=inspector)).findings["ghost"]
            self.assertEqual(finding.strongest, CapabilityState.Unavailable)
            self.assertTrue(finding.has(CapabilityState.Unavailable))

    def test_no_evidence_is_uncertain(self):
        with TemporaryDirectory() as tmp:
            inspector = _inspector(tmp)
            declaration = CapabilityDeclaration(name="unknown", summary="No evidence.")
            finding = build_overlay([declaration], CapabilityContext(inspector=inspector)).findings["unknown"]
            self.assertEqual(finding.strongest, CapabilityState.Uncertain)


class StalenessTests(unittest.TestCase):
    def test_changed_source_revision_marks_the_finding_stale(self):
        with TemporaryDirectory() as tmp:
            inspector = _inspector(tmp)
            declaration = CapabilityDeclaration(name="feature", summary="A feature.",
                                                implemented_refs=("workflows/feature.py",))
            context = CapabilityContext(inspector=inspector)
            first = build_overlay([declaration], context)
            (Path(tmp) / "repo" / "workflows" / "feature.py").write_text(
                "def run():\n    return 2\n", encoding="utf-8")
            second = first.refresh(context)
            self.assertEqual(second.stale_names(), ["feature"])
            self.assertNotEqual(first.findings["feature"].revision, second.findings["feature"].revision)


class DefaultDeclarationTests(unittest.TestCase):
    def test_every_declared_reference_exists_in_repository_or_is_reported(self):
        declarations = default_capability_declarations()
        names = {item.name for item in declarations}
        self.assertIn("source_inspection", names)
        self.assertIn("private_thought", names)
        repo_root = Path(__file__).resolve().parents[4]
        inspector = SourceInspector(SourceGrant("repo", roots=(repo_root,)))
        overlay = build_overlay(declarations, CapabilityContext(
            inspector=inspector, enabled_names=frozenset(names)))
        for name, finding in overlay.findings.items():
            # The declarations must not claim an implementation they cannot show.
            self.assertTrue(finding.evidence["implemented"] or finding.evidence["unavailable"],
                            f"{name} declares no real implementation reference")


if __name__ == "__main__":
    unittest.main()
