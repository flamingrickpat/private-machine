from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from dev.tools.readability_lint import ReadabilityCheck


class ReadabilityLintTests(unittest.TestCase):
    def rules(self, source: str) -> set[str]:
        return {finding.rule for finding in ReadabilityCheck("example.py", source).check()}

    def test_direct_typed_code_is_accepted(self) -> None:
        source = "def render(value: str) -> str:\n    return value.strip()\n"
        self.assertEqual(self.rules(source), set())

    def test_variadics_and_forwarding_are_rejected(self) -> None:
        source = "def forward(*args: str, **kwargs: str) -> None:\n    target(*args, **kwargs)\n"
        self.assertEqual(self.rules(source), {"PM001"})

    def test_aliased_and_forward_annotations_are_checked(self) -> None:
        source = "from typing import Any as Loose\ndef f(x: 'Loose') -> object:\n    return x\n"
        self.assertEqual(self.rules(source), {"PM002"})

    def test_dictionary_alias_cannot_escape(self) -> None:
        source = "def f() -> str:\n    data = {'name': 'x'}\n    alias = data\n    consume(alias)\n    return alias\n"
        self.assertEqual(self.rules(source), {"PM005"})

    def test_local_dictionary_is_allowed(self) -> None:
        source = "def f() -> str:\n    data = {'name': 'x'}\n    return data['name']\n"
        self.assertEqual(self.rules(source), set())

    def test_dictionary_inside_returned_collection_is_rejected(self) -> None:
        source = "def f() -> list[Record]:\n    return [{'name': 'x'}]\n"
        self.assertEqual(self.rules(source), {"PM005"})

    def test_function_alias_assignment_is_rejected(self) -> None:
        source = "def step() -> None:\n    pass\nalias = step\nalias()\n"
        self.assertEqual(self.rules(source), {"PM003"})

    def test_invalid_forward_annotation_reports_a_finding(self) -> None:
        source = "def f(x: 'not a type') -> None:\n    pass\n"
        self.assertEqual(self.rules(source), {"PM002"})

    def test_domain_map_requires_explicit_reason(self) -> None:
        self.assertEqual(self.rules("values: dict[RecordId, Score] = {}\n"), {"PM004"})
        source = "values: dict[RecordId, Score] = {}  # pm-lint: allow PM004 -- scores indexed by record ID\n"
        self.assertEqual(self.rules(source), set())

    def test_callback_and_registry_are_rejected(self) -> None:
        source = "def step() -> None:\n    pass\ndef f() -> None:\n    schedule(step)\n    registry['step']()\n"
        self.assertEqual(self.rules(source), {"PM003", "PM008"})

    def test_callable_protocol_and_lambda_are_rejected(self) -> None:
        source = "from collections.abc import Callable as Callback\ndef f(x: Callback[[], str]) -> None:\n    g = lambda: 1\nclass Handler:\n    def __call__(self) -> None:\n        pass\n"
        self.assertEqual(self.rules(source), {"PM003"})

    def test_data_attribute_with_method_name_is_not_a_callback(self) -> None:
        source = "def name() -> str:\n    return 'x'\ndef f(record: Record) -> None:\n    consume(record.name)\n"
        self.assertEqual(self.rules(source), set())

    def test_bound_method_is_a_callback(self) -> None:
        source = "class Service:\n    def step(self) -> None:\n        pass\n    def f(self) -> None:\n        schedule(self.step)\n"
        self.assertEqual(self.rules(source), {"PM003"})

    def test_json_alias_and_model_dump_are_rejected(self) -> None:
        source = "from json import dumps as encode\ndef f(model: Record) -> str:\n    encode(model)\n    model.model_dump()\n    return model.model_dump_json()\n"
        self.assertEqual(self.rules(source), {"PM007"})

    def test_exemption_is_one_rule_on_one_line(self) -> None:
        source = "import json\njson.dumps(1)  # pm-lint: allow PM007 -- filesystem serialization\njson.dumps(2)\n"
        findings = ReadabilityCheck("example.py", source).check()
        self.assertEqual([(item.line, item.rule) for item in findings], [(3, "PM007")])

    def test_exemptions_in_strings_do_not_apply(self) -> None:
        source = "import json\njson.dumps('# pm-lint: allow PM007 -- trick')\n"
        self.assertEqual(self.rules(source), {"PM007"})

    def test_malformed_exemption_and_syntax_fail(self) -> None:
        self.assertEqual(self.rules("x = 1  # pm-lint: allow PM007\n"), {"PM000"})
        self.assertEqual(self.rules("def broken("), {"PM000"})

    def test_cli_exit_status(self) -> None:
        script = Path(__file__).with_name("readability_lint.py")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.py"
            path.write_text("def f(value):\n    return value\n", encoding="utf-8")
            result = subprocess.run([sys.executable, str(script), str(path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn("PM009", result.stdout)
            path.write_text("def f(value: str) -> str:\n    return value\n", encoding="utf-8")
            result = subprocess.run([sys.executable, str(script), str(path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0)

    def test_staged_reads_index_and_preserves_worktree(self) -> None:
        original = Path(__file__).with_name("readability_lint.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "dev" / "tools" / original.name
            script.parent.mkdir(parents=True)
            script.write_bytes(original.read_bytes())
            subprocess.run(["git", "init", "--quiet", str(root)], check=True)
            sample = root / "space name.py"
            bad_source = "def f(value):\n    return value\n"
            good_source = "def f(value: str) -> str:\n    return value\n"
            sample.write_text(bad_source, encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "--", sample.name], check=True)
            sample.write_text(good_source, encoding="utf-8")
            result = subprocess.run([sys.executable, str(script), "--staged"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("space name.py:1:", result.stdout)
            self.assertEqual(sample.read_text(encoding="utf-8"), good_source)
            subprocess.run(["git", "-C", str(root), "add", "--", sample.name], check=True)
            sample.write_text(bad_source, encoding="utf-8")
            result = subprocess.run([sys.executable, str(script), "--staged"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Checked 1 Python files; 0 findings.", result.stdout)
            self.assertEqual(sample.read_text(encoding="utf-8"), bad_source)


if __name__ == "__main__":
    unittest.main()
