from __future__ import annotations

import argparse
import ast
import io
import re
import subprocess
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True, order=True)
class Finding:
    path: str
    line: int
    column: int
    rule: str
    message: str


@dataclass(frozen=True, slots=True)
class Exemption:
    line: int
    rule: str
    reason: str


def qualified_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return qualified_name(node.value) + "." + node.attr
    return ""


class ReadabilityCheck:
    def __init__(self, path: str, source: str) -> None:
        self.path = path
        self.source = source
        self.findings: list[Finding] = []
        self.exemptions: list[Exemption] = []
        self.imports: dict[str, str] = {}  # pm-lint: allow PM004 -- import spellings map to canonical Python symbols
        self.function_names: set[str] = set()

    def report(self, node: ast.AST, rule: str, message: str) -> None:
        assert isinstance(node, (ast.expr, ast.stmt, ast.arg, ast.keyword))
        for exemption in self.exemptions:
            if exemption.line == node.lineno and exemption.rule == rule:
                return
        self.findings.append(Finding(self.path, node.lineno, node.col_offset + 1, rule, message))

    def name(self, node: ast.AST) -> str:
        raw_name = qualified_name(node)
        first, separator, rest = raw_name.partition(".")
        if first in self.imports:
            return self.imports[first] + separator + rest
        return raw_name

    def check_annotation(self, annotation: ast.expr) -> None:
        if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
            try:
                parsed = ast.parse(annotation.value, mode="eval").body
            except SyntaxError:
                self.report(annotation, "PM002", "Forward annotation must be a valid concrete type expression.")
                return
            for child in ast.walk(parsed):
                if isinstance(child, ast.expr):
                    child.lineno = annotation.lineno
                    child.col_offset = annotation.col_offset
            self.check_annotation(parsed)
            return
        for child in ast.walk(annotation):
            if not isinstance(child, (ast.Name, ast.Attribute)):
                continue
            name = self.name(child)
            if name in {"Any", "typing.Any", "object", "builtins.object"}:
                self.report(child, "PM002", "Use a concrete domain type instead of Any/object.")
            if name in {"Callable", "typing.Callable", "collections.abc.Callable"}:
                self.report(child, "PM003", "Pass a concrete owner; call its named method directly.")
            if name in {"dict", "Dict", "typing.Dict", "Mapping", "typing.Mapping", "collections.abc.Mapping", "MutableMapping", "typing.MutableMapping", "collections.abc.MutableMapping"}:
                self.report(child, "PM004", "Mapping contract needs review: use a named record, or explain the domain-ID map.")

    def is_dictionary(self, node: ast.AST, local_dicts: set[str]) -> bool:
        if isinstance(node, (ast.Dict, ast.DictComp)):
            return True
        if isinstance(node, ast.Call) and self.name(node.func) in {"dict", "builtins.dict"}:
            return True
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            for element in node.elts:
                if self.is_dictionary(element, local_dicts):
                    return True
        return isinstance(node, ast.Name) and node.id in local_dicts

    def is_function_value(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id in self.function_names
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            return node.value.id in {"self", "cls"} and node.attr in self.function_names
        return False

    def check_function(self, function: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        arguments = function.args
        if arguments.vararg is not None:
            self.report(arguments.vararg, "PM001", "Replace *args with explicit typed parameters.")
        if arguments.kwarg is not None:
            self.report(arguments.kwarg, "PM001", "Replace **kwargs with explicit typed parameters.")
        parameters = arguments.posonlyargs + arguments.args + arguments.kwonlyargs
        for parameter in parameters:
            if parameter.annotation is None and parameter.arg not in {"self", "cls"}:
                self.report(parameter, "PM009", "Parameter requires an explicit type annotation.")
        if function.returns is None:
            self.report(function, "PM009", "Function requires an explicit return annotation.")
        if function.name == "__call__":
            self.report(function, "PM003", "Callable objects obscure behavior; expose a named method.")
        local_dicts: set[str] = set()
        pending: list[ast.AST] = list(function.body)
        body: list[ast.AST] = []
        while pending:
            node = pending.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            body.append(node)
            pending.extend(ast.iter_child_nodes(node))
        # Fixed point catches aliases even when assignments appear in different branches.
        changed = True
        while changed:
            changed = False
            for node in body:
                targets: list[ast.expr] = []
                value: ast.expr | None = None
                if isinstance(node, ast.Assign):
                    targets = node.targets
                    value = node.value
                elif isinstance(node, ast.AnnAssign):
                    targets = [node.target]
                    value = node.value
                if value is not None and self.is_dictionary(value, local_dicts):
                    for target in targets:
                        if isinstance(target, ast.Name) and target.id not in local_dicts:
                            local_dicts.add(target.id)
                            changed = True
        for node in body:
            if isinstance(node, ast.Return) and node.value is not None:
                if self.is_dictionary(node.value, local_dicts):
                    self.report(node, "PM005", "Anonymous dictionary escapes as a result; return a named record.")
                if self.is_function_value(node.value):
                    self.report(node, "PM003", "Returning a function/method hides the execution path.")
            if isinstance(node, ast.Call):
                values = list(node.args)
                values.extend(keyword.value for keyword in node.keywords)
                for value in values:
                    if self.is_dictionary(value, local_dicts):
                        self.report(value, "PM005", "Anonymous dictionary crosses a call boundary; use a named contract.")

    def check(self) -> list[Finding]:
        try:
            tree = ast.parse(self.source, filename=self.path)
        except SyntaxError as error:
            return [Finding(self.path, error.lineno or 1, error.offset or 1, "PM000", str(error.msg))]
        for token in tokenize.generate_tokens(io.StringIO(self.source).readline):
            if token.type != tokenize.COMMENT or "pm-lint:" not in token.string:
                continue
            match = re.fullmatch(r"#\s*pm-lint: allow (PM00[1-9]) -- (\S.*)", token.string)
            if match is None:
                self.findings.append(Finding(self.path, token.start[0], token.start[1] + 1, "PM000", "Expected: # pm-lint: allow PM00N -- concrete boundary reason"))
            else:
                self.exemptions.append(Exemption(token.start[0], match[1], match[2]))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.imports[alias.asname or alias.name.split(".")[0]] = alias.name if alias.asname else alias.name.split(".")[0]
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    self.imports[alias.asname or alias.name] = (node.module or "") + "." + alias.name
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.function_names.add(node.name)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.check_function(node)
                if node.returns is not None:
                    self.check_annotation(node.returns)
            elif isinstance(node, ast.arg) and node.annotation is not None:
                self.check_annotation(node.annotation)
            elif isinstance(node, ast.AnnAssign):
                self.check_annotation(node.annotation)
                if node.value is not None and self.is_function_value(node.value):
                    self.report(node.value, "PM003", "Store the concrete owner instead of a function/method reference.")
            elif isinstance(node, ast.Assign):
                if self.is_function_value(node.value):
                    self.report(node.value, "PM003", "Store the concrete owner instead of a function/method reference.")
            elif isinstance(node, ast.Lambda):
                self.report(node, "PM003", "Replace lambda behavior with a direct named method call.")
            elif isinstance(node, ast.Call):
                name = self.name(node.func)
                if name in {"getattr", "setattr", "hasattr", "vars", "eval", "exec", "builtins.getattr", "builtins.setattr", "builtins.hasattr", "builtins.vars", "builtins.eval", "builtins.exec"}:
                    self.report(node, "PM006", "Dynamic access/execution needs an explicit external-boundary justification.")
                if name in {"json.dumps", "json.dump"} or name.endswith((".model_dump_json", ".model_dump")):
                    self.report(node, "PM007", "Use a specialized renderer; exempt only explicit serialization/debug boundaries.")
                if isinstance(node.func, (ast.Subscript, ast.Call)):
                    self.report(node, "PM008", "Indirect call through a registry/factory; call a concrete method.")
                values = list(node.args)
                values.extend(keyword.value for keyword in node.keywords)
                for value in values:
                    if self.is_function_value(value):
                        self.report(value, "PM003", "Possible function/method passed as a value; call the concrete owner directly.")
                    if isinstance(value, ast.Starred):
                        self.report(value, "PM001", "Replace argument unpacking with explicit parameters.")
                for keyword in node.keywords:
                    if keyword.arg is None:
                        self.report(keyword.value, "PM001", "Replace keyword unpacking with explicit parameters.")
            elif isinstance(node, ast.Attribute) and node.attr == "__dict__":
                self.report(node, "PM006", "Use explicit typed fields instead of __dict__.")
        return sorted(self.findings)


def git_output(root: Path, arguments: list[str]) -> bytes:
    command = ["git", "-C", str(root)]
    command.extend(arguments)
    return subprocess.run(command, check=True, stdout=subprocess.PIPE).stdout


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check explicit contracts and readable execution paths without importing application code.",
        epilog="Rules: PM001 variadics/unpacking; PM002 Any/object; PM003 callable behavior; "
        "PM004 mapping contracts (domain maps need an exemption); PM005 dictionary escape; "
        "PM006 reflection; PM007 generic serialization; PM008 indirect calls; PM009 missing annotations. "
        "Exceptions: append '# pm-lint: allow PM007 -- filesystem serialization' to the reported line. "
        "Heuristics do not resolve cross-module types, arbitrary aliases, or prove call-stack hierarchy. "
        "Pair with strict Pyright and source review. --staged checks entire staged files, including existing violations.",
    )
    parser.add_argument("paths", nargs="*", type=Path, help="Python files/directories; default: pm workflows simulation main.py")
    parser.add_argument("--staged", action="store_true", help="Read staged Python blobs, never stash or modify the working tree")
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    findings: list[Finding] = []
    count = 0
    if arguments.staged:
        if arguments.paths:
            parser.error("--staged cannot be combined with paths")
        staged = git_output(root, ["diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"])
        for raw_path in staged.split(b"\0"):
            if not raw_path:
                continue
            path = raw_path.decode("utf-8")
            if not path.endswith(".py"):
                continue
            source = git_output(root, ["show", ":" + path])
            encoding, _ = tokenize.detect_encoding(io.BytesIO(source).readline)
            findings.extend(ReadabilityCheck(path, source.decode(encoding)).check())
            count += 1
    else:
        paths: list[Path] = arguments.paths
        if not paths:
            paths = [root / "pm", root / "workflows", root / "simulation", root / "main.py"]
        files: set[Path] = set()
        for path in paths:
            if not path.exists():
                parser.error(f"Path does not exist: {path}")
            if path.is_dir():
                files.update(path.rglob("*.py"))
            else:
                files.add(path)
        for path in sorted(files):
            with tokenize.open(path) as stream:
                source = stream.read()
            findings.extend(ReadabilityCheck(str(path), source).check())
            count += 1
    for finding in findings:
        print(f"{finding.path}:{finding.line}:{finding.column}: {finding.rule} {finding.message}")
    print(f"Checked {count} Python files; {len(findings)} findings.")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
