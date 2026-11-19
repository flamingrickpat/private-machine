# Development rules and readability checks

This project prioritizes code that its owner can understand, navigate, change, and debug on Windows.
[AGENTS.md](AGENTS.md) defines the mandatory engineering rules. This page summarizes the rules and explains the readability checker.

## Explicit types and direct execution

- Name and type every function parameter and return value. Use explicit parameters instead of `*args`, `**kwargs`, or argument unpacking.
- Use concrete domain types. Do not use `Any` or `object` as substitutes for known inputs, results, or dependencies.
- Use named Pydantic models or typed dataclasses for structured records. Keep anonymous dictionaries inside the function that creates them.
- Use dictionaries for real maps keyed by domain IDs. Do not use string keys as substitutes for named record fields.
- Call concrete functions and methods directly. Pass the object that owns an operation instead of its bound method.
- Do not select internal behavior through callbacks, function pointers, lambdas, callable objects, factories, or function registries.
- For runtime selection among fixed operations, use a domain enum and explicit branches that call named implementations.
- Keep required dependencies and intermediate state in named, typed fields or parameters. Do not hide them in closures or ambient state.
- Keep external callback requirements inside narrow, named adapters. Document the external requirement and call concrete application methods from the adapter.

The owner must be able to use **Go to Definition** from the application entry point to the HTTP client and external library boundary.
The call stack must show the hierarchy of operations. Function pointers must not route lower-level operations back through higher-level application units.
Every wrapper must own a necessary responsibility. Tracebacks must retain the original cause.

The complete requirements are in [AGENTS.md](AGENTS.md), especially sections 0A, 1, 6, 7, and 8.

## Render prompts explicitly

Use a specialized renderer to turn a Pydantic model or knoxel into prompt text.
Prefer utilities that render each knoxel with clear meaning and efficient token use.
Select the fields and format that the prompt needs. Do not dump an entire object into a prompt.

`json.dumps` is permitted for debugging and filesystem serialization. It is not a prompt renderer.
Pydantic dumps also require an explicit boundary purpose. Foreign wire serialization belongs at the final external boundary.
An exception for serialization does not permit anonymous records throughout application code.

## Readability checker

[dev/tools/readability_lint.py](dev/tools/readability_lint.py) uses Python's standard-library AST parser. It does not import application modules or require external services.

Run these commands from the repository root.

Check the default application roots (`pm/`, `workflows/`, `simulation/`, and `main.py`):

```powershell
.venv/Scripts/python.exe dev/tools/readability_lint.py
```

Check selected files or directories:

```powershell
.venv/Scripts/python.exe dev/tools/readability_lint.py pm/llm/http_client.py workflows/inner_thought
```

Check staged Python files:

```powershell
.venv/Scripts/python.exe dev/tools/readability_lint.py --staged
```

`--staged` reads file contents from Git's index. It does not stash, rewrite, or stage working files.
It checks entire staged files, including existing violations. It cannot be combined with file or directory arguments.
The checker does not install a commit hook. A hook can run the staged command and propagate its exit code.

Findings use the format `path:line:column: rule message`.
Exit code `0` means no findings. Exit code `1` means findings exist.
Invalid command arguments or unreadable inputs also cause a nonzero exit.

### Rules

| Rule | What it flags |
| --- | --- |
| `PM000` | Python syntax errors or malformed exemption comments. |
| `PM001` | Variadic parameters and positional or keyword argument unpacking. |
| `PM002` | `Any` or `object` annotations, including recognized import aliases and quoted annotations. Invalid quoted type expressions also produce findings. |
| `PM003` | Callable annotations, lambdas, `__call__` methods, and recognized function or method references assigned, passed, or returned. |
| `PM004` | Dictionary and mapping annotations. Legitimate domain maps require an explicit exemption. |
| `PM005` | Recognized anonymous dictionaries passed to calls or returned from functions, including local aliases and dictionaries inside literal collections. |
| `PM006` | Dynamic access through `getattr`, `setattr`, `hasattr`, `vars`, or `__dict__`, and execution through `eval` or `exec`. |
| `PM007` | `json.dumps`, `json.dump`, `model_dump`, and `model_dump_json` calls. |
| `PM008` | Calls through subscript expressions or factory results, such as `registry[key]()` or `factory()()`. |
| `PM009` | Missing parameter or return annotations. The checker permits implicit types for `self` and `cls`. |

### Narrow exceptions

First determine whether the code belongs at a documented external boundary or represents a legitimate domain map.
If an exception is necessary, append its rule and concrete reason to the reported line:

```python
serialized = json.dumps(payload)  # pm-lint: allow PM007 -- filesystem serialization
scores: dict[KnoxelId, AttentionScore] = {}  # pm-lint: allow PM004 -- attention scores indexed by knoxel ID
```

Each exemption applies to one rule on one line. Blanket file exemptions are not supported.
A missing reason or malformed exemption produces `PM000`.
Exemptions record the boundary reason. They do not override [AGENTS.md](AGENTS.md) or permit forbidden internal behavior.

### Limits and review

The checker uses syntax and local heuristics. It does not resolve all cross-module types, aliases, data flows, or runtime call targets.
It can miss disguised callbacks and dictionary escapes. It can also flag valid code that needs a narrow exemption.
It cannot prove a hierarchical call stack, useful IDE navigation, or efficient prompt rendering.

Use the checker together with strict type checking and the source review required by [AGENTS.md](AGENTS.md).
For changed runtime paths, inspect the concrete calls and named state. Exercise the real path when prerequisites are available.
A clean lint result alone does not establish debuggability or correct behavior.

Run the deterministic checker tests:

```powershell
.venv/Scripts/python.exe -m unittest dev.tools.test_readability_lint -v
```

[dev/tools/test_readability_lint.py](dev/tools/test_readability_lint.py) covers rule detection, exemptions, CLI exit codes, and staged-file isolation.
Its Git test uses a temporary repository and checks that the staged command preserves working file contents.
