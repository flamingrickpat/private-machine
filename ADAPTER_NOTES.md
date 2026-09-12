These notes contain integration details for `HANDOFF.md`.
Use `IMPLEMENTATION.md` for the final API choices and `DIRECT_TESTS.md` for direct debugger checks.
Their exact contracts replace the earlier alternatives in these source notes.
Source inspection occurred on September 12, 2026.
The paths identify reuse candidates, not permission to copy whole subsystems.

| Source | Useful content |
| --- | --- |
| `../pm_next/pm_core/ghost/ghost.py` | `Ghost.from_yaml()` and existing memory helpers |
| `../pm_next/pm_core/ghost/ghost_base.py` | Ghost state, indexes, and knoxel operations |
| `../pm_next/pm_core/model/` | Knoxel types, GWT records, and Ghost state |
| `../pm_next/pm_core/persist/dataprovider_yaml.py` | YAML loader |
| `../pm_next/archive/artifacts/spamuel_agentic_canon.yaml` | Existing Spamuel seed |
| `../pm_next/archive/artifacts/config_spamuel.yaml` | Spamuel character data |
| `../pm_next/ghost_shell.py` | An earlier interface blueprint with abstract adapters |
| `../pm-coder/pm_bash_machine.py` | `BashMachine`, virtual files, and file permissions |
| `../pm-workflows/pm_workflows/kernel.py` | Real kernel constructor and workflow execution |
| `../pm-workflows/pm_workflows/drivers/python_driver.py` | Python roles without an LLM |
| `../pm-workflows/pm_workflows/python_role.py` | `RoleContext` and `run(context)` contract |

Read each copied source file fully before selecting its contents.
Preserve required loader behavior, identity allocation, and index updates.
Remove unrelated runtime dependencies from the copied subset.
The current Ghost module imports LLM infrastructure even though `from_yaml()` defaults to `with_llm=False`.
Do not copy the full dependency list to satisfy those imports.
Keep real knoxel models instead of replacing Ghost with a dictionary.

The source project declares Python 3.12.
The destination starter declares Python 3.13 or later.
Verify package compatibility before choosing the prototype interpreter.
Record the interpreter and local package revisions in the README.
Do not refresh packages in the running Hadal environment.

`BashMachine` already provides named users, virtual files, lazy content, and per-file access.
Its public API is synchronous.
Its filesystem operations share a lock.
Use its public methods instead of rebuilding a Bash emulator.

Two behaviors need explicit treatment:

- Files without explicit access rules default to read/write access.
- Lazy `content()` results remain cached after the first read.

Expose only selected Ghost records in each environment.
Apply explicit read access to every Ghost file in a read environment.
Leave the invocation inbox, scratch space, and outbox writable as appropriate.
Read mode describes Ghost access, not the entire working target.
Refresh the Ghost projection at each invocation so later invocations see new memory.
An invocation can use a stable snapshot until it completes.

Keep Ghost data in RAM through virtual Markdown or YAML files.
Parse workflow reports through the shared Ghost write method.
Do not assume that edits to virtual text automatically update knoxel objects.
Keep MCP reads within the same visible Ghost scope as Bash reads.
Python roles remain trusted application code with host access.
The virtual view does not sandbox arbitrary Python code.

Implement a few tools such as record lookup, text search, and current GWT retrieval.
Register those functions through the installed MCP library.
Reuse the functions directly for the dummy tool-plus-dialogue role.
If the installed library supports in-process transport, exercise one real MCP client call through that transport.
Keep MCP transport details inside `ghost_mcp.py`.
Do not add a Ghost HTTP service for this milestone.

The current kernel converts `workspace` into a real `Path`.
Its Python `RoleContext` also contains real paths.
Virtual working targets are not a verified kernel feature.
Python roles provide an existing route to real workflow execution without model calls.

First, run one minimal manifest with a Python role.
Then connect that invocation to the virtual target through `workflow_adapter.py`.
Use an existing extension point for the target object if one fits.
Otherwise, add the required interface directly in `C:\source\pm\pm-workflows` on the `steward` branch.
The user authorizes these changes without another approval step.
Pass the working target through explicit invocation context.
Keep the generic target contract in the shared library and the Ghost implementation in `pm_next_v2`.
Keep each target binding scoped to one invocation.
Do not use a global current-Ghost variable across module threads.

The kernel can keep its manifests, journals, and required workspace metadata in a local runtime folder.
That folder is separate from the virtual Ghost working target.
The workflow role reads its input and writes its report through the target adapter.
No complete Ghost export is necessary.
Document any request or report files that the kernel still requires on disk.

Do not build a second workflow interpreter to bypass this integration.
Make the smallest coherent kernel change that supports the prototype.
Preserve the existing real-folder workflow behavior.
Install the local checkout in editable mode inside the prototype environment.
Verify a virtual-target invocation and the existing real-folder path after the kernel changes.
Use the existing pm-workflows checks for the affected behavior.
Keep application routing and Ghost-specific logic outside the kernel.
If integration remains incomplete, report the exact unresolved path.
An installed package alone does not satisfy the workflow integration requirement.

The required real-folder target can remain small.
Give it the same file operations that the virtual target uses.
Verify those operations in a temporary demonstration folder.
No coding workflow or external shell task is required.

One invocation needs one cancellation signal.
Fixed roles check that signal between their steps.
The workflow adapter must preserve termination as a distinct outcome.
The adapter must not report success after termination.
Future blocking agent calls will need a separate cancellation design.
