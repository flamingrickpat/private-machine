Design audit, September 12, 2026.

The implementation preserves the main design.
All six cognitive modules use the real workflow kernel and fixed Python roles.
The shared ShellModule owns threads, inboxes, direct steps, timers, and termination.
User input enters MainCogLoop before action selection.
Ghost and its Bash projections remain in RAM.
The implementation does not add triage, a steward, or external task execution.

The audit corrected these material departures:

| Finding | Correction | Evidence |
| --- | --- | --- |
| Ghost files existed only in private BashMachine dictionaries. Directory scans did not find them. | Register lazy files through the public BashMachine API. | Direct listing, recursive grep, exact read, and rejected write checks. |
| The shared workflow adapter copied unrestricted Ghost records into input files. This bypassed a restricted target. | Resolve cognitive input from the invocation snapshot. Missing references raise an error. | A hidden action cannot enter the workflow request. |
| ShellModuleWorkflow required Ghost and knew the knoxel format. | Move that knowledge into CognitiveWorkflow. Keep the generic runner independent of Ghost. | A plain real-folder workflow runs without a Ghost object. |
| Read tools returned the mutable snapshot dictionaries. | Return independent serialized values. | A changed tool result cannot change a later tool or Bash read. |
| Exceptions in termination hooks bypassed the application error handler. | Include cleanup in the shared thread exception boundary. | A cleanup ValueError reaches the application. |
| Repeated kernel steps lost the terminated status. | Preserve the terminal cancellation result without another journal entry. | Kernel cancellation regression check. |
| Individual file registration created thousands of asyncio loops on Windows and stalled in socket creation. | Add `BashMachine.write_texts()` in pm-coder and register each projection as one batch. | Full-seed module checks and BashMachine regression checks. |
| BashMachine stripped whitespace from virtual file reads. | Add an optional raw-output mode and use it in the working target. | A Markdown file retains its leading spaces and final newlines. |

The generic Ghost coupling also appeared in the earlier specification.
The audit corrects that specification choice to match the user's broader module design.
ToolPlusDialogue now shares Dialogue's output hook instead of copying it.

The direct tests remain synchronous and usable in a debugger.
New design checks use small real Ghost objects and actual Python-role workflows.
The existing module tests still use the full Spamuel seed.
Their serialized seed now loads once per test process, with independent Ghost models for each test.
Separate integration checks start actual module threads and terminate an active workflow through the bus.
The MCP check uses an actual in-process client/server call.

The following limits remain visible for the next development pass:

- Agent drivers do not yet receive the virtual runtime. This milestone supports Python roles only.
- Each invocation serializes its visible Ghost records before lazy YAML rendering. Full-memory views can make steps slow.
- Several dummy roles use empty defaults for missing fields. These can hide malformed internal input.
- Bus correlation IDs do not consistently appear in stored knoxel metadata.
- TaskTable exposes mutable records outside its lock. Its current scheduler is the only production writer.
- Status telemetry enters Ghost and MainCogLoop but does not produce a public status reply.
- The terminal does not yet print task acceptance IDs as specified.
- Envelope attributes remain mutable despite its immutable description.
- Several older kernel probes use a dictionary environment. They do not verify Bash behavior.
- PATH selects a WSL launcher without a usable Bash installation. The real-folder check passes with Git Bash selected explicitly.

These limits do not require another module architecture.
No commits or pushes were made during this audit.
The existing uncommitted pm-workflows implementation remains in place on `steward`.
The pm-coder changes preserve the default output behavior for existing callers.

Verification results:

| Check | Result |
| --- | --- |
| Application unittest discovery, including full-seed direct tests and threaded checks | 34 passed in 120.9 seconds |
| pm-workflows runtime-context, driver, and kernel tests | 55 passed |
| pm-coder BashMachine tests | 30 passed |
| Existing demonstration with Git Bash first on PATH | 21 of 21 passed |
| Diff whitespace checks in all three repositories | Passed |

The application suite used unittest discovery with kernel stdout captured to keep the audit output readable.
The rejected MCP lookup intentionally produces an error log during its permission check.
