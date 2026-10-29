# Operating-system boundary

The owner selected one utility class for operating-system and process details.
Windows remains the verification target. Linux and macOS can receive specific branches later.
The model autorestart scripts are external owner tools. They remain unchanged.

Parent: [structural refactor](index.md).

## Observed execution before implementation

`simulation.server` installs a Windows cleanup job, then `simulation.client` launches participant processes.
Participants and terminal entry points also install cleanup jobs.
`pm.console.server` separately launches workers, kills trees, and opens source files.
`pm.llm.activation` launches finite startup commands asynchronously.
Folder commands and source-revision capture call `subprocess` directly.
`ProcessLock` chooses `msvcrt` or `fcntl` inside receipt persistence code.
Signal registration and low-level filesystem calls are spread across other runtime modules.

The prior Windows job implementation was `pm.host.processes.ensure_process_cleanup()`.
It retains a native handle in a module global.
Launch flags and tree termination have separate Windows branches in their callers.
This scatters the locations that would need a platform-specific change.
A child-owned job also cannot bind that child's lifetime to its external launching shell.

## Selected structure and contracts

`pm/utils/operating_system.py` becomes the native boundary under the existing leaf utility package.
Its `OperatingSystem` class replaces `pm.host.processes` and holds the cleanup handle.
This placement lets memory, model access, and host mechanisms share it without reversing package dependencies.
Its import-time dependencies are standard-library modules only.

The class owns these contracts:

- `install_child_cleanup() -> bool` and `child_cleanup_installed() -> bool` retain the existing job policy.
- `start_logged_process(...) -> Popen[bytes]` launches a participant with merged log output.
- `start_piped_process(...) -> Popen[str]` launches an interactive worker with text pipes.
- `run_text(...) -> CompletedProcess[str]` and `output_bytes(...) -> bytes` execute finite commands.
- `start_async_process(...) -> asyncio.subprocess.Process` preserves finite startup execution.
- `kill_process_tree(...) -> None` retains Windows tree termination and the existing other-host behavior.
- `register_stop_signals(...)` returns previous handlers; `restore_signals(...)` restores them.
- `lock_file(...) -> None` performs native nonblocking acquisition.
- Environment access, process identity, immediate exit, file synchronization, hard linking, replacement, and descriptor writes share this boundary.
- Document and source-editor opening keep their existing Windows choices.

Methods use explicit arguments, concrete results, and no untyped forwarding dictionary.
The process objects remain standard typed handles for ordinary lifecycle inspection.
`ProcessLock` keeps resource ownership, cancellation, and waiting policy in `durable_files.py`.
Its native lock operation delegates to `OperatingSystem`.
Application callbacks keep their shutdown behavior. The utility only registers the signals.
No domain state or workflow assembly enters this class.

## Ownership and failure boundaries

The class owns a process-local, noninherited cleanup-job handle.
The operating system closes it at process exit.
Children remain in the job according to the existing Windows inheritance rules.
Existing external model servers remain outside the runtime job.
Job installation still logs a rejected host operation and reports failure through its boolean result.
No compatibility alias will preserve the old module; all known callers and API links move together.

Persistence paths, record schemas, model prompts, startup command arguments, and retry policy remain unchanged.
The class does not create new Linux or macOS guarantees.
Existing `flock` and process behavior remain where already implemented.
Installing cleanup in an external launcher is a separate ownership decision from this encapsulation.

## Implementation order

1. Add the typed class and move the native job implementation into it.
2. Route process launches, native locks, stop signals, and low-level OS calls through the class.
3. Update entry points, development recovery, tests, source links, and the runtime guide.
4. Exercise real process lifetimes, lock contention, finite commands, and asynchronous startup.
5. Run affected architecture and strict type checks; build and inspect the manual.

## Verification and risks

Tests require real local processes and files, without model services or test doubles.
Normal parent exit and forced parent death must remove both a child and grandchild.
A competing process must fail to acquire an owned file lock, then acquire it after release.
Text, binary, and asynchronous command results must retain their actual exit codes and outputs.
Existing startup-script and signal contracts need affected-consumer checks.
The full companion suites and K002 scenarios remain stopped at the owner's request.

The main risks are changed pipe wiring, cancellation, handle inheritance, signal restoration, and import cycles.
The new module therefore uses only the standard library and keeps each existing caller's policy explicit.
The owner has selected the single-class direction. No additional platform behavior is selected here.

## Placement correction

The initial caller map suggested the host package.
The architecture checks show that memory and model access must not import that package.
The implementation therefore uses the existing `pm.utils` leaf package.
The module still has no PM package dependencies. No dependency rule is loosened.

## Implementation and verification

The class now owns the native operations listed above.
All product callers use the class. The old module has no compatibility alias.
An architecture check rejects native imports, OS switches, and process-creation calls outside the boundary.
The process checks now live in `tests/unit/pm/utils/test_operating_system.py`.

The terminal's existing module assembly now has a named protocol.
Its annotations describe the current factory contract without importing companion behavior.
The existing home path supplies no seed when it supplies an open memory store; the factory annotation now permits that case.
The development recovery command also has named input and outbound receipt models.
Its JSON receipt fields remain unchanged.

Evidence is under `.runtime/kanban/k002-audit-20261029/owner-shutdown/`.
Real process checks cover normal exit, forced exit, child and grandchild termination, finite commands, asynchronous execution, signals, and native lock contention.
File checks inspect bytes after synchronization, hard linking, and replacement.
A temporary finite Python startup script writes a real receipt through the model activation helper.
No test starts a model server or changes the owner's autorestart scripts.

Thirty-seven selected process, architecture, checkpoint, and causal-recovery checks passed in `final-focused-tests.log`.
The OS utility and its real-system test module pass strict checking.
The configured strict check still covers only its configured selection.

The selected CLI restart check also fails on baseline commit `99c6b66`.
Its legacy fixture writes a reply without an accepted action and expects that reply in the clean recovery view.
The retained receipts are `cli-architecture-tests.log` and `baseline-cli-restart.log`.
This refactor does not change that causal contract or treat the failure as a pass.
Full companion and K002 live scenarios remain stopped at the owner's request.
Linux and macOS process-lifetime guarantees remain unverified.

## Complete repository scope

The owner explicitly included historical development gates and standalone test helpers.
These callers now use the same class for process creation, termination, file handles, environment access, and process memory measurement.
The architecture guard covers `pm`, `simulation`, `workflows`, `dev`, and `tests`.
Generated child-program strings remain ordinary Python inputs to the real process-lifetime checks.
They exercise job inheritance without installing a cleanup job in each descendant.

Legacy native file-sharing fixtures now open and close their actual Windows handles through the class.
Binary and text process methods preserve the original stdin, stdout, stderr, and locale choices.
Historical worker-group termination retains its existing Windows tree operation and Unix process-group operation.
Development scripts establish the repository import path before loading the shared utility.

The historical RSS helper retains its optional psutil provider and Unix fallback.
On Windows without psutil, it reports the missing measurement prerequisite explicitly.
It does not fabricate a memory result or require installing another package.

`all-scope-native-tests.log` records 21 passing architecture, real-process, file-sharing, and PowerShell checks.
The workflow file-lock case retained its phase journal and confirmed one role execution while projection retried.
`all-scope-pyright-final.json` records zero strict errors for the OS utility, its test module, and the real terminal entry point.
No model or Minecraft acceptance gate was restarted.

The final persistence and launch-plan selection passed 26 checks in `persistence-plan-final.log`.
All 84 modified Python modules compiled and imported without invoking acceptance entry points (`imports.log`, `imports.json`).
The configured strict selection still reports zero errors in `configured-pyright-final.json`.
The RSS probe is `NOT RUN — missing psutil` on this Windows environment; `rss-probe.json` retains the actual prerequisite error.

The final documentation command is `python -m mkdocs build --strict` through the working audit interpreter.
Its receipt is `docs-build-complete-scope.log`.
Browser inspection covers the index, runtime guide, this engineering record, and the generated OS utility API.
The runtime guide's class link must reach that actual API page.
No conceptual diagram changed.
