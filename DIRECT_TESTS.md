Every module must support direct debugger use without a thread.
These checks use real module behavior, real Ghost models, and real Python-role workflows.
The bus and working targets also remain real.
Only time advances through an explicit test clock.

Use Python's standard-library unittest.
Create `tests/__init__.py`, `tests/support.py`, `tests/test_modules.py`, and `tests/test_shell.py`.
Do not add pytest to the application dependencies.
The workflow repository retains its existing pytest setup.

The single-step contract is exact:

```python
module.step(block=False) -> bool
```

`True` means that one batch or one timed tick reached `handle_batch()`.
`False` means that no work ran, including a termination-only step.
The call is synchronous.
It returns after the workflow and output hook complete.
An exception propagates directly to the test and debugger.
It does not pass through the thread error callback.

Use this algorithm inside `ShellModule.step()`:

1. If the stop event is set, finish termination once and return false.
2. Read the supplied clock.
3. Calculate zero wait for a direct call.
4. For a blocking call, calculate the time until the next tick, or an indefinite wait without a timer.
5. Take the current mailbox batch with that wait.
6. If the stop event is set, finish termination once and return false.
7. If the batch contains messages, call `handle_batch(batch)` and return true.
8. Read the clock again.
9. If a timer is due, set its next deadline to `now + tick_seconds`.
10. For that due timer, call `handle_batch([])` and return true.
11. Otherwise, return false.

Initialize the first deadline to `clock() + tick_seconds` during construction.
A data batch does not move that deadline.
A late timer produces one tick, not a burst of missed ticks.
The scheduler uses the same supplied clock for task deadlines.
Its timed handler advances each task by at most one state per call.
It calculates the next task deadline from the current clock value.

`ShellModule.run()` remains the thread entry API.
Its loop calls `step(block=True)` without another behavior path.
Direct callers can use `prepare_input()`, role functions, or `publish_output()` for narrower debugging.
The standard module tests enter through `step()` to exercise the complete module contract.

`build_app()` constructs and registers objects without startup side effects.
Expose modules through these application attributes:

```python
app.user_input
app.main_cog
app.dialogue
app.inner_thought
app.scheduler
app.status
app.tool_dialogue
app.ghost
app.bus
app.tasks
```

`UserInputProxy.submit(text: str) -> str` stores the input Feature and publishes `user.input`.
It returns the correlation ID.
Its `handle_batch()` resolves public reply Features and calls `emit_reply(feature.content)`.
Terminal stdin only calls `submit()`.
Tests pass `list.append` as `emit_reply`, which is a real output sink.
No stdin reader is necessary in these tests.

Create these helpers in `tests/support.py`:

| Helper | Exact behavior |
| --- | --- |
| `ManualClock` | Stores a float, returns it from `__call__()`, and increments it through `advance(seconds)`. |
| `Recorder` | Subscribes to the real bus and appends matching envelopes to a list. |
| `make_app()` | Loads a fresh Ghost and supplies a temporary runtime root, ManualClock, and reply list. |
| `store_feature()` | Stores a real Feature through Ghost and returns its ID. |
| `store_action()` | Stores a real ActionSchemaKnoxel and GWT, then returns an action envelope. |

Parse the full Spamuel seed once in `setUpClass()`.
Keep its serialized state as the reusable seed value.
For each test, deserialize that value into fresh models and construct a fresh Ghost.
Do not share mutable Ghost instances or task tables between tests.
The seed originates from the real YAML file, not a fabricated memory structure.
Add `Ghost.from_state(state)` for this construction path.
`Ghost.from_yaml(path)` delegates to the same constructor after parsing.

Use `TemporaryDirectory` for kernel journals and real-folder checks.
Configure thought and task periods as 30 seconds and 1 second.
Every helper returns its objects directly so debugger inspection remains straightforward.
Do not hide module calls behind a generic test scenario language.
Do not replace the kernel, BashMachine, or MCP tools with mocks.

Implement these module tests in `tests/test_modules.py`:

| Test method | Direct calls | Required assertions |
| --- | --- | --- |
| `UserInputTests.test_submit` | `submit("hello")` | One stored Dialogue Feature, one user envelope, matching returned correlation, no public output. |
| `UserInputTests.test_print_public_reply` | Publish a reply, then `user_input.step()` | The reply sink contains its text exactly once. |
| `MainCogTests.test_batches_two_inputs` | Submit two texts, then `main_cog.step()` | One new cycle, one GWT, two actions, both input IDs in the GWT, separate correlations preserved. |
| `MainCogTests.test_routes_each_command` | One fresh app per command, then `main_cog.step()` | Exact routes and contents from the rule table in IMPLEMENTATION.md. |
| `MainCogTests.test_consumes_telemetry_without_new_action` | Publish real task telemetry, then `main_cog.step()` | One new GWT references the telemetry, with no schedule or dialogue action. |
| `MainCogTests.test_ignores_own_broadcast` | Publish `gwt.updated` and an action, then `main_cog.step()` | False return and no new cycle. |
| `DialogueTests.test_public_reply` | Publish a real dialogue action, then `dialogue.step()` | Exact reply text, one stored Feature, original correlation, and one report file in the invocation environment. |
| `InnerThoughtTests.test_timer` | Step at time zero, advance 30 seconds, then step twice | False, true, false returns, one thought, no public reply. |
| `SchedulerTests.test_dummy_task_transitions` | Publish schedule, step, advance and step twice | Accepted, running, completed in order, with one record and stable correlation. |
| `SchedulerTests.test_duplicate_schedule` | Publish the same action twice, then step | One task record and one accepted event. |
| `SchedulerTests.test_minecraft_is_unsupported` | Publish Minecraft action, then step and advance | One unsupported event and no running or completed event. |
| `StatusTests.test_known_task` | Populate through the scheduler, publish status action, then `status.step()` | Telemetry uses the status-request correlation and the current task status. |
| `StatusTests.test_unknown_task` | Publish unknown status action, then `status.step()` | One unknown-status response, with no new task. |
| `ToolDialogueTests.test_real_memory_search` | Publish tool action, then `tool_dialogue.step()` | Reply count equals the real scoped Ghost search, with one public Feature. |

For command routing, include `/status task-123`, `/memory`, Minecraft text, `/demo-task`, `dumb robot`, and ordinary text.
Use unittest subtests to show which command failed.
Check enum values on stored action objects as well as bus topics.

Create these shared behavior checks in `tests/test_shell.py`:

| Test method | Required behavior |
| --- | --- |
| `test_construction_starts_no_threads` | `build_app()` leaves every module thread absent and starts no application thread. |
| `test_idle_step_returns_without_waiting` | A direct step with an empty mailbox returns false. The mailbox receives a zero wait. |
| `test_fanout` | Two Recorder subscriptions each receive one identical envelope ID. |
| `test_batch_boundary` | A message published during a direct handler remains queued until the next step. |
| `test_input_after_termination` | Terminate delivery sets the event. A step emits one module termination and no workflow result. |
| `test_repeated_termination` | Repeated terminate messages and steps still produce one module termination event. |
| `test_scheduler_termination` | An accepted dummy task becomes terminated once through the normal termination hook. |
| `test_workflow_cancellation_inside_role` | A real temporary role sets its stop event and calls `check_cancelled()`. The kernel reports termination without retry. |
| `test_python_exception_propagates` | A real temporary role raises ValueError. Direct execution raises that same error in the caller. |
| `test_fresh_environment` | A second invocation sees a newly stored Ghost Feature. |
| `test_read_scope` | Bash and an in-process MCP call cannot retrieve an omitted record. |

The active cancellation test requires no second thread.
Its temporary role triggers the cancellation signal at a known instruction boundary.
The separate threaded demonstration still verifies delivery from outside an active role.
These are different checks and must not replace each other.

Keep a reference to the last invocation environment on `ShellModuleWorkflow` as `last_environment`.
Keep `last_kernel_summary` after the kernel returns.
These two fields support debugger inspection and direct assertions.
They retain only the most recent invocation, not an unbounded history.
A direct exception leaves the environment available for inspection.

Use real subscriptions to inspect output instead of private queue fields.
For the batch-boundary check, use a minimal local ShellModule subclass whose handler publishes one additional message.
That subclass exercises the shared algorithm and does not simulate a cognitive workflow.

Add this debugger example to the README:

```python
app = build_app(
    seed_path=seed_path,
    runtime_root=runtime_root,
    thought_period=30.0,
    task_period=1.0,
    clock=clock,
    emit_reply=replies.append,
)
correlation_id = app.user_input.submit("you are a dumb robot!")
app.main_cog.step()       # Breakpoint: prepare input, role, Ghost commit, bus output.
app.dialogue.step()       # Breakpoint: real Python workflow and reply construction.
app.user_input.step()     # Breakpoint: public output.
assert replies == ["You are a dumb human, Jesse! Dumb and nasty."]
```

The example defines no threads and performs no sleeps.
The user can inspect Ghost, task records, the latest environment, and recorded messages after each line.

Support these commands from `pm_next_v2`:

```powershell
.\.venv312\Scripts\python.exe -m unittest discover -s tests -v
.\.venv312\Scripts\python.exe -m unittest tests.test_modules.DialogueTests.test_public_reply -v
```

For IDE debugging, select the second unittest target and set breakpoints in the module, role, or output hook.
Direct workflow tests exercise several real components even though unittest runs them.
Their guarantee is thread-free debugger access, not complete component isolation.
Do not claim that these checks verify concurrent ordering or thread shutdown.
Keep those checks in the separate integration demonstration.
