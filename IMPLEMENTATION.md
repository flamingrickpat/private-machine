This is the implementation specification for the dummy Spamuel prototype.
It resolves the design choices in `HANDOFF.md` and `ADAPTER_NOTES.md`.
If those documents conflict with this specification, use this specification.
Implement these contracts without another architecture pass.
Report a source mismatch with its exact location before changing a contract.

The application uses Python 3.12.
Use a separate `.venv312` for the prototype.
Declare Python `>=3.12,<3.13` in the destination project.
Install `pm-workflows` editable from `../pm-workflows` on branch `steward`.
Install `pm-coder` editable from `../pm-coder` after dependency resolution.
Verify both import locations after installation.
Direct application dependencies are `pm-workflows`, `pm-coder`, `pydantic`, `PyYAML`, and `fastmcp`.
Do not install the old `pm-next` distribution.
Record resolved dependency versions after the first successful demonstration.

The entry point owns construction only.
All module loops use Python threads.
All workflow roles use the real kernel and its Python driver.
All messages remain in RAM.
The kernel keeps its own files under `.runtime/`.
Ghost data remains in RAM after the seed load.

The shared library changes are exact:

| File in `pm-workflows` | Change |
| --- | --- |
| `pm_workflows/working_target.py` | Add the two ABCs and command result described next. |
| `pm_workflows/runtime.py` | Add `WorkflowRuntime` and `WorkflowTerminated`. |
| `pm_workflows/python_role.py` | Append `runtime: WorkflowRuntime | None = None` to `RoleContext`. |
| `pm_workflows/kernel.py` | Accept optional runtime, pass it to Python roles, and handle cancellation at phase boundaries. |
| `pm_workflows/drivers/python_driver.py` | Propagate cancellation and opt-in Python exceptions. |
| `tests/test_runtime_context.py` | Add real Python-role integration checks for the new behavior. |

Use these signatures in `pm_workflows/working_target.py`:

```python
@dataclass(frozen=True)
class CommandResult:
    stdout: str
    stderr: str
    exit_code: int

class WorkingEnvironment(ABC):
    @abstractmethod
    def read_text(self, path: str) -> str: ...
    @abstractmethod
    def write_text(self, path: str, text: str) -> None: ...
    @abstractmethod
    def move(self, source: str, destination: str) -> None: ...
    @abstractmethod
    def list_files(self, directory: str) -> list[str]: ...
    @abstractmethod
    def exec(self, command: str) -> CommandResult: ...

class WorkingTarget(ABC):
    @abstractmethod
    def open(self, invocation_id: str) -> WorkingEnvironment: ...
```

File paths are relative POSIX paths, such as `inbox/request.yaml`.
`list_files()` returns sorted file paths relative to the environment root.
It lists direct child files only.
`write_text()` creates missing parent directories.
`move()` creates the destination parent and moves one file.
There is no implicit disk export or close operation.
The caller releases a virtual environment after its invocation finishes.

Use this runtime contract:

```python
class WorkflowTerminated(Exception):
    pass

@dataclass
class WorkflowRuntime:
    environment: WorkingEnvironment
    stop_event: threading.Event
    tools: Mapping[str, Callable[..., object]]
    propagate_python_errors: bool = False

    def check_cancelled(self) -> None:
        if self.stop_event.is_set():
            raise WorkflowTerminated()
```

The application supplies only approved tool functions to each invocation.
The kernel does not interpret those functions or import Ghost.
This context is an in-process reference, not a JSON field or a manifest variable.
It does not grant capabilities to CLI agents.
Virtual-target execution through those agents remains outside this milestone.

Append `runtime: WorkflowRuntime | None = None` to `Kernel.__init__()`.
Store it as `self.runtime`.
Pass it into the existing `RoleContext` construction in `_execute_role()`.
Pass it into child kernel construction to preserve cancellation and target identity.
Existing callers without runtime retain their current behavior.

At `step()` entry, check cancellation before dispatch.
After `_execute_phase()`, check cancellation before acceptance and routing.
Catch only `WorkflowTerminated` for this new terminal path.
Use one `_terminate()` helper for both boundaries and cooperative role cancellation.
The helper sets `exit_reason="terminated"`, `_finished=True`, and `_next_phase_name=None`.
Append one journal entry with `kind="termination"`, `status="terminated"`, and `ok=False`.
Render the terminal state through the existing kernel method.
Return `_step_result()` with `valid=False`, `status="terminated"`, and empty errors.
The existing `_summary()` must then report `ok=False` and `terminal_status="terminated"`.
Repeated `step()` calls return that terminal state without another journal entry.
Do not route termination through role retry, repair, or checkpoint acceptance.

In `PythonDriver._call_entry_point()`, re-raise `WorkflowTerminated` before the existing broad exception handler.
If `context.runtime.propagate_python_errors` is true, re-raise other Python exceptions.
If that flag is false or runtime is absent, preserve the existing result conversion.
Keep the existing `sys.modules` cleanup in `finally`.
The prototype sets the flag to true.
No driver factory change is necessary: `kind="python"` already exists.

The application files have these owners:

| File in `pm_next_v2` | Contents |
| --- | --- |
| `main.py` | `build_app()`, CLI selection, terminal input, and shutdown |
| `shell.py` | `Envelope`, `Mailbox`, `BusSystem`, `ShellModule`, and `Application` |
| `modules.py` | Cognitive subclasses, fixed routing predicates, and `TaskTable` |
| `workflow_adapter.py` | `ShellModuleWorkflow` and the kernel invocation |
| `working_target.py` | `GhostBashWrapper`, virtual environment, and `FolderTarget` |
| `ghost.py` | Small Ghost subclass, synchronized writes, and snapshot creation |
| `ghost_mcp.py` | Scoped tool functions and FastMCP registration |
| `workflows/roles.py` | All six fixed Python role functions |
| `workflows/*.workflow.md` | Six manifests, one for each cognitive module |
| `workflows/roles/*.py` | Six required `run(context)` entry files |
| `demo.py` | Scripted integration scenarios through `build_app()` |
| `tests/test_modules.py` | Direct module behavior checks without threads |
| `tests/test_shell.py` | Direct bus, batch, timer, and termination checks |

The six entry files only adapt the kernel's fixed `run(context)` convention.
Each imports and calls the matching function from `workflows.roles`.
These wrappers are required by the current Python driver.
Do not add further wrapper layers.

Use this envelope:

```python
@dataclass(frozen=True)
class Envelope:
    id: str
    source: str
    kind: Literal["data", "telemetry", "control"]
    topic: str
    correlation_id: str
    payload: dict
    target: str | None = None
```

Payloads contain JSON-compatible data, never live owner-backed knoxel objects.
A knoxel crosses the bus as `{"knoxel_id": 123}` after Ghost stores it.
An action adds `gwt_tick` to that payload.
A new user input uses its envelope ID as its correlation ID.
Every downstream action and result retains the original correlation ID.
One batched GWT can serve several correlations.
There is no single replacement correlation ID for that batch.

Use these exact topics and subscriptions:

| Producer | Topic | Payload | Consumer |
| --- | --- | --- | --- |
| UserInputProxy | `user.input` | `knoxel_id` | MainCogLoop |
| MainCogLoop | `gwt.updated` | `gwt_tick` | Debug/demo subscriptions only |
| MainCogLoop | `action.dialogue` | `knoxel_id`, `gwt_tick` | Dialogue |
| MainCogLoop | `action.schedule` | `knoxel_id`, `gwt_tick` | ScheduleSubprocesses |
| MainCogLoop | `action.status` | `knoxel_id`, `gwt_tick` | CheckSubprocess |
| MainCogLoop | `action.tool_dialogue` | `knoxel_id`, `gwt_tick` | ToolPlusDialogue |
| Dialogue or ToolPlusDialogue | `dialogue.reply` | `knoxel_id` | UserInputProxy and MainCogLoop |
| InnerThoughtLoop | `thought.created` | `knoxel_id` | MainCogLoop |
| ScheduleSubprocesses or CheckSubprocess | `task.telemetry` | `knoxel_id`, `task_id`, `status` | MainCogLoop |
| Application | `terminate` | Empty | The module named by `target` |
| Any module | `module.terminated` | `module_id`, `invocation_id` or null | Application/demo only |

All topics use `data` except task/module telemetry and terminate control.
The MainCogLoop does not subscribe to its own GWT or action topics.
The user output does not subscribe to general Feature records.
That rule prevents internal thoughts from appearing as public replies.

`BusSystem.subscribe(name, accepts, deliver)` registers one subscriber.
`BusSystem.put(envelope)` visits matching subscribers in registration order.
The bus holds a lock while it selects subscribers and enqueues messages.
Predicates only inspect the envelope.
Delivery only enqueues data or sets a termination signal.
Neither operation runs a workflow or recursively publishes a message.
All subscriptions exist before module startup.
There is no bus worker thread and no shared destructive `get()`.

`Mailbox` contains a deque and a condition variable.
`put()` appends under the condition lock and wakes the consumer.
`take_batch(stop_event, timeout)` waits, swaps the deque, and returns the old contents.
The swap fixes the batch boundary even while producers remain active.
The condition also wakes for termination.

`ShellModule` owns its mailbox, stop event, thread, lifecycle state, and application error callback.
Its constructor takes `module_id`, `bus`, `on_error`, `clock`, and `tick_seconds: float | None`.
`clock` is a callable that returns monotonic seconds.
Its ABC hooks are `accepts(envelope)` and `handle_batch(batch)`.
`run()` starts one non-daemon thread and returns.
`join()` joins that thread.
`step(*, block: bool = False) -> bool` processes at most one batch or one timer tick.
The thread repeats `step(block=True)` until the stop event is set.
Direct tests call `step()` without starting a thread.
Both modes call the same `handle_batch()` implementation.
Read `DIRECT_TESTS.md` for the exact step algorithm and module checks.

Control delivery sets the stop event immediately and wakes the mailbox.
It does not place terminate behind queued data.
The same stop event belongs to the active `WorkflowRuntime`.
The base publishes one `module.terminated` event during normal termination.
Use an idempotent `_finish_termination()` helper in direct steps and the thread exit path.
It calls `on_terminate()` before it publishes the module event.
The base `on_terminate()` does nothing.
On termination, the module abandons queued data.
Termination does not mean pause or automatic restart.

The thread boundary catches exceptions only to notify `Application` and stop its other modules.
Retain the exception and traceback for the main thread to re-raise.
Do not convert a programming error into a successful telemetry result.
The application owns an error queue and a shutdown event.
Terminal stdin uses one daemon reader thread because Windows stdin cannot be cooperatively interrupted.
The application main thread waits for input events or worker errors.
On EOF, Ctrl+C, or worker error, stop and join every module thread.
Re-raise a worker error after those joins.

`ShellModuleWorkflow` extends `ShellModule` with these constructor arguments:

```python
working_target: WorkingTarget
workflow: Path
tools_factory: Callable[[WorkingEnvironment], Mapping[str, Callable]]
runtime_root: Path
ghost: Ghost
```

Its abstract hooks are `prepare_input(batch) -> dict` and `publish_output(report, batch) -> None`.
`handle_batch()` creates one invocation UUID and opens its environment.
It writes `inbox/request.yaml` with `yaml.safe_dump()`.
It builds `WorkflowRuntime` with scoped tools and the module stop event.
It constructs the kernel with the following values:

```python
Kernel(
    manifest_path=self.workflow,
    workspace=self.runtime_root / self.module_id / invocation_id / "workspace",
    task_id=invocation_id,
    task_text=f"Read inbox/request.yaml from runtime.environment ({invocation_id}).",
    base_dir=PROJECT_ROOT,
    coding_agent="python",
    kernel_data_root=self.runtime_root / "kernel",
    run_id=invocation_id,
    resume=False,
    runtime=runtime,
)
```

It calls `kernel.run()` once.
On termination, it skips all output publication.
On another unsuccessful summary, it raises an error with the kernel result.
On success, it reads `outbox/report.yaml` and parses one report.
Before publication, it checks cancellation again.
It then calls the subclass output hook.
Environment creation always precedes scoped tool creation.
There is no registry keyed by thread ID or a global current Ghost.

Each normal manifest uses this structure:

```yaml
---
name: dialogue
driver: {kind: python}
human_resolver: {mode: forbid}
state_policy: {mutation_model: append_only}
roles:
  work:
    skill: workflows/roles/dialogue.py
    result_contract:
      type: json
      schema:
        status: {enum: [done]}
        summary: string
phases:
  - name: work
    kind: role
    role: work
    on_status: {done: stop}
---
```

Use corresponding names for main cognition, inner thought, scheduling, status, and tool-plus-dialogue.
Do not declare `checkpoint_backend`, gates, human phases, or agent repair.
Each role reads `inbox/request.yaml` through `context.runtime.environment`.
It writes the report to `scratch/report.yaml`, then moves it to `outbox/report.yaml`.
It returns `{"status": "done", "summary": "dummy <module> completed"}`.
The report contains semantic output, not the kernel result contract.

All requests contain `invocation_id`, `module_id`, and `messages`.
Each message contains its envelope fields and serialized referenced knoxel under `knoxel`.
Action messages also contain their serialized GWT under `gwt`.
Serialization uses the existing `__type__` and `data` knoxel format.
The adapter resolves references through Ghost before writing the request.
An empty inner-thought request contains an empty messages list.

Use these report shapes:

| Role | Report fields |
| --- | --- |
| MainCogLoop | `summary: str`, `decisions: list[{correlation_id, route, content, task_id, task_kind}]` |
| Dialogue | `replies: list[{correlation_id, content}]` |
| InnerThoughtLoop | `thought: str` |
| ScheduleSubprocesses | `tasks: list[{correlation_id, task_id, task_kind}]` |
| CheckSubprocess | `queries: list[{correlation_id, task_id}]` |
| ToolPlusDialogue | `replies: list[{correlation_id, content}]` |

Main decisions use routes `dialogue`, `schedule`, `status`, and `tool_dialogue`.
Use empty strings for unused task fields.
The fixed main role applies these case-insensitive rules in this order to each `user.input`:

| Input | Decisions |
| --- | --- |
| Starts with `/status ` | One status action for the following task ID |
| Equals `/memory` | One tool-plus-dialogue action |
| Contains `minecraft` | Schedule `task_kind="minecraft"`, then dialogue: `I will schedule your Minecraft request.` |
| Equals `/demo-task` | Schedule `task_kind="dummy"`, then dialogue: `I will schedule a dummy task.` |
| Contains `dumb robot` | Dialogue: `You are a dumb human, Jesse! Dumb and nasty.` |
| Any other text | Dialogue: `I heard you, Jesse.` |

For schedule decisions, use `task_id="task-" + correlation_id`.
For telemetry, thoughts, and public replies, produce no new action.
They still enter the next GWT batch.
The GWT summary is `Dummy cognition processed N inputs.`
The dialogue role returns the action content unchanged.
The thought role returns `I wonder what Jesse will ask next.`
The tool-plus-dialogue role calls `ghost_search(query="Jesse")`.
Its reply is `I found N memories about Jesse.`
Fixed rules are explicit prototype behavior, not intended character policy.

The main output hook creates one cycle for the entire batch.
Under the Ghost lock, increment the tick and add an AttentionFrameKnoxel.
Its focus IDs are the incoming knoxel IDs.
Its content and frame reason use the fixed summary.
Use empty goal and winning-attention lists with `control_mode="reactive"`.
Use the default scope for this prototype.
Create all ActionSchemaKnoxel objects for the decisions under the same lock.
Set `attention_frame_id` on each action.
Map dialogue to `ActionType.Reply`, tool-plus-dialogue to `ActionType.ToolCallAndReply`, and schedule/status to `ActionType.ToolCall`.
Store task fields and route in action metadata.
Create `GwtTurnRecord(tick_id=tick, attention_frame_id=frame.id, knoxel_ids=ids, summary=summary)`.
Its IDs contain the incoming records, frame, and actions.
Store it in `ghost.gwt_turns[tick]` and append a CycleState with the frame and published main slot.
After releasing the lock, publish `gwt.updated`, then the action messages in decision order.

Other output hooks store their Feature before they publish its ID.
Use `FeatureType.Dialogue` for user input and public replies.
Use `FeatureType.InnerMonologue` for thoughts and `FeatureType.WorldEvent` for telemetry.
Set `causal=True` and `override_source` to the producing person or module.
Set `InterlocusType.Public` for dialogue only.
Use the existing undefined visibility default for thoughts and telemetry.
Set correlation identity in metadata.
The main loop references existing Features without storing them again.

MainCogLoop and action modules have no timer.
InnerThoughtLoop has a configurable period with a default of 30 seconds.
ScheduleSubprocesses has a configurable period with a default of 0.1 seconds.
Each timer module uses the supplied clock and its mailbox condition.
Production wiring supplies `time.monotonic`.
Direct tests supply a manually advanced clock.
Do not call `step()` concurrently with that module's thread.

`TaskTable` is one stateful object with an RLock and a dictionary.
Each record contains task ID, kind, correlation ID, status, and next transition time.
The scheduling output hook creates a record only once per task ID.
For a dummy task, publish accepted immediately, running after one scheduler period, and completed after the next period.
Each timed transition emits one WorldEvent Feature and one telemetry envelope.
The scheduler's empty iteration advances this table without another kernel invocation.
The scheduling module overrides the shared `handle_batch()` only for this empty-tick branch.
For a nonempty batch, it calls the shared workflow implementation.
For Minecraft, emit `unsupported` immediately and schedule no transition.
The status output hook reads TaskTable and emits current status or `unknown`.
These host hooks own task state. Python roles only decide requests.
In the scheduler's `on_terminate()`, mark active dummy tasks terminated and emit their final telemetry once.

The Ghost reuse boundary is explicit:

- Copy `pm_core/model/*.py` except `knoxel_list.py` into a local `pm_core/model/` package.
- Copy `pm_core/consts.py` and `pm_core/utils/datetime_utils.py`.
- Copy `pm_core/persist/dataprovider_yaml.py`.
- Copy `pm_core/ghost/ghost_base.py`.
- Create empty package initializer files for this subset.
- Remove the unused `CodeletState` import from the copied `ghost_state.py`.
- Replace the copied BaseGhost LLM annotation with `Any` and remove its runtime LLM import.
- Copy both Spamuel YAML files named in `ADAPTER_NOTES.md` into `data/`.

Implement `ghost.py:Ghost` as a small BaseGhost subclass with no LLM attachment API.
Load the seed with the copied YAML loader.
Provide `Ghost.from_state(state)` for direct test construction.
`Ghost.from_yaml(path)` parses YAML and calls that same constructor.
Construct GhostConfig from the state fields, as the source `Ghost.from_state()` does.
Preserve source knoxel IDs, cycle records, ownership, and index reconstruction.
Initialize the next tick from the maximum cycle or knoxel tick.
Initialize the next ID from the maximum stored ID.
Add `lock: RLock` and `gwt_turns: dict[int, GwtTurnRecord]`.
Use `add_knoxel(..., generate_embedding=False)` for all new records.
Every public memory operation takes the lock.
Copies returned to tools use serialized data, without the private `_owner` object.
Do not deep-copy the owner graph.

`Ghost.create_bash_environment(mode="read")` returns a `GhostBashWrapper` target factory.
Only read mode is required now.
The wrapper also accepts an explicit tuple of visible knoxel IDs or `None` for all records.
That visibility choice comes from application wiring, never from the role.
`open()` snapshots those records and the stored GWT records under the Ghost lock.
It builds a fresh BashMachine with cwd `/work` and a named workflow user.
Create `/work/inbox`, `/work/outbox`, `/work/scratch`, and `/work/ghost`.
Each visible knoxel has `/work/ghost/knoxels/<id>.yaml` in the registered serialization format.
Each GWT whose referenced records are all visible has `/work/ghost/gwt/<tick>.yaml`.
Expose a readable index with visible IDs, types, and content summaries.
Use lazy content providers over serialized snapshot dictionaries for knoxel YAML.
Set explicit `Access.R` on every Ghost file.
Do not place hidden records in this BashMachine at all.

Environment file methods must use the named user's shell, not BashMachine host methods that bypass access rules.
Use `shlex.quote()` for every path and text argument in constructed shell commands.
Use `cat`, `printf`, `mv`, and `find` through `BashMachine.exec()`.
Return stdout without stripping whitespace.
Raise an error for a failed file operation.
Keep `exec()` exit codes available to the caller for permission checks.
Host-only environment setup can use BashMachine host methods.

`FolderTarget(root, bash_executable)` creates environments under `root / invocation_id`.
Its file operations use pathlib and preserve text exactly.
Its `exec()` calls `[bash_executable, "--noprofile", "--norc", "-c", command]` with that environment as cwd.
Resolve Bash through `shutil.which("bash")` in the demonstration.
If Bash is unavailable, report that exact folder-shell check as blocked.
Do not silently replace it with PowerShell or a virtual shell.
The real-folder target is trusted host execution, not a sandbox.

`ghost_mcp.py` builds three tools against the environment snapshot:

```python
ghost_get(knoxel_id: int) -> dict
ghost_search(query: str) -> list[dict]
ghost_gwt(tick_id: int) -> dict
```

Lookup returns the visible serialized record or raises KeyError.
Search uses casefolded substring matching on content, sorted by knoxel ID.
GWT lookup returns only a visible GWT record.
No embeddings, indexes beyond Ghost's existing indexes, or network retrieval are necessary.
`build_tools(environment)` returns those three bound functions by name.
`build_mcp(environment)` registers the same functions on a FastMCP server.
The workflow constructor accepts `tools_factory=build_tools`.
The explicit factory replaces the pseudocode `mcps` list because each invocation needs its own visibility snapshot.
Use `fastmcp.Client(server)` with `async with` for the real MCP demonstration call.
That installed client already supports an in-process FastMCP server.

`build_app()` constructs Ghost, bus, TaskTable, and all seven modules in one place.
It accepts `seed_path`, `runtime_root`, `thought_period`, `task_period`, `clock`, and `emit_reply`.
Use `clock=time.monotonic` and `emit_reply=print` as application defaults.
It registers subscriptions but starts no threads.
It returns Application with module references for the demonstration.
`Application.run()` starts the module threads.
Direct tests never call that method or `ShellModule.run()`.
`python main.py` starts the terminal session.
`python demo.py` runs the finite scenarios and exits.
The banner states `DUMMY COGNITION — no LLM calls`.
The printed task acceptance includes its task ID for `/status`.

Implement these checks in order:

1. Load the full Spamuel seed and verify unique IDs and preserved knoxel types.
2. Run a virtual Bash read and a rejected Ghost-file write.
3. Add a Feature through Ghost and verify that the next environment sees it.
4. Build a restricted environment and verify that Bash and MCP cannot retrieve an omitted record.
5. Run a real one-role kernel that reads virtual input and writes virtual output.
6. Verify that the default kernel behavior still works without runtime.
7. Verify that two subscribers receive one published envelope.
8. Queue two user inputs before MainCogLoop starts and verify one GWT with two dialogue actions.
9. Verify both public replies retain their separate correlations.
10. Run `/demo-task`, query its ID, and observe ordered task telemetry in later GWT records.
11. Run `/memory` and compare the reply count with the actual scoped search result.
12. Produce one thought through the normal workflow and verify that it never prints as dialogue.
13. Terminate an idle module and join its thread.
14. Terminate a real Python role that waits on `stop_event`, then calls `check_cancelled()`.
15. Verify one termination journal entry, no success output, and no retry.
16. Raise a deliberate ValueError in a real Python role and verify traceback propagation with the flag enabled.
17. Verify legacy exception conversion with the flag disabled.
18. Run the real-folder file and Bash checks in a temporary directory.

Use events and message receipts to coordinate checks, not timing guesses.
Keep cancellation and error roles in temporary files created by the demonstration or existing test fixtures.
Use existing pytest infrastructure only inside `pm-workflows`.
Run its affected driver, kernel, and new runtime checks.
Use standard-library unittest for the direct tests in `DIRECT_TESTS.md`.
Use assertions in `demo.py` for the application integration run.
Record actual commands and results in the README.
Do not claim that this specification itself verifies runtime behavior.
