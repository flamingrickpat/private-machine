# Singleton managed workflows

## Summary

A singleton permits one active managed run for its configured key.
A conflicting start returns an explicit rejection and publishes its owner to cognition.
The companion chooses whether to wait, update, or terminate that owner.
Cancellation retains admission until worker cleanup finishes.

Parent: [managed work](managed-work.md).
Related: [body leases](body-leases.md), [`workflows.self_management.host.SelfManagement`][].

## Configuration

The owning API is [`pm.runs.workflow_spec.WorkflowSpec`][].
The setting belongs to PM's managed-run configuration, rather than the shared workflow kernel.
An ordinary workflow retains parallel execution by default.

```python
WorkflowSpec(manifest, singleton=True)
```

An empty `singleton_key` uses the configured workflow name.
Names that share one resource can use the same explicit key:

```python
WorkflowSpec(manifest, singleton=True, singleton_key="minecraft_body")
```

`workflows/companion.py` applies this configuration to both `minecraft` and `minecraft_project`.
A project can still invoke its own child actors.
Those children belong to the same managed run and do not request another supervisor admission.
The body lease independently fences their actual world calls.

## Start, rejection, and feedback

[`pm.runs.supervisor.ManagedRunSupervisor.start`][] checks admission under its mutex.
The accepted receipt reserves the key before the worker thread starts.
Concurrent start requests therefore cannot both pass that check in one supervisor.
An active `accepted`, `running`, or `waiting_for_input` receipt occupies its singleton key.

A rejected request retains its own receipt with these fields:

| Field | Meaning |
| --- | --- |
| `state: failed` | The requested start did not run |
| `singleton_key` | The occupied admission key |
| `blocked_by_run_id` | The existing managed run that occupies the key |
| `latest_event` | The reason and available management choices |
| `artifact_path: ""` | No second worker or task directory was created |

The supervisor publishes a durable `workflow.event` with `event: start_rejected` and `reason: singleton_busy`.
Its detail names the owner. Its correlation belongs to the rejected request.
Main cognition materializes that event as workflow evidence for its next cycle.
The event is not user dialogue or a public error message.

The direct `minecraft` and `minecraft_project` action aliases still enter [`workflows.self_management.host.SelfManagement`][].
That module also accepts explicit `self_management` actions.
Both routes call the same [`pm.runs.management_api.build_management_tools`][] operations.
The existing receipt path reports a rejected operation without inventing a successful start.

Repeated delivery of the same action ID returns the same receipt.
It does not emit another busy event or retry the rejected request after the key becomes free.
A later attempt needs a new action ID.

## Cancellation and recovery

[`pm.runs.supervisor.ManagedRunSupervisor.terminate`][] records cancellation before the worker finishes external cleanup.
Admission checks the owned worker lock as well as the durable run state.
A cancelled worker therefore retains its slot during cleanup.
The companion can start replacement work after cleanup releases that slot.
The supervisor does not queue a replacement or terminate an owner automatically.

Explicit `resume` checks the same singleton key before it accepts a new worker epoch.
A conflicting resume publishes `resume_rejected` with the owner and raises an expected management error.
It does not advance the rejected run's execution epoch.

## Other processes and late contention

Singleton admission belongs to one supervisor and its managed-run table.
It does not replace the operating-system [body fence](body-leases.md).
Another process can still own the configured Minecraft endpoint.
A late `BodyBusy` ends only the requesting managed task and publishes its failure detail.
The application remains available for the next cognitive choice.
Programming errors and failures to record an outcome still reach the application error handler.

## Evidence

The boundary tests replay [captured overlapping starts](../work_items/verification/a18b-singleton.md) through real local kernels and locks.
The separate real-service gate uses the full canon, the real Minecraft server, and one real cognitive cycle.
Neither control-boundary success nor a model's chosen action establishes complete A18B acceptance.
