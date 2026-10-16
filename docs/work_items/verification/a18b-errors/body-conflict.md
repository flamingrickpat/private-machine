# Overlapping projects and the body lock

## Summary

`minecraft_frustration_and_choice` started multiple projects inside one application.
One actor acquired the body. A later actor received `BodyBusy` before its first body call.
The lock preserved exclusive ownership.
The failure involved admission and error isolation: the second project started without a body queue, and its conflict stopped the application.

Parent: [A18B error analysis](../a18b-errors-20261016.md).
Related: [managed runs](../../../subsystems/managed-work.md).

## Observed overlap

The case ran for `6987.484` seconds and published ten companion replies.
Its managed task directory contains three independent Minecraft projects:

| Run prefix | Request | Terminal state after cleanup |
| --- | --- | --- |
| `57935d00` | Punch trees and build or destroy things with Jesse. | Cancelled after the application failed. |
| `2ec17745` | Join Jesse near a spruce and wait before construction. | Cancelled after the application failed. |
| `389f2b69` | Join Jesse near the same spruce after another user message. | Failed with `BodyBusy`. |

All three have distinct action IDs and parent correlations.
This was not the same worker launched twice under one identity.

The first project's actor journal records epoch `208` for owner
`57935d00c38642e492879e1f3de65c9c-actor-0001`.
Its acquisition time was approximately 09:06:23 UTC.
Its last recorded body result was approximately 09:14:00 UTC.
The later project's actor attempt began approximately 09:21:11 UTC.
It failed at lease acquisition and produced no actor body journal.

The occupied lock is confirmed by the exception.
**Strong inference:** the earlier actor still held it through its native session.
That actor is the only recorded lease acquisition in this case, and its project remained active until application shutdown.
The exception itself does not identify the lock holder or retain a point-in-time owner snapshot.
That missing diagnostic limits absolute attribution.

## Logic that allowed overlap

| Source at `f7eb226` | Current logic and consequence |
| --- | --- |
| [dev/work_items/a18b/cli.py:220](https://github.com/flamingrickpat/pm_next_v2/blob/f7eb2268f3eff4c9732e76b4b1b48e241d4a4065/dev/work_items/a18b/cli.py#L220) | The scheduler admits at most one Minecraft **case** per batch. It does not serialize projects within that case. |
| [workflows/main_cognition/host.py:794](https://github.com/flamingrickpat/pm_next_v2/blob/f7eb2268f3eff4c9732e76b4b1b48e241d4a4065/workflows/main_cognition/host.py#L794) | `_active_game_request` prevents repeated starts for the same active request. |
| [workflows/main_cognition/host.py:807](https://github.com/flamingrickpat/pm_next_v2/blob/f7eb2268f3eff4c9732e76b4b1b48e241d4a4065/workflows/main_cognition/host.py#L807) | A batch containing new `user_input` bypasses that guard. A new correlation can also differ from the older request. |
| [pm/runs/supervisor.py:208](https://github.com/flamingrickpat/pm_next_v2/blob/f7eb2268f3eff4c9732e76b4b1b48e241d4a4065/pm/runs/supervisor.py#L208) | `start` deduplicates by `action_id`. Distinct accepted actions each launch a worker. There is no body-key admission queue here. |
| [workflows/minecraft_project/planning/dispatch.py:77](https://github.com/flamingrickpat/pm_next_v2/blob/f7eb2268f3eff4c9732e76b4b1b48e241d4a4065/workflows/minecraft_project/planning/dispatch.py#L77) | Each leaf receives a lease factory for its own child identity. |
| [workflows/minecraft/specialist_session.py:102](https://github.com/flamingrickpat/pm_next_v2/blob/f7eb2268f3eff4c9732e76b4b1b48e241d4a4065/workflows/minecraft/specialist_session.py#L102) | The actor acquires the body before its initial observation. It retains ownership through the native session and cleanup. |
| [workflows/minecraft_body/lease.py:64](https://github.com/flamingrickpat/pm_next_v2/blob/f7eb2268f3eff4c9732e76b4b1b48e241d4a4065/workflows/minecraft_body/lease.py#L64) | Ownership uses `ProcessLock.acquire()` with the default `wait=False`. |
| [pm/host/durable_files.py:89](https://github.com/flamingrickpat/pm_next_v2/blob/f7eb2268f3eff4c9732e76b4b1b48e241d4a4065/pm/host/durable_files.py#L89) | A busy operating-system lock raises `BodyBusy` immediately. |
| [pm/runs/supervisor.py:331](https://github.com/flamingrickpat/pm_next_v2/blob/f7eb2268f3eff4c9732e76b4b1b48e241d4a4065/pm/runs/supervisor.py#L331) and `:353` | The supervisor records failure, then forwards unexpected errors to the application. Only cancellation and `TaskServiceError` avoid that fatal path. |
| [pm/host/durable_files.py:15](https://github.com/flamingrickpat/pm_next_v2/blob/f7eb2268f3eff4c9732e76b4b1b48e241d4a4065/pm/host/durable_files.py#L15) | `BodyBusy` is a plain `RuntimeError`, not a `TaskServiceError`. |

The body command lock already supports cancellable waiting at [workflows/minecraft_body/lease.py:54](https://github.com/flamingrickpat/pm_next_v2/blob/f7eb2268f3eff4c9732e76b4b1b48e241d4a4065/workflows/minecraft_body/lease.py#L54).
That command lock differs from long-lived body ownership.
It does not queue a second actor's owner lease.

There is no evidence of a stale lock file causing this incident.
The lock uses an operating-system byte lock, which closes with its handle or process.
Lock-file removal bypasses ownership.
No `GhostConflict` revision error appears as this case's terminal cause.

## Proposed repair

Body admission needs one explicit policy before a second actor starts.
An owner can receive an instruction update, a second task can queue, or an explicit replacement can cancel and join the prior owner.
Replacement must finish external cleanup before new ownership begins.
The operating-system fence must remain in place for other processes and uncertain external outcomes.

A busy body needs an expected task disposition with owner and request evidence.
It must not become a global application failure merely because two valid requests overlap.
The public explanation can use that disposition without claiming that any world action ran.

Evidence:
`.runtime/a18b-current/fresh/minecraft_frustration_and_choice/failure.txt`,
`runtime/episode-01/managed_tasks/*/control.json`, and the first actor's `scratch/actor-journal.jsonl`.

Code:
[`pm.runs.supervisor.ManagedRunSupervisor.start`][],
[`workflows.minecraft_body.lease.BodyLeases.acquire`][],
[`pm.host.durable_files.ProcessLock.acquire`][],
[`workflows.minecraft.specialist_session.run_specialist_session`][].
