# Configured inference, scaled simulation time, and in-memory cognition

Owner-selected implementation, 2026-11-16. Preserve the earlier immersion repairs.

## Current path and problem

`LazyStageModel` has a four-slot semaphore. `JsonClient.request` then enters
`InferenceAdmission`, which permits one chat request per endpoint. Worker pools
cannot overlap their HTTP calls. Critical reservations also exclude ordinary work.

Each main-cognition kernel phase parses `CycleWork` from an in-memory file,
executes one Python function, and serializes the entire object again. This is
temporary cycle state, not a reload of the Ghost checkpoint.

Simulation time advances by monotonic wall seconds. Actor delays also use wall
seconds, so changing only the clock would change the actor's simulated cadence.

## Selected changes and contracts

- `Connection.max_concurrent_requests: int = 8` controls shared endpoint capacity.
  Values <= 0 or >= 9999 mean unlimited. Each free slot serves queued triage,
  critical, interaction, then ordinary requests in FIFO order within priority.
  Remove the duplicate client semaphore, `CHAT_CONCURRENCY`, and cross-phase
  exclusive reservations. Cancellation still releases owned slots.
- `Connection.request_timeout_seconds: float | None = None` controls HTTP read,
  write, and pool waits. Set the DeepSeek configuration to 99999 seconds and
  `override_context: 65536`. No environment switches own these settings.
- `SimulationTemplate.time_multiplier: float = 1.0` is persisted in run state.
  Virtual elapsed seconds equal wall elapsed seconds times this multiplier.
  Actor delay constants remain simulated seconds. CLI and browser creation expose
  the setting. Existing runs without the field retain speed 1.
- Main cognition remains an actual kernel invocation. Its one Python role validates
  `CycleRequest` once and runs the existing ordered phase functions against one
  `CycleWork`. Intermediate state is neither serialized nor recoverable. Preserve
  final reports, phase receipts, host-owned atomic memory acceptance, cancellation,
  and failure tracebacks. Interrupted cycles restart from accepted Ghost state.

No module moves, dependency changes, alternate memory store, or concurrent writes
to shared cycle state. Serialization remains at actual file/model boundaries.

## Execution and verification

Implement scheduler/config, then clock scaling, then the phase handoff. Update
affected docs and narrow deterministic contracts. Use the configured real endpoint
to measure overlapping calls, a real cognition kernel to exercise the new handoff,
and real participant clock advancement where feasible. No fake model/server passes.
Inspect actual prompt-selection budgets at 64K. Run configured static checking and
the strict manual build. Preserve evidence under `.runtime/inference-clock-repair/`.

Risks: endpoint rate limits, genuine cross-phase dependencies, stale async sources,
and server timeouts while requests wait remotely. Existing commit/source checks
remain responsible for accepted state. The active month attempt uses frozen source
and configuration; editing this checkout does not hot-reload that attempt.

## Implementation and handoff (2026-11-16)

Owner clarified that a consistent restart takes precedence over retaining recent
messages. No mid-cycle resume is wanted. Owner also requested stopping all PM
processes when finished. Do not restart a long run automatically after this handoff.

### Implemented

- `pm/llm/config.py`: per-connection YAML `max_concurrent_requests` and
  `request_timeout_seconds`. Both DeepSeek entries in ignored `config.yaml` use
  8, 99999, and existing `override_context: 65536`. Example YAML exposes them.
- `pm/llm/concurrency.py`: multiple active direct HTTP requests. Queued triage,
  critical, interaction, then ordinary work take free slots. <=0 or >=9999 means
  unlimited. Removed whole-workflow reservations and the second semaphore in
  `LazyStageModel`; removed `CHAT_CONCURRENCY` and its assembly wiring.
- Direct `JsonClient` HTTP read/write/pool timeout comes from model YAML. Connect
  timeout remains 10 seconds. Native pm-coder sessions retain their own transport
  and session deadlines; these are not globally routed through JsonClient.
- `simulation/templates.py`, `plans.py`, `engine.py`, `server.py`, and setup UI:
  `time_multiplier` defaults to 1 and persists in controller state. CLI supports
  `--time-multiplier`. Clock and actor-delay elapsed time scale together. Scheduled
  events that are already due execute without attempting to rewind the clock.
- Main cognition now has one kernel role. `steps.run_selected_phases` passes one
  `CycleWork` through the existing ordered phase functions. Removed intermediate
  JSON helpers and files. Final report and phase receipts remain.
- `MemoryStore.commit` accepts in memory, without constructing/writing a full
  checkpoint on every small update. Completed main cycles explicitly save before
  releasing actions. Explicit saves, shutdown, and exports still write pickle.
- `pm/memory/restart.py`: ordinary checkpoint loads close pending exchanges,
  discard unpublished replies, cancel pending actions and workers, and suppress
  old task event/public-reply obligations. Records remain historical evidence.
  Disposition uses the accepted companion identity as operational authority.
- Startup no longer invokes parked-workflow, reply, interaction, main-action,
  private-thought, or focus replay. Already resolved Critical input cannot become
  a new outstanding Critical obligation. Raw inspection bypasses disposition.
- Clean export applies disposition to a separate output and preserves the source.
- Managed-run serialized fields now have a named `ManagedRunData` contract.

### Evidence actually obtained

Everything is under `.runtime/inference-clock-repair/`.

- `parallel.log` shows 8 admitted requests before releases, against configured
  OpenRouter DeepSeek. `parallel-result.json`: eight real completions, zero replay.
- `full-cycle/result.json`: actual companion, real kernel/models/embeddings/native
  codelets; accepted tick 808 after 299.906 seconds; checkpoint 82,111,839 bytes;
  zero intermediate cycle files. Journal has one `cycle` role and stop receipt.
- `dialogue-64k/`: actual retrieved persona evidence, generation and review accepted
  in one attempt. Estimated input 16,454 <= 52,428 (80% of 65,536). This demonstrates
  fitting, not that every possible mandatory context can fit every small window.
- `clock-result.json`: real monotonic clock and real persisted controller file,
  isolated without participants: 2x produced 4 simulated seconds in 2.016 wall
  seconds; 0.5x produced 1 simulated second in 2 wall seconds. No fabricated server.
- `focused-final.log`: 30 narrow admission, checkpoint, causal recovery, and template
  checks passed in 0.538 seconds. No full suite. Old causal tests now explicitly
  save forensic fixtures instead of expecting every memory commit to persist.
- `types-final.log`: configured strict Pyright, zero errors. `types-new.log` also
  checks the new restart/phase/scheduler code explicitly with zero errors.
  Broad strict checking of previously excluded legacy engine/config code still
  reports existing errors; no checker setting was weakened.
- JavaScript syntax check and `git diff --check` passed.
- An earlier strict documentation build passed. A final clean strict build covers
  the last Ghost documentation corrections; inspect `docs-final.exit` and
  `docs-final.log` for its final outcome.
- The PowerShell wrappers for full-cycle and dialogue scripts reported exit 1
  despite their completed result artifacts and success messages. Native child
  stderr was redirected through PowerShell; the underlying Python exit was not
  separately captured. Do not mislabel those wrapper exits as zero.

### Remaining checks / cautions

1. Confirm final docs build and sample current HTML/local links. No visual browser
   inspection has been performed. Canonical pages changed: `docs/ghost.md`, models,
   cognition/state, inference settings, public/recovery publication, simulation,
   and README. This work changes persistence and restart semantics intentionally.
2. No new multi-hour scenario was run. The previous month run uses frozen older
   source/config. Owner requested all PM processes stopped, not another live run.
3. Real remote concurrency passed; a queued local llama.cpp server at 99999-second
   timeout was not exercised. Native pm-coder transport/deadlines remain separate.
4. No actual crash was injected into a running full cognition process. Real file
   checks establish loss of unsaved messages, retirement of saved executable
   leftovers, clean export, and source preservation. Full cycle ran the real stack.
5. `save_parked`, explicit task resume, and legacy replay helper definitions still
   exist for callers, but companion startup no longer invokes them. Do not restore
   automatic replay as a generic reliability improvement.
6. No commit or push. Preserve prior immersion changes and the owner's `.gitignore`.
   Do not mark beta cards done from this handoff.

Next useful action, if requested: review the diff with the owner, then create a
fresh comparison using the chosen multiplier and new config. Never reuse the old
stopped checkpoint as a clean comparison.


### Final handoff status

The final clean strict documentation build completed in 204.97 seconds and exited
0 (`docs-final.exit`). Six changed HTML pages and their local link targets were
checked programmatically. Visual browser inspection remains unperformed.
Configured strict Pyright and 30 focused checks passed; no full suite ran.
All identified running PM processes and their descendants were stopped as requested.
A final process enumeration found zero matching PM processes. The simulation
browser is also stopped. See `stopped-processes.json` for the final sweep.
An initial stop sweep encountered an already-exited child after killing its parent;
the second sweep completed and verified the resulting process state.

No long test was restarted. No commit or push was made. Implementation is present
in the working tree; the remaining limitations above are verification scope, not
permission requests. In particular, native pm-coder transport is separate from the
direct HTTP pool, and the clock check did not include participant processes.
