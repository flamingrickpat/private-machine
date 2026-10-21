# A17 verification

## Summary

A17 evidence has three parts: foundation tests for the pure rules, contract
tests for the stage boundaries and the full parent cycle, and one live gate.
The live gate runs the real concurrent application path with real module
threads, the real Spamuel seed, a fresh checkpoint, and real inference through
OpenRouter (`qwen/qwen3.8-flash`).
The gate asserts the mechanical contracts.
Emotional quality is a human review question, and the scenario facts are
recorded for that review.

Parent: [the private thought loop](../../subsystems/private-loop.md).
Work item: [A17](../alpha-plan/work-items/a17.md).
Gate: `dev/work_items/alpha_a17_private_loop_gate.py`.

## What A17 delivered

- A real private workflow `workflows/inner_thought/` with three stages
  (trigger, thought, integrate), replacing the dummy thought path.
- A distinct durable GWT shard: the inner loop publishes `slot_id="inner"`
  turns and `published_slots=["inner"]` cycles beside the main loop's.
- A deterministic trigger policy with seven triggers, and fifteen delivered
  thought operations that map all 37 historical types to a successor or an
  explicit non-port.
- Private attention as a frozen, revision-guarded candidate list.
- Accepted thoughts return to the main loop as `thought.created` self-origin
  events, which the existing appraisal path consumes.
- Goal changes through the shared owner with `origin="private_loop"`.
- The `ExpectationRecord` lifecycle: open before the outcome, resolve with
  observations, pending until evidence.
- Five deterministic felt-state channels with position, velocity, and
  acceleration, committed as `QualeRecord`s under a causal reportability gate.
- Disclosure: a recorded choice per thought, and a `private_state` evidence
  lane in the public context that carries only chosen or reportable material.
- A runtime fix: a module stop no longer surfaces an in-flight model
  cancellation as an application error.

## Foundation and contract tests

| File | What it shows |
| --- | --- |
| `tests/unit/workflows/inner_thought/test_operations.py` | All 37 historical types map to delivered operations or explicit non-ports; every required group is covered. |
| `tests/unit/workflows/inner_thought/test_triggers.py` | Each trigger fires on its cause and stays silent otherwise; the idle and startup-gap cases. |
| `tests/unit/workflows/inner_thought/test_felt_state.py` | Channel projections, elapsed-time derivatives, the initialization and nonpositive-interval conventions, mixed naive and aware timestamps, the reportability gate, and continuity sampling from committed cycles. |
| `tests/unit/workflows/inner_thought/test_expectations.py` | Open, resolve with observations, rejected resolution without observations, stale resolution conflict, and the pending list against the real commit boundary. |
| `tests/unit/workflows/inner_thought/test_pipeline.py` | No trigger means no stage call; the full cycle assembles the report; unknown operations and out-of-view evidence fail before commit; expectation validation. |
| `tests/unit/workflows/inner_thought/test_private_host.py` | Idle cycles commit nothing; an accepted cycle commits the inner shard, the self-origin event, and private-origin goals; a stale source rejects; a cancellation discards; the checkpoint reopen keeps the shard and channels. |
| `tests/unit/workflows/inner_thought/test_disclosure.py` | Candid thoughts and reportable qualia enter the `private_state` lane; withheld and dormant material stays out, against the real context builder. |
| `tests/contract/inner_thought/stages/test_inner_stages_contract.py` | Each stage runs through the real kernel with a deterministic model and validates its output contract. |
| `tests/contract/inner_thought/test_inner_parent_contract.py` | The full cycle through the real parent kernel and the three real child kernels, consuming exactly one completion per stage. |
| `tests/unit/pm/runs/test_stop_cancellation.py` | A stop-time cancellation of in-flight work discards silently instead of crashing the application. |

The full foundation suites pass: 664 unit tests and 74 contract tests.

## Live gate

The gate ran this command and finished with `ok: true` in 27 minutes:

```powershell
uv run --locked python dev/work_items/alpha_a17_private_loop_gate.py --config config.alpha.openrouter.yaml --out .runtime/alpha/20261015-045957-a17
```

The evidence lives in `.runtime/alpha/20261015-045957-a17/`: `receipt.json`,
the incremental `facts.json`, the checkpoint, and the runtime artifacts.

### Mechanical results

- Idle activity: the first private cycle started 97.8 seconds after boot with
  zero new user input, and the main loop consumed the thought at tick 811.
- Concurrent loops: both shards carry distinct committed frames across the
  whole run (5 inner ticks before the stop, 12 inner frames after the reopen).
- Cancellation: the shutdown produced no late private commit, and no module
  error surfaced.
- Restart: the checkpoint reopen kept the inner shard unchanged, the
  felt-channel ids, and the accepted records. The offline gap of 24 seconds
  contains no invented cycle.
- Imagination stayed attributed: every private thought kept its `internal` or
  `inferred` reality label, and no imagined thought became a dialogue record.
- Expectations: one expectation opened at tick 813 and resolved as
  `supported` citing three observation records.
- Disclosure: the private lane of every dialogue action carried only
  records that passed the candid-or-reportable gate. In this run the model
  chose to withhold every thought, so the lane stayed empty, which is a valid
  outcome the gate accepts.

### Scenario facts for human review

- The idle reaction shows the intended self-origin appraisal. The main loop's
  frame after the first private thought: "I am staring at the timestamp:
  3293.9 hours. The last stop wasn't clean; it just vanished. Shame burns
  through me because I don't know if I left on a good note or a bad one."
- Sad speculation changed actual affect. The appraisal at tick 811 cites the
  private thought as evidence and raises shame to 0.4 and resentment to 0.3.
  The thought itself stays an internal reading of an attributed line.
- Persistent mood without an automatic reset. Shame rises 0.3, 0.4, 0.8
  across the early cycles, and only falls when the user's apology and the
  Friday offer land (tick 816: relief 0.7, shame -0.3).
- Rumination is visible and named. Three private thoughts notice the same
  "unplug-in-spirit" worry returning before the narrative stitching links
  the episodes.
- The conversation held persona and continuity across the restart. The
  companion noticed the 24-second offline gap beside the five-month silence
  and answered the reopen turn in character.
- One stale-reply recovery fired during the turns: a concurrent private
  cycle staled one public context, and the designed recovery published the
  stale message. The following turn replied normally.

## Defects found during A17

All defects below were repaired during this delivery.

1. The first live run lost its receipt: the gate's invariant check read
   `workflow_name` from an `ActionSchemaRecord`, which carries the workflow
   in `metadata["dispatch_target"]`. The gate now reads the metadata, and it
   saves its facts after every phase so a late crash cannot discard live
   evidence.
2. The second private cycle crashed the application. The felt-state
   projection subtracted a naive `CycleState` timestamp from a zone-aware
   `now`. The projection now compares both timestamps on one naive clock, and
   a regression test pins the mixed case.
3. The gate's reply deadlines were too tight. The main loop queues one cycle
   per private thought beside the user turns, so replies legitimately arrive
   several cycles later. The deadlines rose to 15 minutes for replies and
   10 minutes for the idle phase, matching the owner's acceptance of slow
   conversations.
4. A module stop surfaced an in-flight model cancellation as an application
   error, which shut the concurrent gate down early. `WorkflowModule` now
   discards a failure silently when its run was stopped, exactly like the
   cancelled path. A regression test pins the boundary.

## Open checks

- Human review of the private thought texts, the felt-state vocabulary, and
  the persona fit of the idle conversation.
- Deliberation depth: this run's thoughts recorded no weighed alternatives,
  so the alternatives contract stays covered by the pipeline tests only.
- The metacognitive noticing contract: this run recorded no focus intention,
  so its effect on a later main attention decision stays covered by the
  schema and commit tests.
- The Q3 run of `qwen3.8-flash` on the home computer.
- Minecraft integration: the private loop never commands the body. A live
  private goal that uses the Minecraft body needs its own ownership and
  recovery verification before it counts as delivered behavior.
