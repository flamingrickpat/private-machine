# Recorded scenario replay

## Summary

A recorded scenario replays one real cognitive situation without paying for the model again.
You record it once against the configured endpoint, then replay it as often as you like.
The replay uses the same classes, the same memory store, the same kernels, and the same control fences as the running application.
Only the model answers come from the recording instead of from a provider.

This answers a question the long live gates cannot answer cheaply: *did this change alter what the architecture does with the same evidence and the same model answers?*
It does not answer whether the model is good, and it does not replace the integrated live gates at the end of a milestone.

The canonical owner is `tests.replay`; start at its package docstring.
The private-loop behaviour it was introduced for is described in [the private thought loop](../subsystems/private-loop.md#what-supersedes-private-work).

## Why it exists

Live gates cost provider money and take minutes.
Editing one control fence and re-running a live gate for every iteration would spend the milestone budget on checking that a loop still terminates.
Resampling the model is also a poor comparison: a difference between two live runs could come from the code or from the model drawing a different sentence.

A recording fixes the model, so a difference between two replays comes from the code.
A recording is also a *sample*: the model is stochastic, so several recordings of the same situation capture several continuations, and a change that only works for one lucky sample is not working.

## What a recording holds

Each sample directory holds:

| File | Content |
| --- | --- |
| `calls.jsonl` | Every model request in call order, with the exact request text and the recorded answer |
| `ghost.pickle` | The starting memory store, written before the run and never written again |
| `situation.pickle` | The store at the end of the run, useful for inspection |
| `manifest.json` | Scenario, sample, model, endpoint, revision, and whether the trace is complete |
| `outcome.json` | The typed scenario outcome: what the run actually did |
| `runtime/` | The kernel artifacts: prompts, reports, and repair samples |

A replay writes `replay-<sample>.json`, `replay-outcome-<sample>.json`, and `replay-<sample>.pickle` beside them and never modifies the recording.

## Commands

```powershell
python -m tests.replay list
python -m tests.replay record --scenario private-continuity --samples 3
python -m tests.replay replay --snapshot .runtime/replay/private-continuity-20261113T211905Z
python -m tests.replay compare --snapshot .runtime/replay/private-continuity-20261113T211905Z
```

`record` spends provider money: every sample is one independent live run.
`replay` spends nothing and needs no reachable provider; it serves the recorded answers over a copy of the recorded checkpoint.

The harness reruns itself once under `PYTHONHASHSEED=0`.
Python randomizes string hashing per process, so a set of names prints in a different order in each run, and some of that text reaches prompts.
A replay can only compare identical prompt text if the search order is fixed.

## What determinism requires

A replay must ask the recording's questions, in its order, or the comparison means nothing.
`tests.replay.script.ReplayDivergence` stops at the first difference and reports both texts, so a change that alters a prompt is a loud result rather than a silent cache miss.

Three things made that possible, and each is worth knowing before you write a scenario:

- **Identity is repeatable.** `tests.replay.determinism.deterministic_identities` replaces the process identity source with a seeded counter for the duration of one run. Invocation ids and commit ids reach prompts, so they must repeat.
- **Measured durations are normalized.** `tests.replay.script.normalize_timing` replaces accumulated wall-clock seconds with a placeholder. That number is a receipt of how long the run took, never evidence the model reasons about.
- **The virtual clock starts where the recording started.** The manifest retains it, because every record timestamp, opportunity age, and deadline reads it.

## Deliberate limits

- **Replay proves behaviour, not model quality.** A recorded answer is evidence of what the model said once.
- **A recording is valid only for the prompts that produced it.** A changed prompt stops the replay; re-record to sample the new code.
- **Replay serves the recorded context measurement.** It cannot discover that a newly lengthened prompt no longer fits the context window. A recording does hit the real budget check, so re-record when prompts grow.
- **An aborted invocation cannot be replayed.** A cancellation races the model call, so the recording is a partial trace. The manifest marks such a sample and replay reports `NotReplayable` instead of blaming the current code.
- **Only the modules a scenario steps are exercised.** A scenario that never steps main cognition says nothing about it.

## Adding a scenario

A scenario is a small script over `tests.replay.runtime.ScenarioRuntime`.
It states its own starting situation, drives the real modules in an explicit order, and returns a typed outcome whose `summary` is what a replay compares.

The runtime starts no threads.
It calls `BusModule.step(block=False)` in an order the caller states and advances the virtual clock when it wants a timer tick, so the production module graph runs in one thread with no wall-clock scheduling.

When a scenario has to deliver outside input *while a module is thinking*, it installs `ScriptedModel.call_hook`.
That hook fires inside the model call, which is where the running application would also be.
If such an injection cancels work, say so in the outcome's `replay_blockers`.

## Related material

- [The private thought loop](../subsystems/private-loop.md) explains what the current private scenarios exercise.
- [Prompt log](tool-call.md) records individual live prompts beside a run; a recording serves a similar purpose for repeated comparison.
- The <a href="../../dev/work_items/beta-b05.md">B05 engineering record</a> holds this harness's first results and their limits.
