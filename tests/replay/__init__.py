"""Deterministic replay of one real cognitive situation.

The harness answers one question cheaply: *given the exact same situation and
the exact same model answers, what does a change to the architecture do?*

A **recording** runs the real modules against a real endpoint once per sample
and stores every completion and context measurement in call order, beside a
copy of the memory checkpoint the run started from. A **replay** loads that
checkpoint and drives the same modules again, serving the recorded answers
instead of calling a model. Nothing else is substituted: the store, the write
batches, the control fences, the kernels, and the child workflows are the real
ones.

Why this exists: the end-to-end gates in ``tests/live`` cost provider money,
need a reachable endpoint, and take minutes. They answer "does the integrated
system work at all". They cannot answer "did this control change preserve the
private inquiry across one ordinary message" without being re-run from scratch
for every edit. A recording is paid for once and replayed in seconds.

A recording is also a **sample**, not a ground truth. The model is stochastic,
so ``record --samples N`` captures N independent continuations of the same
situation and ``replay`` reports what each one does under the changed code.
That range is the point: a change that only works for one lucky sample is not
working.

Reading map:

- :mod:`tests.replay.script` owns the recorded call list and divergence checks.
- :mod:`tests.replay.model` binds that list to the one completion seam every
  workflow uses.
- :mod:`tests.replay.runtime` owns the store, clock, modules, and the
  deterministic step order.
- :mod:`tests.replay.situations` builds the starting situation for a scenario.
- :mod:`tests.replay.scenarios` holds the concrete scenarios.
- :mod:`tests.replay.__main__` is the ``record``/``replay``/``compare`` CLI.

Deliberate limits:

- Replay proves behaviour, not model quality. A recorded answer is evidence of
  what the model said once, never of what it says now.
- A recording is only valid for the prompts that produced it. When a change
  alters a prompt the replay stops with a :class:`ReplayDivergence` naming the
  first differing call; that is a signal to re-record, not to relax the check.
- Replay serves the recorded context measurement, so it cannot discover that a
  newly lengthened prompt no longer fits the context window. Re-record to check
  that.
- Only the modules a scenario steps are exercised. A scenario that never steps
  main cognition says nothing about it.
"""
