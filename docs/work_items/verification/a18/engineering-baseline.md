# A18 engineering baseline

## Summary

This page preserves the engineering delivery and smoke evidence before the 2026-10-15 live gate.
The 693-test count is historical.

Parent: [A18 verification](../a18-self-model.md).

## What A18 delivered

- `pm/introspection/grants.py`: an explicit source grant and a read
  inspector with digests, repository and configuration revisions, path
  checks, credential exclusion, size limits, and cancellation.
- `pm/introspection/capabilities.py`: a per-state capability overlay that
  keeps intended, implemented, enabled, observed, unavailable, and uncertain
  evidence apart, with a stable revision and a staleness check.
- `pm/introspection/episodes.py`: a read-side episode view over GWT turns,
  attention frames, private thoughts, action schemas, and managed-run
  receipts, with explicit causal links and the nine inspector questions.
- `pm/introspection/self_model.py`: revisable capability and narrative
  beliefs over `BeliefRecord` and `BeliefShiftRecord`, with supersede and
  staleness history.
- `workflows/introspection/snapshot.py`: the live overlay builder from the
  real repository, active modules, and real receipts.
- Private loop: the `inspect` operation group, the `capability_inspection`
  and `episode_inspection` operations, the `capability_doubt` trigger, the
  `{inspection}` thought-stage prompt slot, and the capability-belief commit.
- Companion assembly: the cached overlay on the application, the private
  loop rebuild, and the dialogue capability provider.
- New architecture rule: `pm.introspection` may import `introspection`,
  `records`, `memory`, and `paths`.

## Foundation and contract tests

The full foundation suites pass:

```
uv run --locked python -m unittest discover -s tests/unit -t . -b
Ran 693 tests. OK

uv run --locked python -m unittest discover -s tests/contract -t . -b
Ran 75 tests. OK
```

| File | What it shows |
| --- | --- |
| `tests/unit/pm/introspection/test_grants.py` | digest, revisions, observation time, credential denial, traversal denial, include bounds, oversized refusal, change detection, cancellation |
| `tests/unit/pm/introspection/test_capabilities.py` | the six states stay separate; enabled is not observed; a receipt raises the finding; missing and unknown evidence; staleness on a source revision change |
| `tests/unit/pm/introspection/test_episodes.py` | episodes from committed records; explicit cross-loop links; action outcome; distinct loop identities; question answers and limits |
| `tests/unit/pm/introspection/test_self_model.py` | first belief, no confidence stacking, supersede with history, staleness without erasure, narrative revision retaining the original episode |
| `tests/unit/workflows/introspection/test_snapshot.py` | module-to-capability mapping, real-repository overlay, public projection |
| `tests/unit/workflows/inner_thought/test_inspection.py` | the `capability_doubt` trigger, prompt rendering, capability belief commit, and no commit for another operation |
| `tests/contract/inner_thought/stages/test_inner_stages_contract.py` | inspection evidence reaches the thought prompt through the real kernel |

## Smoke run with real inference

A bounded smoke ran the real application path with the complete Spamuel seed,
a fresh checkpoint, and the real OpenRouter chat model (`qwen/qwen3.8-flash`).
It submitted one capability-discovery turn and waited for a reply.

```powershell
uv run --locked python .runtime/a18-smoke.py
```

The smoke is not the full gate.
It does not assert restart, concurrency, or the inspector questions.
It checks that the mechanism works on the real path.

Observed result:

- The private loop chose the `capability_inspection` operation by itself,
  driven by the `capability_doubt` trigger.
- It committed five capability beliefs with distinct states: `public_dialogue`,
  `private_thought`, `memory_tools`, and `source_inspection` were `enabled`;
  `minecraft_body` was `implemented`.
- The committed thought carried `reality=inferred` and a `capability_states`
  map beside its evidence.
- The public reply explained a capability limit in character: "What I cannot
  do is magically fix that wrong-file ..."

The local checkpoint is `.runtime/a18-smoke/ghost.json` (ignored).
The run shut down cleanly with no module error.

## Defects found during A18

1. The first grant glob used `pathlib.PurePosixPath.match`, which treats
   `**` as one segment. `pm/**/*.py` did not match `pm/module.py`. The grant
   now compiles its own glob with `**` crossing directory boundaries.
   A regression test pins both the shallow and the deep case.
2. The capability revision first included the observation timestamp, so an
   unchanged finding produced a new revision every cycle and defeated the
   retain rule. The revision now hashes only the stable evidence fields.
3. The episode id space overlapped between a record id and a tick id.
   `EpisodeIndex.by_id` now prefers record anchors over workspace turns.
