# A18 verification

## Summary

A18 has an engineering delivery and one pending live gate.
The engineering delivery is the inspection substrate, the capability overlay,
the episode view, the self-model beliefs, and the private-loop integration.
Its foundation and contract checks pass.
The live gate needs real model inference and the complete Spamuel history.
It has not run yet, so this page records no behavioral acceptance claim.

Parent: [the self-model subsystem](../../subsystems/introspection.md).
Work item: [A18](../alpha-plan/work-items/a18.md).
Design: [self-model and reflection](../alpha-plan/reflection.md).

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

## Pending live gate

The live gate must run the real application path with real inference and the
complete Spamuel history.
Use [the local commands](../alpha-plan/baseline/commands.md) to select
`config.alpha.openrouter.yaml`.

Run it as a new gate script under `dev/work_items/`:

```powershell
uv run --locked python dev/work_items/alpha_a18_self_model_gate.py `
    --config config.alpha.openrouter.yaml --out ".runtime/alpha/<stamp>-a18"
```

The gate must assert the mechanical contracts and record the scenario facts
for human review.

| Case | Required evidence |
| --- | --- |
| Active, disabled, and planned feature | three distinct statuses with actual files and wiring |
| Stale capability | a configuration or source change invalidates an old belief and produces a sourced revision |
| Decision explanation | the inspector answers the episode questions from retained records |
| Parallel thought and action | explicit links establish influence without inferring causality from timestamp order |
| Missing trace | the companion retains uncertainty instead of inventing a mechanism |
| Capability misunderstanding | a schema or tool presence does not become an unsupported observed capability |
| Narrative revision | counterevidence changes an interpretation while retaining original episodes |
| Emotional discovery | inspected evidence reaches appraisal and affects a later conversational choice |
| Restart | beliefs, episode references, validity intervals, and uncertainty survive |

## Open checks

- The live gate above.
- Human review of naturalness and self-explanation, recorded separately from
  the mechanical contract.
- Persona comparison with matched source evidence: the cheerful and dramatic
  personas must share the same inspected facts.
- The installed distribution origin versus the repository revision.
  The default inspector records the repository revision only.
- Minecraft body capability: a private goal that uses the body still needs
  its own ownership and recovery verification.

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
