# A18B review materiality verification

## Summary

The change extends sufficient-result classification from Minecraft review to public speech and A18B criterion review.
It preserves minor judgments, their reasons, and citation requirements.
The model decides practical materiality from prompts and supplied evidence.
Real inference calibration and complete A18B acceptance remain pending.

Parent: [A18B verification](a18b-cognitive-evaluation.md).
Mechanism: [review materiality](../../subsystems/review-materiality.md).
Original failures: [runtime speech review](a18b-errors/reviewers.md).

## Changed behavior

Minecraft already accepted `good_enough` and `equivalent_or_better` in its default flexible mode.
Public speech now records both labels explicitly.
A passing speech label cannot contain a material `false_task_result` flag.
A18B adds `good_enough` to its provider schema, host validation, aggregate scoring, and calibration comparison.
Every decided criterion still needs exact citations.
A model's reason explains why a discrepancy is incidental to the actual request.

Prompt guidance covers both dialogue generators, speech review, action selection, project planning, plan review, independent verification, and both Minecraft controller paths.
Both A18B audiences receive materiality and chronology examples.
There is no new distance table, item allowance, score threshold, or numerical error percentage.
Existing independent task checks remain in effect.

## Local checks

Evidence belongs under `.runtime/a18b-current/reviewer-materiality/`.
The final focused suite passed 50 tests in 1.644 seconds.
The contract suite passed 80 tests in 29.469 seconds.
The full unit result and HTML inspection are recorded beside those logs after completion.

```powershell
uv run --locked python -m unittest tests.unit.workflows.public_speech.test_public_context tests.unit.workflows.public_speech.test_speech_review tests.unit.workflows.minecraft_project.test_review_policy tests.unit.dev.test_cognitive_evaluation tests.unit.dev.test_cognitive_review_scope -b
uv run --locked python -m unittest discover -s tests/unit -t . -b
uv run --locked python -m unittest discover -s tests/contract -t . -b
uv run --locked --with-requirements requirements-docs.txt mkdocs build --strict
```

The boundary checks establish class routing, scoring, and citation behavior.
They do not establish the model's semantic accuracy.

The first full unit run completed 836 tests in 921.999 seconds with one failure.
Its context-reduction assertion compared current instructions against an older captured prompt length.
The test now compares both payloads with identical current instructions.
The captured payload reduces from 107,642 to 102,120 characters under the existing scope selection.
The runtime still measures the complete prompt against its serving token budget.
The failed log remains in `unit.log`.
Subsequent results use separate files.
The corrected single-test replay passed in 3.301 seconds.
Final foundation results belong in `unit-final.log` and `contract-final.log`, with their process exit codes beside them.

## Pending real calibration

`test_data/captures/alpha/a18b-materiality-evidence.json` retains an actual harvest, a later world observation, and an earlier inventory observation.
Its source paths and SHA-256 digests identify the preserved originals.
The new live gate uses authored claims against these captured facts and the complete Spamuel seed.
It calls the configured real model without starting Minecraft or changing the world.
Controls cover harmless approximation, material coordinate error, invented diamonds, incidental inventory, count imprecision, exact placement, historical collection, and conditional mechanics.

```powershell
$env:PM_MODEL_CONFIG = 'config.alpha.openrouter.yaml'
$env:PM_REVIEW_MATERIALITY_LIVE = '1'
uv run --locked python -m unittest tests.live.test_review_materiality -v
```

Each execution writes a fresh `live-<timestamp>-<id>/` directory under the evidence root.
It retains every model request and response, observed judgment, expected blocking result, and human review status.
A ten-minute cancellation deadline fails incomplete execution.
It cannot grant approval.
These controls test review of captured inputs, rather than new gameplay or complete companion behavior.
The gate remains unrun while the owner keeps live verification stopped.

Historical evidence retrieval and draft repair are still pending.
The unchanged rejected-draft runtime error remains visible until that separate repair receives evidence.
