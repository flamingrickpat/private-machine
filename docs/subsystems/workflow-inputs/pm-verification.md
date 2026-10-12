# A05 PM classification verification

## Summary

A05 checks classification through PM's actual runner, lazy service provider, and isolated recovery path.
The gate uses preserved test reports and the complete original Spamuel seed.
It checks changed evidence, concrete failures, every class, and real context recovery.
The current Story module retains its original chat path with Laya stopped.

Parent: [PM connections](pm-connections.md).
Operation: [demonstration](../../workflows/classification-demo.md).
Delivery: [A05](../../decisions/alpha-plan/work-items/a05.md).

## Reproduce M1

Use the existing chat on port 8080, alias `qwen`.
Stop optional Laya before the acceptance run.
This gate starts no optional service and changes no deployment configuration.
Use a fresh ignored output directory:

```powershell
uv sync --locked
uv run --locked python dev/work_items/alpha_pm_classification_gate.py --config config.alpha.example.yaml --out .runtime/alpha/a05-live-new
```

[`dev.work_items.alpha_pm_classification_gate.main`][] loads all 11,202 seed records through `build_app`.
It uses a separate checkpoint path and checks that construction performs no activation.
It checks provider injection into shell modules and the managed supervisor.
It copies the actual A01 and A03 logs and records original hashes, encoding, and copy hashes.
Only these copies receive UTF-8 conversion.
The original history and historical logs remain unchanged.

`--skip-overflow` and `--allow-laya-present` are development flags.
Either flag leaves `passed` false even when its executed cases succeed.
The final receipt remains under `.runtime/alpha/a05/live-final-delivery/`.

## Cases and independent facts

| Case | Required observation |
| --- | --- |
| Host and Ghost folders | Latest real foundation log, all_good, exact selected digest, known host time and unknown Ghost time |
| Changed Ghost report | Complete A04 foundation bytes at the same path, new captured-value digest, all_good |
| Missing file | Distinct missing_file primary cause; actual handler explains INVALID |
| Oversized direct prompt | Distinct oversized_input; actual handler quotes the preserved current test footer |
| Missing producer metadata | uncertain_selection; actual handler remains INVALID |
| Stale named producer | Successful first attempt followed by a failed projection; stale_output blocks old-value reuse |
| Unavailable chat | Actual closed local port; endpoint_unavailable in retained service and classification receipts |
| Independent synthetic summaries | Actual inference for some_failed, all_failed, and contradictory INVALID |
| Long report | Actual overflow rejection, summary inference, checkpoint, resumed exact question/classes and decisive evidence |
| Current Story module | Actual accepted nonempty reply through existing Python roles and lazy chat tool during a shared active lease; no module classification events |

Synthetic sources are visibly labeled in receipts.
Their expected classes follow explicit source counts, rather than either model's answer.
They test boundary routing and supplement the captured application reports.
They never replace real inference or establish human interpretation quality.
Every classification receipt also passes the existing replay validator.
Replay checks source correspondence, rather than repeating semantic interpretation.

The gate checks that classification leaves the original Ghost records unchanged.
The separate Story compatibility run can commit its accepted dialogue to the isolated checkpoint.
It does not establish repaired C01 task/public-reply behavior or C02 recall.
Human immersion ratings remain unrecorded.

## Inspect a run

1. Open `receipt.json`, `identity.json`, and `sources.json`.
2. Open each `*-result.json` and its actual `outbox/report.yaml`.
3. Open `runtime/kernel/<case>/journal.jsonl` for full captures, requests, answers, and routes.
4. Open the adjacent `model-sessions.jsonl` for readiness, serving context, and lease release.
5. Open `overflow-input.json`, `overflow-http/`, and `overflow-receipt.json`.
6. Open the pm-coder `messages.json` and compact request files for actual context recovery.
7. Open `compatibility-story.json` and its module journal for the original execution path.

Service receipts omit credentials.
The full source and model evidence stays under ignored runtime paths.
Direct classifications use conservative bounded context; no source clipping occurs.
The long case uses pm-coder's unchanged accumulated-history compaction.

## Optional candidate disposition

PM Laya wiring remains deferred.
The public `pm_decision` client remains installed through the unchanged pm-coder pin.
A02 recorded incorrect report and public-reply interpretations, including confidently wrong answers.
Those [observations](../decisions/quality.md) reject these interpretive uses for adoption.
This item supplies no new qualifying Laya use and sends no decision request.
Exact metadata checks remain ordinary Python code.

A future simple use needs its own realistic endpoint tests and untouched cases before activation.
The [model policy](../../decisions/alpha-plan/model-policy.md) supplies that procedure.
No adoption target or pending historical label review blocks the normal LLM delivery.

## Measured live results

The final M1 gate passed all eleven ordinary cases and the forced-overflow case.
The long tool result measured 145,258 tokens against the 131,072-token serving context.
The provider rejected the first accumulated request at 148,130 tokens.
Actual summary inference and resumed inspection retained the exact `Ran 261 tests in 493.985s` and `OK` evidence.
The source digest and original question/classes survived.
The accepted Story reply also passed while the same alias held an active classification lease.

The captured test footer belongs to the preserved A03 report.
It does not describe the current foundation validation count.
The workflow package passed 199 tests in 66.64 seconds.
PM pins pm-workflows 0.7.1 at `7c9fab4381ef77ef3f63ee9ef669a993c8df7452`.
The pm-coder pin remains `ac8327baf669dad54ea9f91580f2bed1023825b4`.
The application wiring subset passed 23 tests in 3.812 seconds.
The final root foundation suite passed 268 tests in 411.420 seconds.
Five unchanged central coding manifests passed static checks.
The session receipt records final foundation timing, exact package pins, HTML checks, and protected-source hashes.

The [diagnostic guide](pm-failures.md) records development failures, service cleanup, and the historical HTML evidence correction.
A06 through A20 remain pending.
C01 and C02 remain open for A08/A09.
