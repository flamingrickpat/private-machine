# A04 classification verification

## Summary

A04 adds bounded normal-LLM classification and a dedicated INVALID coding-agent handler.
The live engine demonstration uses actual logs, original Spamuel history, and real model inference.
It verifies source digests, routing, isolated inspection, and recovery after actual context overflow.

Parent: [classification](classification.md).
Related: [recovery and context](classification-recovery.md), [A03 evidence](verification.md), and [A04](../../decisions/alpha-plan/work-items/a04.md).

## Packages and activation

A04 pinned pm-workflows 0.7.0 at `823b06abf69506ef8e9dc4989065b9f60894fff6`.
The [A05 verification](pm-verification.md) records the current PM integration and package pin.
Its pm-coder pin remains `ac8327baf669dad54ea9f91580f2bed1023825b4`.
The coder implementation and compaction mechanism did not change.
The workflow adapter now enforces its configured bound-session deadline.

The demonstration supplies explicit `classification_models` and `agent_model` connections.
A05 owns application connection wiring.
No existing application or coding manifest needs a classification node.
Optional Laya is disabled for this use.

Chat was initially stopped.
The session restored `C:/source/ai/start38_uncensored_nvfp4.ps1` with its existing configuration.
Chat serves `qwen` on port 8080 with a 131,072-token context.
Embeddings on 8081 and optional Laya on 8082 remained unchanged.
The gate records actual `/models`, `/props`, installed origins, and seed hashes.

## Reproduce the engine demonstration

[`scripts.alpha_classification_gate.main`][] creates fresh host and Ghost environments, manifests, journals, and receipts.
It loads all 11,202 original Spamuel records.
It reads the original seed without changing it.
Run from the root repository with the locked installed package:

```powershell
uv sync --locked
uv run --locked python scripts/alpha_classification_gate.py --config config.alpha.example.yaml --out .runtime/alpha/a04-live-new
```

Use a new output directory for each run.
`--skip-overflow` is a development option; its receipt deliberately leaves acceptance incomplete.
The full delivery evidence belongs in `.runtime/alpha/a04/live-validated-delivery/`.

The gate copies three preserved logs from A01 and A03.
It records each source hash, encoding, timestamp, and UTF-8 copy hash in `sources.json`.
Only the test copies receive encoding conversion.
The originals keep their exact bytes.
The filename includes the actual source UTC timestamp.
Host modification times remain available; Ghost modification times remain unknown.

## Independently checked cases

| Case | Observed result and required evidence |
| --- | --- |
| Host mixed folder | Select the most recent foundation log, then `all_good`; exact A03 log digest |
| Ghost mixed folder | Same source and class; virtual timestamp remains null |
| Missing file | Primary `missing_file`; actual coding handler explains INVALID |
| Oversized direct prompt | Primary `oversized_input`; actual coding handler returns `all_good` with source quotes |
| Truncated response | Actual one-token completion; `incomplete_response` with retained provider response |
| Unavailable endpoint | Actual connection refusal on a closed local port; `endpoint_unavailable` |
| Absent producer metadata | `uncertain_selection`; actual coding handler also returns INVALID |
| Synthetic mixed run | Actual inference returns `some_failed` for two passes and two failures |
| Synthetic failed run | Actual inference returns `all_failed` for three failures |
| Synthetic contradictory summaries | Actual inference returns INVALID for contradictory, undated summaries |

The synthetic sources are visibly labeled in the receipt.
They test independently defined facts through actual inference.
They do not substitute for the captured application logs or original history.

Each case retains its manifest, static/dry-run report, primary debugger result, and recovery result when applicable.
Every accepted answer must quote actual evidence and copy its canonical captured-value digest.
Receipt replay verifies the same retained values and answer without inference.
It does not establish semantic interpretation independently.

## Forced overflow evidence

The overflow log contains an exact prefix of original Spamuel history plus the original A03 test footer.
The actual server tokenizer measures 145,258 tokens in this tool result.
The serving context holds 131,072 tokens.
The recovery skill requires one full virtual `cat logs/overflow.log`, then focused inspection after recovery.

The direct classifier rejects its configured 512-token prompt budget before dispatch.
The handler runs actual pm-coder inference with automatic recovery enabled.
The server rejects the accumulated request because it exceeds context.
Its retained response measures 148,144 request tokens against the 131,072-token context.
Real summary inference creates a checkpoint, and the agent resumes.
The final answer quotes the current footer's `Ran 261 tests` and `OK` evidence and preserves the captured-value digest.
The session retains five agent requests and four summary requests.

The overflow acceptance check requires all of these artifacts:

- `overflow-input.json`: actual tokenizer count, byte count, serving context, and source seed hash.
- `overflow-http/http-*.json`: actual upstream status and response, including a context rejection.
- `kernel/*/pm-coder/*/turn_*_compact_*.json`: real summary requests.
- `kernel/*/pm-coder/*/messages.json`: checkpoint text containing the decisive count and OK evidence.
- `overflow-receipt.json`: unchanged original instruction and classes in resumed requests, validated answer, and checkpoint details.
- `receipt.json`: overall pass, all ten ordinary cases, and unchanged seed.

A temporary relay on an ephemeral loopback port captures actual provider responses.
It forwards request bodies unchanged to the existing chat server and never substitutes a response.
The relay closes after the overflow session.
No server context, model, console configuration, or Minecraft state changes.

## Regression and boundary checks

The final workflow package suite passed 199 tests in 64.54 seconds.
It includes actual existing coding roles on host and virtual targets, inference cancellation, and the bound-session deadline.
The existing role manifests retain their original fields and omit classification connections.
Static checks also inspect the five unchanged central coding manifests.

Classification boundary checks cover manifest errors, denied and stale evidence, changed contracts, item scope, origin reuse, isolation, and replay corruption.
They also reject empty candidate identifiers and the internal exact-capture key.
The incomplete-response fixture preserves an actual truncated Qwen response from the development gate.
It tests protocol parsing; live inference remains the behavioral acceptance source.

The root foundation suite, installed adapter checks, strict MkDocs build, compilation, and browser inspection are required delivery checks.
Their exact commands, timings, package origins, and final commits are recorded in `.runtime/alpha/a04/session-receipt.json`.
Browser inspection opens local HTML, follows classification navigation, and verifies exact generated API anchors.

## Retained failures and limits

The delivery retains failed development runs and their repairs:

| Failure | Repair and retained evidence |
| --- | --- |
| Chat connection refused | Existing startup script restored chat; `chat-start.json` and startup logs |
| Empty bearer header rejected locally | Omit authorization for an empty local key; `live-restored/` |
| Selection invented a minimum log size | Require metadata-only selection and forbid unstated size rules; `live-header-fixed/` |
| Missing Ghost file appeared inaccessible | Check actual virtual file existence before reading; `live-selection-fixed/` |
| Agent used raw-file digest or exact-file candidate ID | Explicit canonical-value identifiers and cleaned descriptors; `live-missing-fixed/` |
| Truncation cause became generic malformed JSON | Preserve `InputError` and validate finish reason first; `live-full-development/` |
| Correct label appeared as an array | Typed JSON Schema and explicit single-string status; `live-delivery/` |

Accepted quotes prove correspondence with captured text; they cannot guarantee every model interpretation.
Unknown timestamps and absent producer metadata remain unknown.
Capture is version-checked but is not a filesystem transaction.
The conservative UTF-8 budget supports the tested byte-based tokenizer; other tokenizer families require a suitable bound.
Oversized initial agent instructions remain INVALID because compaction needs generated execution to summarize.

A04 does not establish the original requested note, repair public replies, or verify memory claims.
C01 and C02 remain open for A08/A09.
A05 through A20 were pending at A04 delivery.
The [A05 verification](pm-verification.md) records the later PM wiring.
