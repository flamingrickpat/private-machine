# A02 verification and remaining limits

## Summary

A02 verifies the public client against real typed inference and real service failures.
The primary inputs come from Spamuel's unchanged seed and a captured A01 task report.
Laya remains optional because protocol validity does not establish interpretation quality.
The A01 companion failures remain open.

Parent: [optional typed decisions](../../subsystems/decisions/index.md).
Repeat: [operation](../../subsystems/decisions/operation.md).
Historical baseline: [A01 receipts](../alpha-plan/baseline/receipts.md).
Model observations: [quality and limits](../../subsystems/decisions/quality.md).

## Source and installation

| Repository | Tested revision | Change |
| --- | --- | --- |
| pm-coder | `ac8327baf669dad54ea9f91580f2bed1023825b4` | Version 0.10.0, public `pm_decision`, captured boundary tests, client manual |
| pm-workflows | `c46a5e3f947755c21e388618fb00a1ccef2628f0` | Version 0.5.1, exact tested pm-coder dependency |
| PM | Parent `93a403eb4f02cefbf15893bf8739705b038bf587` | Exact dependency constraints and locks, live runner, linked manual, generated dependency API |

The final live runner records installed package origins in `live-final/identity.json`.
It uses PM's installed distribution, rather than importing the sibling checkout.
A fresh environment also installed the built wheel and made eight real synchronous and asynchronous calls.
Its isolated import loaded no coding agent, Ghost, Pydantic AI, or workflow engine.
The distribution still installs the agent's dependency set.

All evidence paths on this page are relative to `.runtime/alpha/a02/`.
The source snapshot, original lock hashes, and dirty trees are in `snapshot-before/snapshot.json`.
The root and supporting package origins matched their original locks before implementation.

## Real inputs and service identity

The primary seed remains `test_data/personas/spamuel/canon.yaml`.
The runner reads its 11,202 records without modifying that source.
Its SHA-256 remains `0071b51dda2155a1f635375c1b604be0c2e6159983889ed5b070be4de11af0bd`.
The A01 note report hash is `66d5064a6fce3d5535939ef7853ad3dc176ec016470055692d7fcd5c08a29f77`.
Its checkpoint identifies managed run `dd8a80caec16462b97e0e85cbdf87346` as completed.

The runner stores report and checkpoint paths, hashes, selected history layouts, and the run ID in `live-final/provenance.json`.
Added layout labels and questions are generated demonstration inputs.
They remain distinct from the captured application report and inherited history.
The A01 routing development and holdout sets remain untouched.
Human routing review and immersion ratings remain pending.

| Artifact | Verified identity |
| --- | --- |
| Binary build | `b11374-b92761a51` |
| Binary SHA-256 | `07954e81e04924cd1fc88b560b2d1d88d825785e7fe0dafadeaecea216be522f` |
| Laya BF16 SHA-256 | `edeea94939cd1575c1f984c28982170c0e47caab66e4b80993fed992519573b4` |
| Shared alias and endpoint | `laya`, `http://127.0.0.1:8082/v1/systemone` |
| Owned failure endpoint | `http://127.0.0.1:8084/v1/systemone`, CPU, same binary and artifact |

## Failure and cancellation evidence

| Gate | Observed result | Evidence |
| --- | --- | --- |
| Malformed local non-finite input | `DecisionInputError` before dispatch | `live-final/malformed.json` |
| Empty provider questions | HTTP 400 | `live-final/server-malformed.json` |
| Decision request sent to chat port | HTTP 404, `DecisionHTTPError` | `live-final/wrong-endpoint.json` |
| Wrong expected model | Server returned `laya`. Client raised `DecisionSchemaError` | `live-final/wrong-model.json` |
| HTTP timeout | `DecisionTransportError` | `live-final/timeout.json` |
| Async cancellation during active inference | `DecisionCancelled`, request digest retained | `live-final/owned-failures.json` |
| Synchronous event cancellation during active inference | `DecisionCancelled`, local worker joined | `live-final/owned-failures.json` |
| Process termination during active inference | `DecisionTransportError` with `ReadError` | `live-final/owned-failures.json` and `owned-service.log` |
| Request after cancellation | Valid real typed result on the same owned service | `live-final/owned-failures.json` |
| Missing answers and invalid numeric/schema values | Captured-response boundary checks passed | `coder-boundary-final.log`, 42 tests |

The failure process stopped and released port 8084.
Cancellation receipts leave server completion unknown.
The client sends no retry and commits no application state.
No fake world or fixed inference response supplied a live gate.

## Foundation and documentation

The pm-coder broader run passed 114 tests in 508.47 seconds, including its real LLM file and image scenarios.
It excluded the separate session-isolation and virtual-permission live modules.
The final boundary run passed 42 tests in 0.68 seconds.
The additional provider, tool, and workspace checks passed 53 tests before the last two input-boundary additions.
The pm-workflows suite passed 133 tests in 73.09 seconds.
The clean-environment receipt is `clean-python.json`.

The root foundation passed all 257 tests in 399.608 seconds.
Its receipt and complete output are `foundation-command.json` and `foundation.log`.
The native-exit anomaly did not recur in this run.
The real-server gate passed one test in 8.266 seconds after the authorized complete world reset.
Its receipt is `minecraft-real/receipt.json`, and its output is `minecraft.log`.
It used `C:/source/minecraft-server-real`, seed `23452352345`, and a separate test character.
Its owned Minecraft and MCP processes stopped after the gate.
Script compilation also passed in `compile-command.json`.
The documentation builds generate `pm_decision` objects from source and the installed distribution.
HTML remains ignored in both repositories.
The final site inspection verifies local HTML navigation and generated API links.
Its receipt is `html-inspection.json`, with screenshots in the same runtime directory.

## Preserved failures and exclusions

The first live runner stopped before inference because its checkpoint path selected the wrong parent directory.
Its retained evidence directory is `live/`.
The corrected runner passed in `live-verified/` and again through installed packages in `live-final/`.
This harness error does not establish a model failure.

The first documentation shell reported a redirection error after MkDocs completed.
The captured subprocess repeat returned exit zero in `docs-before-repeat-command.json`.
The log retains both the shell anomaly and the completed build.

A01's completed control-worker run still lacks its expected public reply.
Its artifacts do not establish an exact-content note or arbitrary instruction execution.
The [baseline clarification](../alpha-plan/baseline/receipts.md#worker-completion-and-actual-task-acceptance) separates run completion from actual task acceptance.
The cheerful restart scenario still contains a false memory-loss claim despite equal restored records.
The first A01 native foundation exit remains unexplained.
A02 changes no cognitive stage or public-reply policy to conceal these gaps.
The Minecraft gate remains the narrow one-action and recovery check.
It establishes no house, self-crafted iron pickaxe, sustained survival, or visual interpretation.
