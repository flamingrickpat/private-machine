# A01 baseline receipts

## Summary

These receipts identify the captured implementation and observed gates on 2026-10-03.
The real-server Minecraft gate passed.
A managed worker reported completion but produced no public reply.
The current LLM comparison matched the proposed development labels.
Historical acceptance and current runtime evidence remain separate.

Parent: [A01 baseline](index.md).
Repeat: [commands](commands.md).
Review: [scenario matrix](scenarios.md).

## Source snapshot

The initial `pm_next_v2` parent is `54149fca1f5de757f93dcb35160735ed99d158fd`.
The initial tree has 296 nonignored source files.
Its source archive SHA-256 is `504b8205108ec4cdf7fa006f96da7096b07801d2ea327628aaaf38f6d704315a`.
The archive, file manifest, and binary patch remain under `.runtime/alpha/a01/`.

The implementation baseline commit is `a6d459349e84734603ab8d6af4fc702531b03a7d`.
The [machine receipt](receipt.json) contains the selected identities, hashes, commands, and observations.
The second commit adds the A01 procedures, fixtures, and results.
The original brief survives byte for byte in [original-brief.md](original-brief.md).

| Repository | Full captured revision | Working-tree status |
| --- | --- | --- |
| `private-machine` | `7f9bd048dd211ffe0c027cf9865e0250101628c7` | Clean. Older runtime not exercised |
| `private-machine-head` | `5fa78b04346519d57d8ee8f0f5393f9cdb893273` | Clean. Older runtime not exercised |
| `pm_next` | `d5ddda71881db812ceb2de4ee59a3f871aa1c1cc` | Existing staged, unstaged, and untracked files preserved locally |
| `pm_next_v2` | `54149fca1f5de757f93dcb35160735ed99d158fd` | Existing console, memory, Minecraft, and alpha-plan edits preserved in the baseline commit |
| `pm-workflows` | `d76c5aa023189cfc4b42fd1e9d79b5580f6c6731` | Clean. Matches the dependency lock |
| `pm-coder` | `0efcafbc4891b287d8f311dac671326b56a31585` | Clean. Matches the dependency lock |
| `coding-workflow` | `c32e6bd7b7b75265bc31527aed9dc34959cfeea3` | Clean. One existing commit ahead of its remote tracking branch |
| `pm-minecraft` | `2afee3342a4095a6cabf2b2c0bbd347569d54c7c` | Clean. Used by the real-server gate |
| `pm-agentic-babymode` | `909ca91cdf3e5559c0d930543482140111b21afb` | Clean. Mod version 0.5.0 |
| `minecraft-server-real` | `51c4b1e8716b6ab8221434384f6e0535202e0326` | Existing configuration and mod changes captured before reset |

The supporting source archives remain in `.runtime/alpha/a01/snapshot-before/`.
Their receipts record dirty files and exact hashes.
The installed PM package origins and locked revisions agree.
A01 does not upgrade a dependency or execute an older PM runtime.

## Current verification

| Gate | Observed result | Evidence |
| --- | --- | --- |
| Strict documentation before A01 | Passed in 28.81 seconds | `docs-before.log` |
| Root foundation, first uninterrupted attempt | Native process exit `0xC0000005`. No test assertion or final count | `foundation.log`. Diagnostic repeat recorded separately |
| Root foundation, diagnostic repeat | Passed: 257 tests in 428.110 seconds. Count change from planning: zero | `foundation-verbose.log` |
| Real console task request | Failed: no public reply within 600 seconds. The managed worker reported completion | `console-spamuel/receipt.json`, worker checkpoint, and trace |
| Short console conversation and restart | Passed: three exchanges per persona. Memory, GWT, and run receipts matched across restart | `console-spamuel-dialogue/` and `console-cheerful/` |
| Behavioral recall after restart | Sunny claimed that earlier memory was lost despite identical restored records | Final cheerful-persona reply and both snapshots |
| Strict documentation after A01 | Passed. Final build duration appears in the machine receipt | `docs-final.log` |
| Existing narrow Minecraft gate | Passed: one test in 9.837 seconds | `minecraft-real/receipt.json` and W8 artifacts |
| Current LLM development comparison | 12 responses agreed with 12 proposed labels. Human label review remains pending | `routing-comparison/receipt.json` |
| Holdout comparison | Not run. Six cases reserved | `tests/fixtures/alpha/routing-holdout.json` |
| Laya inference quality | Not run in A01. Model discovery succeeded | Snapshot service identity |
| Human immersion and regeneration ratings | Not recorded by automation | Operator record remains pending |

All paths in this table are relative to `.runtime/alpha/a01/` unless stated otherwise.
The first foundation launch was interrupted by user steering before it produced a suite result.
The subsequent native exit remains an explicit anomaly rather than a passing receipt.
The diagnostic repeat uses the locked environment's Python directly with `-X faulthandler` and verbose unittest output.
The same 257 tests passed without an assertion failure or another native exit.
The native-exit root cause remains unknown.

Spamuel restored 11,210 accepted records before the final exchange.
Sunny restored eight records.
Both final replies mentioned `copper lantern`.
Spamuel repeated wording across its first two replies and answered the final recall prompt indirectly.
Sunny claimed that an earlier codeword did not survive the restart, although both snapshots contained identical accepted records.
The scenario supplied only `copper lantern` before restart.
That loss claim contradicts the captured state and conversation facts.
It establishes a recall or interpretation gap for A08–A09.
The two runs overlapped on the chat service and used different seed histories.
Their timing cannot establish a controlled persona effect or a stable latency target.
The restart result establishes durable state equality, rather than emotional continuity or reply quality.

## Diagnosed conversation failure

The first request asked the companion to write `copper lantern` through `local_task`.
Action selection produced `ToolCallAndReply` for `local_task`.
The supervisor stored a completed run and consolidated its artifacts.
Two later cognitive cycles treated run telemetry as background activity and selected no action.
The console therefore retained a pending exchange without a `chat.reply` event.

This is a behavioral gap in the current action/reply path.
It is not an unavailable endpoint or a failed managed-worker execution.
A09 must address useful public response and recovery.
Dependent conversation acceptance must retain this case and its failure evidence.
A01 does not patch the cognitive path to make its baseline pass.

### Worker completion and actual task acceptance

The completed receipt establishes that the `local_task` control worker finished.
Its Python role writes the instruction, observations, a result summary, and `outbox/report.yaml`.
It never invokes a coding model or interprets arbitrary instructions to create a requested note.
The captured report repeats the instruction and contains an empty `answer` field.
These artifacts do not establish a note containing exactly `copper lantern`.

The earlier description of a completed note was too strong.
Retain C01 as a managed-run/public-reply regression.
A repaired task-success scenario must also use a real agent-backed note task and inspect its actual output bytes.
Run completion, task acceptance, and public reply delivery require separate evidence.

## Minecraft evidence

The gate used the actual real server after its wipe script.
The server logged Babymode 0.5.0 startup and a fresh world with zero tracked players.
The character connected as `tdd_player` with its files under the run's isolated directory.
The existing gate verified a mined block became air, screenshot evidence, instruction revision 1, cancellation, and checkpoint recovery.

Server seed: `23452352345`.
Server-properties SHA-256: `e554f9c2ca712c47a49b6df7b2e0e378807fb59e33051de6442a51dd9b77252f`.
Wipe-script SHA-256: `dba67e9dfad9840a23a019986229a93d64be06583e312ad845adafaaffe2f052`.
Babymode-jar SHA-256: `d53cf9b424bc69ae6fd0bfad39e31c7d61089e91f65410952e2d4e864507d453`.

This gate establishes one-action and recovery behavior.
It does not establish a completed house, a self-crafted iron pickaxe, sustained survival, or visual interpretation.
The runner stopped its owned MCP and game-server processes.
It retained the generated real world for inspection.

## Historical gate evidence

| Historical receipt | Result | Interpretation |
| --- | --- | --- |
| Planning foundation, 2026-10-03 | 257 tests in 363.527 seconds | Historical engineering baseline |
| `w7/conversations/20260930T053223Z-720a2130/summary.json` | Reported 200 exchanges and nine good trajectories at `88f2a8f184c6455c7e939e8fc88a643d422d0e66` | Older conversation harness. Does not establish current application acceptance |
| `w7/conversations/20260930T214238Z-ff099a25/summary.json` | Failed: 180 completed application cycles out of 200. Six evaluated good trajectories at `3ea45a1a42c47dd2194e13f2c3769c6f94bb8c3f` | The later full application gate failed |
| `w7/conversations/20261002T175830Z-9956f0eb/summary.json` | Four completed exchanges at `d6180c8f3cb0719fa2e0317701e68ad071bf6f0e` | Short application smoke, without full quality acceptance |
| Existing `w8/live-minecraft/*/minecraft.json` receipts | Narrow action and evidence records retained | Historical body behavior, not current sustained gameplay |

The source snapshot records hashes for every available W7 summary and W8 Minecraft receipt.
The A01 gate adds current real-server evidence without replacing historical receipts.
