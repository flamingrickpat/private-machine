# A12 interruption verification

## Summary

The real-server protocol gate passed six interruption and control checks.
Each replacement session observed the surviving body and rejected the old epoch.
Real companion publication recovery passed nine checks.
Native wood and house recovery passed independent verification.
The final foundation and focused checks passed.
This page records only measured outcomes.

Parent: [body recovery](../../subsystems/body-recovery.md).
Contracts: [body leases](../../subsystems/body-leases.md) and [publication recovery](../../subsystems/recovery-publication.md).

## Deployment and commands

The gate uses `C:/source/minecraft-server-real`, Fabric 1.19.4, Babymode 0.5.0, and seed `23452352345`.
It uses the real pm-minecraft checkout and the `tdd_player` avatar.
The local chat endpoint is `http://127.0.0.1:8080/v1`, alias `qwen`.
Native task checkpoints use the unchanged Spamuel seed.
Its SHA256 is `0071b51dda2155a1f635375c1b604be0c2e6159983889ed5b070be4de11af0bd`.
The gate provides no world reset, item grant, creative mode, or operator gameplay command.

```powershell
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
uv run --locked python dev/work_items/alpha_a12_recovery_gate.py --out ".runtime/alpha/$stamp-a12-protocol" --protocol
uv run --locked python dev/work_items/alpha_a12_recovery_gate.py --out ".runtime/alpha/$stamp-a12-native" --native
uv run --locked python dev/work_items/alpha_a12_recovery_gate.py --out ".runtime/alpha/$stamp-a12-house" --native --house-only
uv run --locked python dev/work_items/alpha_a12_publication_gate.py --out ".runtime/alpha/$stamp-a12-publication"
```

[`dev.work_items.alpha_a12_recovery_gate.run`][] owns the real stack and stops only its processes.
The failure-injection workers call the real MCP and retain their full logs.
The native workers use `build_app`, the production supervisor, and pm-coder inference.

## Captured protocol results

Receipt: `.runtime/alpha/a12-protocol-01/receipt.json`.
Final receipt: `.runtime/alpha/a12-protocol-final/receipt.json`.
The final run also tests active-call cancellation and an actual MCP-process restart.

| Boundary | Observed result |
| --- | --- |
| Before call | The killed worker sent no planned rotation |
| In-flight call | The sneak pulse retained a dispatch without a result; reconciliation kept it uncertain |
| After effect | The real rotation result existed outside the journal; reconciliation kept the unmatched dispatch uncertain |
| Before publication | The journal retained the real rotation result before process exit |

Each case rejected a concurrent process before any body call.
The operating-system lock became available after process termination.
A higher epoch then obtained a fresh observation.
The MCP rejected the original token with `Stale Minecraft body epoch`.
The gate did not repeat the uncertain mutation.
Its cancellation watcher stopped the real active navigation in 0.5 seconds.
After hard process restart, MCP rejected the earlier active token at epoch 48.
The replacement used epoch 49 and obtained a fresh body observation.
The character directory and actual world survived that restart.

`test_data/captures/alpha/a12-interruptions.json` contains the captured original scenarios.
These records came from real calls, rather than generated routing examples.
Foundation checks use them to preserve uncertainty and journal parsing behavior.
The protocol gate does not establish native planner quality or a finished gameplay task.

## Native recovery and retained failures

The native wood gate accepts an instruction revision while work remains active.
It cancels after actual collected wood, disconnects and reconnects the body, then explicitly resumes the same task.
The native house gate restarts the process during a structure-inspection leaf.
Its exact physical criteria come from the preserved A11 house specification.
Its independent verifier reads the surviving structure after restart.
The native task receipts are under `.runtime/alpha/a12-final-03/`.
The wood task is `9c3352e00a2444c3968da01551fcc215`.
Its explicit resume retained worker identity and changed its execution epoch from 1 to 2.
Its body token changed from epoch 32 to 33.
The final instruction revision was 2.
The operator clarified that repeated partial harvests in one trunk column count as one tree.
The accepted update preceded eleven later dispatch markers; all eleven used revision 2.

`wood/independent-verification.json` recomputes the current verifier against the actual retained journal.
It records two trunk columns, 11 original logs, 19 final logs, and a return distance of 0.753 blocks.
Maximum observed travel was 13.551 blocks.
This recovery scenario requires two trees and permits zero minimum travel.
It establishes lifecycle recovery, rather than the earlier five-tree expedition claim.

The house test uses the exact physical criteria from `.runtime/alpha/a11h16/reviewed-tree.json`.
Keep that preserved source available when rerunning this local gate.
The test inspects and preserves the existing house; it makes no new house-entry claim.
Final house receipt: `.runtime/alpha/a12-house-final/receipt.json`.
Task ID: `fb1d092f0a4d431d9ac89112c87907f1`.
The explicit replacement advanced its worker epoch from 1 to 2.
`actor-0001` remains interrupted; `actor-0002` completed after fresh observation.
Independent verification accepted 54 structure cells, two door cells, ten air cells, and 17 support cells.
The separate alive condition passed, giving 84 original reads.
The aggregate goal certificate also passed.

## Companion publication results

Receipt: `.runtime/alpha/a12-publication-02/receipt.json`.
The gate uses the production seven-stage cognition workflow and real qwen inference.
It interrupts bus delivery after the accepted action commits.
Restart republishes action ID 11208 with its original exchange correlation.
The real public generator commits reply ID 11209 before another injected delivery failure.
Restart delivers that same reply once and acknowledges it.
It creates no duplicate accepted action or replacement user turn.

An operator-selected real pm-coder note task supplies terminal progress.
The independently checked output contains 39 exact UTF-8 bytes.
Its SHA256 is `e215beffe5ac316d2b9dc8369e0fd6189da7b9cfef95c463240143718b7bba87`.
Publication failure preserves the completed state and original completed event ID.
Restart replays progress without creating a worker.
Acknowledgment stops further progress replay.
This operator note case tests the publication boundary, rather than companion task selection.

## Retained failures and repairs

`.runtime/alpha/a12-native-01/receipt.json` retains the first Windows receipt-read failure.
Its actual 600-second navigation call also exposed pm-coder's earlier five-minute MCP deadline.
The repaired package honors the actor's 660-second deadline.
No overlapping retry replaced that in-flight call.

`.runtime/alpha/a12-final-02/receipt.json` retains a rejected recovery attempt.
Its fresh tree-base read returned explicit air with `ok: false` and `reason: not_found`.
The original reconciliation code incorrectly treated that known empty cell as an unavailable observation.
The current code accepts exact air observations and still rejects unreadable targets.

`.runtime/alpha/a12-final-03/receipt.json` retains the subsequent house-review failure.
Mechanical checks accepted the ten exact air cells.
The real model reviewer still interpreted their raw `not_found` outcome as unavailable evidence.
Current model evidence adds `observation_available` while preserving the raw result and its digest.
The verifier instructions explain that exact air observations establish clearance.
The fresh final house case passed in `.runtime/alpha/a12-house-final/`.

The first publication harness omitted the supervisor clock argument after real cognition had committed its intent.
Its failed receipt remains under `.runtime/alpha/a12-publication-01/receipt-prior-*.json`.
The corrected harness continued from that same checkpoint and passed.
The fresh second run passed all nine current checks.

The A11 failed house receipts remain unchanged.
`hs_err_pid13288.log` remains untracked and preserved.
The sibling pm_next changes remain outside this delivery.
A13 remains pending.

## Foundation and dependency checks

The full root suite passed 397 tests in 453.890 seconds.
Its log is `.runtime/alpha/a12/foundation-delivery.log`.
The final focused suite passed 83 tests in 6.539 seconds after the control crash repair.
Its log is `.runtime/alpha/a12/focused-final-control.log`.
The first shared-registry fixture collision remains in `foundation-01.log`.
A later suite overlapped a steward grant edit and retained its failure in `foundation-final-complete.log`.
The final run used the corrected fixture and passed.

The control boundary check commits a replacement instruction to Ghost and leaves its old mirror untouched.
The surviving supervisor rejects the old dispatch marker and a stale control write.
A separate read-only check uses the actual full-history native house checkpoint.
It recovers the accepted control row without changing that checkpoint.
Its receipt is `.runtime/alpha/a12/checkpoint-control-read.json`.
These checks establish control durability, rather than new model or gameplay acceptance.

pm-coder passed 22 targeted tests and six subtests, including transport timeout configuration and compaction.
pm-workflows passed 29 driver, context, scope, and checkpoint tests.
pm-minecraft passed ten adapter, runtime-wait, and durable-fence tests.
Its pre-existing `uv.lock` drift rejected an earlier `uv run --locked` command.
The recorded server and checks used its actual `.venv/Scripts/python.exe`; A12 changes no Minecraft package dependency.

The root lock now pins pm-coder `061170544037d323c376acac8d7c241ffacf2685`.
It pins pm-workflows `e3d05a35d84122030d6992afc5d168559435457c`, which consumes that same pm-coder revision.
The server source is pm-minecraft `3c007701f61263a3293280b91a70ab47ef24e17f`.
Installed distribution origins were checked after the locked sync.
The pm-coder manual also passed its strict build and local-link inspection.

## Scope and uncertainty

Fresh observation proves current state, but cannot assign an uncertain change to a specific caller.
The body server can finish an action after the client exits.
A checkpoint cannot restore that external state.
Human persona ratings and visual interpretation are outside this item.
Failed live receipts remain available when a later attempt repairs their gate.
