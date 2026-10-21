# PM classification diagnosis

## Summary

Classification retains concrete connection, context, source, and answer errors.
Use those receipts before changing prompts or service configuration.
A05 preserves its failed development runs and records each repair.
The final gate restores the optional service after acceptance.

Parent: [PM connections](pm-connections.md).
Evidence: [A05 verification](../../work_items/verification/a05-pm-classification.md).
Boundaries: [classification recovery](classification-recovery.md).

## Diagnose a run

| Symptom | Diagnostic and next step |
| --- | --- |
| `missing_connection` | Compare the manifest name with configured chat presets. Supply that connection without silently substituting another classifier. |
| `endpoint_unavailable` | Inspect the redacted session reason. Check `/models`, actual readiness, startup script, and active listener identity. |
| No serving context | Set a positive integer `context_window` for a remote endpoint without compatible `/props`. Use the deployed serving bound. |
| `stale_output` | Inspect the latest named producer attempt. Obtain a new valid producer result before classifying again. |
| `unsupported_evidence` | Compare returned quotes with the retained captured value. Keep INVALID when the quotes do not match. |
| `agent_failure` | Inspect the agent trace, deadline, provider response, and compaction checkpoint. Do not replace a failed live run with a fixed answer. |

The direct path never clips required evidence to fit a context bound.
A suitable complete short artifact can use direct classification.
A longer artifact needs the isolated agent within its configured snapshot and session limits.
Compaction summarizes accumulated execution and retains the original request.
It cannot repair an oversized initial instruction with no generated execution.

## Preserved development failures

The first run observed `/models` while `/props` still returned a loading response.
Inspection found two identical chat restart wrappers competing for port 8080.
The session stopped A04's duplicate restore wrapper and retained the existing wrapper.
Both used the existing `start38_uncensored_nvfp4.ps1` configuration.
The failed discovery receipt remains visible.

The publisher initially compared a role session ID with a workflow run ID.
The repaired publisher uses retained task/item scope and reports the journal's run ID.
A boundary fixture initially used an unsupported `goto` failure action.
Its corrected manifest uses the existing `route_to` contract.
The first foundation run rejected the manifests' inherited 999-attempt warnings.
Both demonstrations now declare a two-attempt bound.

The changed-report development case exceeded the direct budget with a complete verbose A01 log.
Its handler supplied an unsupported quote, so strict validation retained INVALID.
The repeat uses a complete short A04 report without clipping.
The separate long case still requires actual inspection, context rejection, summary inference, and resumed evidence.

Review also found a shared-owner activation conflict.
The first cognitive call attempted activation while classification already held the same ready alias.
The lazy stage model now reuses that alias.
The final live Story run checks this case under an active lease.

## Lifecycle and evidence correction

The final gate temporarily stops the optional Laya server and pauses its restart wrapper.
Cleanup resumes that same wrapper and restores its existing server configuration.
Lifecycle identities and events remain under `.runtime/alpha/a05/`.
Chat 8080, embeddings 8081, optional Laya 8082, and console configuration remain unchanged.
No Minecraft reset occurs.

The session accidentally invoked A04's historical HTML checker against its original output directory.
It restored `a04/html-inspection.json` from the preserved A04 session receipt.
The original `a04/classification.png` bytes are unavailable after regeneration.
The replacement PNG shows the current site, rather than the original A04 render.
`a05/evidence-correction.json` records this limitation and the separately retained rerun.
A04 inference and failure receipts remain unchanged.
