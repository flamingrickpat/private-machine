# A08 verification

## Summary

A08 separates original-evidence retrieval, durable state, attributed beliefs, and public behavior.
Boundary tests cover actual Ghost snapshots and artifact files.
The console gate uses the real Spamuel seed and separate checkpoints.
The active embedding connection uses OpenRouter after the user requested rollback.
The initial CUDA failure and isolated CPU comparison remain separate evidence.

Parent: [experienced memory](../../subsystems/experienced-memory.md).
Related: [consolidation](../../subsystems/memory-consolidation.md), [baseline C02](../alpha-plan/baseline/receipts.md#current-verification), and [model policy](../alpha-plan/model-policy.md).

## Boundary checks

Run:

```powershell
uv run --locked python -m unittest tests.unit.pm.memory.test_experienced_memory -b
```

The current A08 subset contains 19 passing tests.
The combined A08 and Story contract check passes 34 tests in 2.207 seconds.
It verifies older original records after distractors and restart, frozen snapshots, permitted IDs, reality/frame boundaries, and causal frontiers.
It also verifies nested summary boundaries, linked revision conflicts, explicit correction, unresolved conflict, and stale world observations.
Unsupported generated claims and imagined events cannot become factual beliefs.
Deception retains a prior intent and known evidence.
Temporal queries exclude background and simulated execution.
The real Spamuel checkpoint returns 13 executed atomic events for its recorded run interval and none for the earlier offline interval.
The receipt is `activity-real-checkpoint.json`.

Captured Minecraft artifacts produce 12 attributed memories, including 10 historical world records.
Repetition after checkpoint restart returns the same consolidation receipt.
Real OpenRouter retrieval preserves all historical flags.
The receipts remain in `.runtime/alpha/a08/captured-artifacts/`.
This check reuses captured real tool data and performs no new Minecraft action.
Changed progress snapshots reuse unchanged artifact memories after restart.
Long record chunks retain late source evidence.
Small retrieval limits preserve relevance ahead of recency.
Long excerpts retain rare evidence beyond repetitive prefixes.
Belief proposals require the current stimulus grant.
Replacement proposals retain earlier evidence revision guards.
Generated Features cannot supply deception intents.
Internal-state recall and indexing expose accepted affect values instead of generated sensory descriptions.
The same interpretation applies to A07 state records without the new category marker.
The Story boundary preserves the selected action beside the current user stimulus.
The five-message contract remains unchanged.

These checks do not establish model interpretation or public immersion.
The separate live console sequence supplies behavioral evidence.

## Console gate

```powershell
uv run --locked python dev/work_items/alpha_a08_memory_gate.py --config config.alpha.example.yaml --out .runtime/alpha/NEW-a08
```

[`dev.work_items.alpha_a08_memory_gate.run`][] owns the console service and its workers.
The default diagnostic console port is `8094`.
It verifies the selected chat and embedding identities before execution.
It waits for the startup cognitive cycle and matches replies to the submitted correlation ID.
An unsolicited lifecycle reply cannot satisfy a user exchange.

Spamuel receives the codeword `copper lantern` and an evening preference for unsweetened ginger tea.
Six real conversation turns distract the companion before the same checkpoint restarts.
The codeword and relevant drink situation then require their original evidence IDs in the dispatched reply.
A semantic probe uses `bedtime beverage` against the original preference and a competing codeword record.
Correction, imagination, canon attribution, a failed method, and a shared achievement remain separately inspectable turns.

The cheerful C02 control uses its separate persona fixture and checkpoint.
It receives the same codeword, a distraction, and a restart recall question.
Its small seed cannot establish recall from Spamuel's inherited history.
The gate preserves transcripts, snapshots, selected evidence, competing retrieval scores, and real model traces.

The original Spamuel sequence retains codeword record `11212` unchanged through restart.
Its public reply recalls `copper lantern` and dispatches that original evidence.
Preference record `11221` supports the ginger choice after six distractions and restart.
Correction record `11291` creates active mint belief `11293` and supersedes ginger belief `11224` with the same key.
The next public reply chooses unsweetened mint tea.
`primary-review.json` verifies the original IDs, key, supersession link, and preserved statements.

The first clean OpenRouter cheerful gate returns `ok=true` under its token and original-ID assertions.
Evidence remains under `.runtime/alpha/20261004-a08-openrouter-cheerful/`.
Sunny retains original codeword record `7` unchanged and recalls `copper lantern` after restart.
Its dispatched memory evidence includes that original record.
Source review still finds a misleading gap claim and an incorrect restart attribution in the public reply.
The initial automatic assertions therefore do not establish fully grounded C02 behavior.
Memory rendering now retains each source's UTC timestamp beside its original ID.
This repairs chronological context lost before generation.
The timestamped control passes with `ok=true` and no failures.
Its independent source review also passes C02 chronology.
Sunny says the codeword was `copper lantern` and recognizes continuity across restart without a gap or loss claim.
Original record `5` remains unchanged and appears in the dispatched evidence.
Its recorded time, `13:15:22 UTC`, precedes the clean stop at `13:21:07 UTC`.
The restored startup occurs at `13:21:11 UTC`.
Evidence remains under `.runtime/alpha/20261004-a08-openrouter-cheerful-timestamps/`.

## Service failure and comparison

The initial default run failed before its first completed exchange.
The configured embedding endpoint returned 1,024 null values.
The failure also occurred for short direct inputs after the endpoint entered that state.
No invalid vector entered the cache or accepted cognition.

Evidence remains under `.runtime/alpha/a08/` and `.runtime/alpha/20261004-a08-live-01/`.
`incompatible-embedding-response.json` retains the raw response.
`embedding-diagnosis.log` identifies the failing source ID, text length, and digest.
The first failing source was startup Feature `11203`, with 133 characters.
`.runtime/alpha/20261004-a08-live-02/` records the repeated failure after a same-argument service restart.

The same executable and model produced finite vectors on CPU.
Isolated GPU comparisons retained null vectors with prompt caching disabled, flash attention disabled, CUDA graphs disabled, and a newer executable.
Float32 cache and CPU cache comparisons also failed after port isolation was verified.
These observations establish a CUDA-path failure for this deployment.
They do not identify its native root cause.

An isolated CPU console comparison uses port `8083` and an ignored test configuration.
It initially left the deployment on `8081` unchanged.
The first CPU harness attempt counted an unsolicited startup reply and then encountered the still-pending user exchange.
Its failed receipt remains in `.runtime/alpha/20261004-a08-cpu-live-03/`.
The gate now waits for the correct correlation ID and the startup cognitive boundary.
The corrected comparison remains in `.runtime/alpha/20261004-a08-cpu-live-04/`.
Its captured harness omitted the speaker Entity from its restricted semantic probe.
That source grant correctly returned no evidence.
The corrected probe includes Entity `1` and original utterances `11212` and `11221`.
The real CPU result ranks the preference at `0.173739` and the codeword at `0.003231`.
The standalone corrected receipt remains `semantic-paraphrase-corrected.json`.

The initial failed-method reply invented a failed recap despite receiving relevant historical records.
Inspection found that Story omitted the accepted action intent whenever a user stimulus supplied the current message.
The repaired boundary retains both fields and asks recall actions to identify a supported episode.
`.runtime/alpha/20261004-a08-openrouter-recall/` repeats recall from an exact copy of the same checkpoint.
The source and copied checkpoint hashes must match before the worker starts.
Original failed replies remain available beside the new continuation.
The repeated generic failed-method reply still invents the recap failure after the action-intent repair.
Original user report `537` instead describes a wrong-file attachment followed by confirmation that it remained wrong.
`primary-review.json` preserves that comparison and the original source text.
The remaining public-grounding defect belongs to A09.
The accepted reply remains an attributed utterance and cannot establish a factual belief.
The OpenRouter continuation again chooses mint correctly but incorrectly describes the actual correction as hypothetical.
Its accepted belief state still contains the explicit correction and supersession link.
Plant-status recall agrees with original user reports and keeps the diamond hypothetical.
Its selected evidence relies on earlier generated plant callbacks, rather than the original user status report.
Neither result establishes a companion plant-care action.
`final-live-review.json` retains these source comparisons and the verified Story action inputs.
`historical-retrieval.json` compares the generic question with a focused query about the actual wrong-file follow-up.
The real OpenRouter query ranks cluster `1332` first and retains original user report `537` in its source chain.
The generic question instead favors recent recall questions and generated replies.
This establishes retrievable source evidence, rather than a repaired public answer.
The first cheerful worker overlapped the Story interface edit and raised an incompatible-schema error.
Its receipt remains in the CPU comparison folder.
The clean OpenRouter control starts fresh workers with one interface version.

## OpenRouter rollback and fallback

The user requested restoration of OpenRouter embeddings instead of CPU deployment.
`config.alpha.example.yaml` now reuses the existing OpenRouter embedding entry.
Its model is `perplexity/pplx-embed-v1-0.6b`, with 1,024 dimensions.
The credential reference is `OPENROUTER_API_KEY`.
Chat remains local Qwen on port 8080.
Laya, Minecraft, and console settings remain unchanged.

`openrouter-probe.json` records finite vectors for the three original captured inputs.
For `bedtime beverage`, the remote service ranks original preference `11221` above codeword `11212`.
Their weighted scores are `0.166790` and `0.002695`.
`models-embeddings-openrouter.json` records the active model catalogue identity.
`openrouter-rollback.json` records the configuration change and unchanged chat mapping.

The running continuity sequence learned through the temporary CPU service.
Its checkpoint restart reloads the restored OpenRouter connection.
Original state and evidence IDs remain the continuity boundary.
The temporary CPU service remained independently owned.
It stopped after the OpenRouter restart and released its unused memory.

A permanent provider error now preserves scoped lexical and recent retrieval.
`memory_trace.semantic` records `status=unavailable`, the error, and `fallback=lexical_and_recent`.
Cancellation and local filesystem errors still propagate.
`default-fallback.json` records the actual CUDA failure and original codeword `11212` as the top fallback result.
The invalid response never becomes a cached vector or a remembered fact.

Full OpenRouter preparation passes with eight scoped workers in 260.105 seconds.
`openrouter-index-prepared.json` records 8,138 eligible records and 8,143 indexed chunks.
No chunk remains missing.
The request model namespace keeps these vectors separate from the local comparison cache.
The interrupted sequential preparation retained its completed cache entries.
The bounded parallel run reused them and completed the same accepted seed snapshot.

## Foundation and documentation

The final foundation run passes 316 tests in 597.241 seconds.
`foundation-final.log` records the complete root unittest suite.
The workflow package passes 199 tests in 68.00 seconds in its own virtual environment.
`workflows-package.log` records that unchanged package check.
The broader A08, cognition, module, and Story subset passes 131 tests in 422.754 seconds.
The final prompt, timestamp, and memory contract check passes 80 tests in 3.192 seconds.
The final strict MkDocs result and HTML link inspection remain recorded with the session evidence.

The seed SHA-256 remains `0071b51dda2155a1f635375c1b604be0c2e6159983889ed5b070be4de11af0bd`.
The workflow pin remains `7c9fab4381ef77ef3f63ee9ef669a993c8df7452`.
The pm-coder pin remains `ac8327baf669dad54ea9f91580f2bed1023825b4`.
No Minecraft reset or console deployment change forms part of this item.
The authorized embedding rollback is the sole deployment change.

## Remaining acceptance

The unused local CUDA endpoint still has its recorded defect.
C01's real note task and public-reply recovery remain with A09.
C02 passes for Spamuel and the timestamped cheerful control with original evidence and grounded codeword claims.
The first clean cheerful control's misleading chronology remains a preserved failed source review.
Public voice, generic historical recall, unsupported prose, and the earlier cheerful Story rejection remain A09 concerns.
Human immersion ratings remain with the operator.
The second conscious loop remains A17 work.
