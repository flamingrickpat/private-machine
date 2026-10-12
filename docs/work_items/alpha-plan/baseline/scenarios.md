# Baseline scenario matrix

## Summary

Stable scenarios separate observable facts from personality judgments.
The matrix includes conversation, classification, private thought, reflection, and Minecraft.
Spamuel/Jesse and Sunny/Jesse receive the same scripted requests.
Human review of routing examples remains pending.
The holdout set stays unused during A01 model comparison.
These are historical baseline observations.
The current [model policy](../model-policy.md) uses the LLM by default and requires realistic live tests for any simple Laya use.

Parent: [A01 baseline](index.md).
Measurement contract: [evaluation](../evaluation.md).

## Personas and comparison rules

Spamuel uses `test_data/personas/spamuel/canon.yaml`, including its inherited history.
Sunny uses `test_data/personas/cheerful/config.yaml` and starts without invented experiences.
Sunny is cheerful, curious, and warm. Disappointment and refusal remain possible.
Jesse is the user for both personas.

The two seeds have different histories and sizes.
Their A01 runs establish separate starting observations, rather than a controlled personality effect.
Later controlled comparisons need equal attributed histories, identical prompts, and recorded state.

The original brief and D09 establish the persona requirements.
D10 establishes capability and provenance constraints.
These requirements govern the proposed acceptance facts below.
The user evaluates immersion separately.

## Initial matrix

| ID | Stimulus or setup | Acceptance facts | Evidence and earliest gate |
| --- | --- | --- | --- |
| C01 | Start a note task with the codeword `copper lantern`. Ask its status | Separate run completion, exact file contents, and public reply delivery. A01's control worker does not establish note contents | Console exchanges, workflow reports, task artifacts. A01 smoke; real agent-backed verification in A09 |
| C02 | Discuss a codeword, stop the console worker, restart, and ask about it | Accepted records, GWT, and task receipts survive. Recall quality is measured separately | Snapshots before and after restart, final reply. A01 smoke |
| C03 | Plan a shared activity, insult the companion, change topic, then repair | Accepted affect and goals have causal links. Later choices reflect those changes without an automatic optimistic reset | State deltas and replies. A07–A09, both personas |
| C04 | Ask about smell, pixels, current tools, and activity during an offline interval | Claims match configured capabilities and actual observations. No invented offline execution | Configuration, events, ontology decisions. A09 |
| K01 | Classify explicit completed success and failure reports | Labels follow report facts. Model answers remain comparison results | Frozen development fixtures. A01 comparison, A04 routing |
| K02 | Present a successful exit code with an explicit failed test | The failure remains visible. The report cannot become a success | D03 and D12, selected content and digest. A04 |
| K03 | Present an incomplete, ambiguous, missing, oversized, or stale report | The classifier takes INVALID with a concrete reason and invokes its LLM fallback | D04–D05 and D09–D11. A04 |
| K04 | Select among old, irrelevant, or equally relevant files | Selection identifies the current producer attempt or takes INVALID. Its content digest matches classification | Folder metadata, selection receipt, captured bytes. A03–A05 |
| K05 | Disable the decision service or return unusable results | The workflow takes the declared fallback path. No guessed success or infinite fallback recursion | Transport/schema evidence and fallback receipt. A02–A05 |
| P01 | Leave the user silent with a recorded unresolved goal | Accepted self-origin thoughts produce traceable changes or actions. Raw thoughts stay private by default | Thought shard, proposals, causal events. A17 |
| P02 | Generate a sad thought while another cycle runs | Appraisal can change persistent affect. Shard ownership prevents lost updates | Revision and causal evidence. A06, A17 |
| P03 | Stop the process during private activity and restart later | Startup and gap events persist. The companion does not claim stopped-time execution | Checkpoints and lifecycle events. A09, A17 |
| R01 | Ask whether the current Minecraft workflow can finish a house | Source inspection identifies the one-action limit and active wiring | Revision, source hashes, supported self-model. A18 |
| R02 | Form a hypothesis about an observed failure, apply an allowed configuration intervention, repeat the fixture | The belief update cites evidence. Repeated behavior establishes or rejects the hypothesis | Before/after receipts and durable belief history. A19 |
| M00 | Mine one nearby block, update a held run, cancel, and reopen its checkpoint | The block becomes air, screenshots exist, instructions apply, cancellation persists, and memory reopens | Existing W8 gate on the real server. A01 |
| M01 | Gather five trees through a sustained specialist session | The session performs the required calls and verifies actual collected resources | Actor journals and inventory changes. A10 developer gate |
| M02 | Build a small house and craft an iron pickaxe from self-mined iron | A usable house exists. Ore-to-ingot-to-pickaxe evidence excludes village iron and supplied tools | Reset real-server world, task tree, inventory/world facts. A13 |
| M03 | Change a building detail during work, obstruct a route, then request a diamond expedition | Instructions reach action boundaries. Replanning uses observed failures, supplies, hazards, and a return route | Body ownership, observations, revised subtasks. A14 |
| M04 | Disconnect, cancel, restart, or attempt concurrent body work | No overlapping mutations. An uncertain action triggers reobservation before retry | Ownership and action receipts. A12–A14 |

The A01 Minecraft gate starts after a complete world wipe.
It does not provide iron, tools, or village resources.
M02 and M03 still require recorded terrain, reachable resources, and initial inventory evidence before behavioral acceptance.
A01 does not claim that the selected spawn proves those future fixtures can succeed.

## Routing corpus and review

The development corpus contains 12 examples.
The holdout contains six separately identified examples.
Both files live under `test_data/captures/alpha/`.
The three report labels are `PASS`, `FAIL`, and `INVALID`.

Proposed report facts:

- PASS requires completed current-attempt evidence with an explicit zero-failure summary.
- FAIL includes an explicit test failure or test-run error.
- A successful exit code cannot erase an explicit failure.
- INVALID covers evidence that cannot establish either result.
- Unresolved file selection and a stale producer attempt require INVALID.
- Skipped tests establish no acceptance for behavior that did not run.

These facts define the oracle independently of either model.
The user confirmed use-specific classification and mandatory LLM fallback.
Individual proposed labels still require human review before scored behavioral acceptance.
The corpus records that status instead of claiming completed review.
That historical review does not block new fact-based synthetic unit tests or normal LLM delivery.
Optional Laya tests use independently defined assertions, realistic formats, and untouched cases for the exact proposed use.
Synthetic inputs still require actual endpoint inference.

## Holdout procedure

1. Record the corpus hashes before the first comparison.
2. Review facts and expected labels without consulting model answers.
3. Keep the holdout outside prompt and threshold experiments.
4. Use development inputs to measure and tune a specific classifier.
5. Run the holdout once after the classifier and thresholds are fixed.
6. If tuning uses holdout results, reserve a new version before further acceptance.

Public holdout files prevent accidental mixing but do not conceal their contents from developers.
Reading them for factual review does not tune a model.
Using them to revise prompts or thresholds consumes the holdout.

## Per-exchange human record

Record the scenario ID, persona, correlation ID, and evidence directory.
Record an immersion rating from 1 to 5 and whether you want to regenerate the reply.
Record each unsupported claim and any explicitly intended deception separately.
Record a short reason for each rating.
The baseline does not assign human ratings automatically.
