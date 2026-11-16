# Shared prompt fitting

## Accepted scope

The owner requests shared fitting for shard prompts and other growing prompts.
The internal cognition window must be configurable in YAML, with a 32,768-token default.

## Current graph and problem

Python roles call `workflows.main_cognition.model_json`, then `pm.llm.structured`, then the configured model tool.
Main cognition adds `StoryPrefix` and private grounding through `phases.model`.
Only triage selects shard records against the complete prompt.
Other phases reuse full shards, and the common preflight raises on overflow.
Dialogue already uses `select_rendered_context`. Reflection uses `select_context` before constructing its final payload.
Simulation fits actor history before adding review instructions and correction feedback.

## Target contracts and ownership

- Keep `ContextGroup`, `PromptSelection`, and `select_rendered_context` as the single selection algorithm in `pm/context_render/budget.py`.
- Add typed optional text annotations to composable `Content`. A general adapter fits complete turns, preserving required text, media, roles, and tool contracts.
- Move triage's shard selection policy into `workflows/main_cognition/prompt_context.py`. This module belongs to the cycle that owns `CycleWork`.
- Fit each main-cognition attempt through this adapter, including prose and correction attempts. Render selected views without changing the cycle or Ghost.
- Annotate optional evidence in centralized source renderers. Existing instructions and unannotated task content remain required.
- Fit simulation's typed payloads against their complete actor, review, and judgment prompts through the same selector.
- Add `cognition_context_window: int = 32768` to each chat connection. It caps internal cognitive prompts, not the endpoint's serving limit or dialogue window.
- Keep required overflow explicit. No substring clipping, synthetic evidence, implicit summarization, or state mutation.

## Implementation and verification

1. Add shared annotations, complete-turn fitting, and configurable capacity.
2. Extract and generalize shard selection; connect all phase and prose calls.
3. Connect growing non-shard evidence, simulation, dialogue, reflection, and project decisions.
4. Run small deterministic selector checks and strict type checks.
5. Measure captured real Ghost prompts at multiple windows and make real endpoint calls.
6. Update the conceptual manual and configuration examples; build and inspect HTML.

The real configured model and captured Ghost are available prerequisites. Minecraft world effects are outside this verification.
Retain useful evidence under `.runtime/shared-prompt-fitting/`.
Mandatory content can still exceed a small window. This is a visible configuration/task constraint, not permission to discard instructions.

## Owner additions during implementation

- Optional selection always combines 40% recency, 40% existing salience, and 20% randomness.
- Tasks that must consider every item use complete batches under the same fitted context.
- Attention merges all ratings before choosing global winners per shard.
- Recall scans every candidate, retains supplied supporting originals, then makes one coherent goal/belief judgment.
- Expectation checks batch offered expectations and preserve their observation grants.
- Main triage batches new events only when their required payload overflows.
- Immediate entity-model revisions and holistic character judgments remain coherent; they are not blindly merged.

## Live findings and corrections

The first real cycle reached attention and exposed an unbounded candidate list outside the shard story.
The next attention attempt exposed a single rating payload near 26.5k estimated tokens plus a 5.7k required frame.
Long member records had previously been clipped to `ATTENTION_LINE_CHARS`.
Whole-record rendering now reviews oversized structures through source-linked evidence batches.
Every original member remains on the structure. The review is a prompt-only derived view with explicit coverage receipts.

An initial fixed 1200-token digest ceiling caused repeated compression attempts on the stress input.
It was replaced with a per-batch allowance derived from the remaining attention window.

The live early-triage test still assumed that every memory commit wrote a checkpoint and re-armed critical work after restart.
The test now saves explicitly and checks that its saved assessment survives without re-arming the old input.

A long verification process loaded different module revisions while files were being edited.
Final long-run verification uses a frozen production-source snapshot, with the real endpoint and real workflow kernel.
No resumable mid-cycle production state was added.

## Final implementation notes

The batch planner reserves required context and one complete task first.
It splits the remaining space between optional common context and additional task items.
An earlier allocation filled common context too aggressively and produced near-single-candidate recall calls.
The final planner retained all 109 stress candidates in six real provider requests.

Oversized-structure digests explicitly cite their top-level member IDs.
IDs mentioned inside a summary or linked source are not additional inspected-record grants.
Correction feedback lists invalid IDs and the exact allowed member set.
A structure's stored members are never replaced by its prompt digest.

Focus adjudication also keeps its proposal, state, and current events required while codelet impulses remain selectable.
Simulation history indexing uses the shared 40/40/20 ranking and retains embedding salience for final prompt fitting.
Project decisions keep specifications, fresh world evidence, requested reads, and retrieval catalogs required.
Recoverable archived history can yield space; estimated text counts no longer masquerade as unmeasured binary media.

## Verification receipts

All runtime artifacts are ignored under `.runtime/shared-prompt-fitting/`.
No completion-cache substitution or fake endpoint was used.

| Check | Evidence and result |
| --- | --- |
| Clean companion cycle | `clean-cycle/result.json`, `clean-cycle.log`, `clean-cycle-exit.txt`: accepted tick 808 in 344.516 seconds after startup; process exit 0; all phases completed |
| Checkpoint consistency | Reopened the final saved Ghost successfully at tick 808 with 808 cycle states; no `cycle_work.json` intermediate files |
| Oversized structure, current code | `attention/result.json`, `attention-head.log`, `attention/coverage.json`: all 109 original CSM members covered in two review batches; one final rating call; three real provider requests; 40.578 seconds |
| Complete recall | `recall/result.json`, `recall-live.log`: all 109 candidates present in actual provider prompts; 53 selected; six real provider requests; 24.125 seconds |
| Early triage | `input-triage-final.log`; `.runtime/beta-input-triage/20261116T195715Z-fe5bb8d6/result.json`: ordinary input and consequential correction passed through the real kernel; two real provider requests |
| Simulator generation and review | `simulation/result.json`, `simulation-live.log`: 912 real historical turns supplied; actor generation and immersion review completed; review accepted; 41.891 seconds |
| Real history retrieval | `history-index.json`: 145 turns selected from stored vectors using a real embedding query, at 15,952 of 16,000 estimated tokens; salience IDs match selected record IDs |
| Focus rendering | Deterministic renderer smoke: required goal and event retained; 28 optional impulses displaced; 2,019 estimated tokens under a 3k window |
| Focused deterministic checks | 25 checks passed across context budget/turns, dialogue fitting, global attention winner count, and project context projection; no whole suite launched |
| Static checks | `pyright-final.json`: configured strict check, 60 files, zero errors and warnings; supplemental history-module check also clean |
| Documentation | Strict MkDocs builds passed; generated HTML checked for eight pages and 5,416 local links/anchors, with no broken targets. No browser visual inspection was available |

The clean cycle used `source-clean/`, a frozen production-source snapshot.
A few later changes were narrow renderer/typing corrections, checked separately; obsolete long runs are not represented as successful runs.
The retained-checkpoint long run exposed recall packing and digest-citation defects and was stopped after diagnosis.
The final clean-start cycle and focused real-kernel stress runs are the acceptance evidence.
Minecraft world behavior was not exercised; its context projection received deterministic checks only.
A full simulated month was not launched.

## Handoff

- YAML key: `chat.<preset>.cognition_context_window`, positive integer, default `32768`.
- The existing `override_context: 65536`, concurrency, and HTTP timeout settings remain separate.
- Dialogue and simulation use their serving-window budgets; internal cognitive roles and native codelet context use the configured cognition window.
- Optional context uses 40% recency, 40% existing salience, and 20% randomness.
- Required coverage uses complete batches. Holistic judgments and immediate entity revisions remain coherent.
- A truly indivisible required input that exceeds the configured window still fails explicitly. There is no silent clipping.
- No production mid-cycle resume state was introduced.
- Runtime processes are stopped after verification; launch the next end-to-end run explicitly.
