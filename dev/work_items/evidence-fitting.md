# Evidence fitting and private inquiry prompts

Owner accepted this design on 2026-11-17.

The observed path was thought-stage construction, structured-output repair,
internal prompt fitting, evidence review, and repeated compression. A captured
expectation-review prompt measured 28,103 estimated tokens against a 27,852-token
ceiling. Rejected answers occupied about 7,057 estimated tokens. Full grants
appeared both in inquiry state and at the outer request boundary. A 251-token
overflow eventually triggered a 1,283-token digest allowance and over 319
compression passes without convergence.

The accepted boundaries remain unchanged: the host owns complete grants and
validates citations; source records remain immutable; digests are derived
evidence. Prompt construction exposes one named InquiryPromptGrant with scope,
observer, and permitted evidence IDs instead of persistence fingerprints.
The remaining graph and source evidence stays available to the model.

Structured repair retains original task turns, the latest rejected answer,
and distinct reported constraints. Existing rejection callbacks retain failed
answers. Compression targets the mandatory prompt's measured deficit plus
attribution headroom. Three successive attempts without useful reduction fail
explicitly with retained receipts. Network retry and cancellation remain at
their existing boundaries.

One-node discriminatory operations receive explicit instructions that original
observations belong in evidence_ids, not graph comparisons. They may state an
outcome without claiming discrimination progress. The validator is unchanged.

Verification uses captured synthetic-persona evidence as input to the real
configured OpenRouter endpoint. Focused deterministic annotation and span
checks do not certify cognitive quality. Evidence and type-check comparison
are retained in .runtime/evidence-fitting-20261117/. Full simulation quality
and semantic fidelity of summaries require continued real-run inspection.


Verification on 2026-11-17:

- Four focused deterministic checks passed (repair state, alternating errors,
  annotation preservation, and complete ordered spans).
- Real OpenRouter/DeepSeek fitting of captured evidence passed. The production
  compression wrapper selected 2,465 estimated tokens within a 6,484 ceiling.
- A captured one-node expectation review passed the unchanged graph validator
  on its first real answer: 10,085 estimated prompt tokens, no repair calls,
  progress none. The captured prompt lacked original_source_ids; the gate
  therefore prohibited new-evidence acquisition claims rather than inventing
  that authority. This direct check was followed by a real thought-stage workflow/kernel run.
- Changed reduction, graph instruction, and deterministic-test modules pass
  scoped strict typing. The two legacy modules retain existing diagnostics;
  the recorded baseline comparison found no new diagnostics.
- The actual thought-stage workflow and kernel completed on reconstructed
  captured inputs, through the configured real endpoint. The unchanged validator
  accepted information_gap with progress none. The saved request envelope and
  captured source candidates supplied the inputs; original_source_ids remained
  unavailable and no new-evidence acquisition authority was invented.
- Long-run immersion, complete application acceptance, and the compression-stagnation
  failure path have not been exercised live. No model-quality certification is
  claimed. The month simulation has not been resumed: the earlier controller
  restart command was rejected by automatic approval review.

Strict MkDocs build exited 0. Rendered cognition policy text and local HTML
navigation targets were inspected successfully.

## Retrieval fallback correction

The subsequent eventful-month-20261117-a attempt exposed an untested batching
path. Its required shared context measured 26,151 tokens and the largest task
needed 3,415 against a 27,852 ceiling. The fallback incorrectly reserved half
the input for a task, leaving 13,926 for shared evidence. That artificial
deficit drove grounding into a 256-token allowance and the stagnation guard
correctly rejected the unsuccessful compression. The earlier focused gates
did not exercise batch_internal_prompt; they did not establish this path.

The corrected fallback prepares complete single tasks, reviewing one only
when it exceeds the real input window. It reserves the largest measured task
before fitting shared evidence. All final tasks receive the same frozen
shared context, and retain their identities and coverage contracts.

A focused live gate uses captured shared context and two task packets from
the failed run. It invokes production batch_internal_prompt with the real
configured endpoint and runtime. It checks that both tasks remain complete,
both prompts fit, and shared context is identical across batches. Receipts:
.runtime/retrieval-budget-20261117/. This gate is not a full retrieval cycle
or a new month-long simulation. The failed run remains preserved.


Retrieval correction verification:

- The final replay-disabled real-endpoint gate required one completion request.
  It produced batches of 20,066 and 24,837 estimated tokens against 27,852.
  All task packets remained complete and shared context was identical.
- Four focused deterministic checks passed; model_json.py passed scoped strict
  typing. No new month run or complete retrieval cycle was launched.
- The reported 256-token target was not below the estimator's absolute floor;
  it was an unsuitable evidence allowance caused by the equal-split fallback.
  The observed initial failure was 26,151 shared tokens plus a 3,415-token task.
- The unrelated httpx event-loop shutdown warnings are outside this correction.

The correction passed the strict documentation build (exit 0). The rendered
batching policy and local navigation targets were inspected.
