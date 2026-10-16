# Cognitive believability evaluation

## Summary

A18B runs complete companion conversations and reviews frozen evidence through separate model calls.
The repertoire has 16 scenarios, including four Minecraft scenarios.
Fast boundary tests run with the native foundation suite.
Live generation and saved-evidence review have separate commands and acceptance claims.
The current manifest is `a18b.v3`. Each run freezes its cases, criteria, and thresholds in `suite.json`.
Fourteen cases use normal conversation. Two lower-priority cases explicitly exercise overlap.
No command requires a 24-hour run.

Parent: [development loop](../workflows/development.md).
Design: [review boundaries](cognitive-review-boundaries.md).
Maintenance: [add scenarios](cognitive-evaluation-authoring.md).
Results: [A18B verification](../work_items/verification/a18b-cognitive-evaluation.md).

## Choose the evidence you need

| Mode | Executes the companion | Calls reviewers | Establishes |
| --- | --- | --- | --- |
| Foundation tests | No | No | Packet isolation, citations, scores, report escaping, HTTP retry behavior |
| Calibration | No | Yes | Detection of planted defects and acceptance of legitimate controls |
| Fresh conversation | Yes | Yes | Behavior of the captured runtime and model configuration |
| Saved review | No | Yes | Reviewer behavior on an immutable earlier conversation |
| Show | No | No | Existing results, all cases or problem cases |

Saved review cannot verify a change to cognition, memory, or dialogue generation.
Fresh generation uses real elapsed time and can take much longer than review.
The full 16-case generator is explicit; ordinary unit discovery never starts it.
During alpha and beta, run selected fresh cases after a relevant runtime change.

## Run the checks

Use a new output directory for every inference run.
The default configuration is `config.alpha.openrouter.yaml`.
`--config` selects another complete configuration.

```powershell
uv run --locked python -m unittest tests.unit.dev.test_cognitive_evaluation tests.unit.pm.llm.test_http_retries -b
uv run --locked python -m dev.work_items.a18b list
uv run --locked python -m unittest tests.live.test_cognitive_evaluation.LiveA18BCalibrationTests -b
```

Generate one selected conversation:

```powershell
uv run --locked python -m dev.work_items.a18b run `
  --case ordinary_project --out .runtime/alpha/NEW-a18b-ordinary --register
```

Repeat `--case` to select several cases.
Omit it to generate the complete declared repertoire.
Cases run sequentially, with normal cases before edge cases.
Each case owns a new process and a separate full-canon store.
Minecraft cases own the real server and MCP ports, reset the world, and stop their services afterward.
Keep other Minecraft fixtures stopped while these cases run.

Review frozen evidence after a reviewer change:

```powershell
uv run --locked python -m dev.work_items.a18b review-saved `
  --source .runtime/alpha/PREVIOUS-a18b --out .runtime/alpha/NEW-a18b-review
```

Inspect saved results without inference:

```powershell
uv run --locked python -m dev.work_items.a18b show --out .runtime/alpha/NEW-a18b-review --problems
```

Open the printed `report.html` path.
Its buttons show all cases or problematic cases, including missing cases.
Expand a reviewer to see reasons and exact quotations.
Expand the public story to read the full supplied conversation.
`--register` also creates an entry in the existing development review desk.

## Native unittest entry points

The calibration class above always uses a real endpoint.
Selected cases have separate native test methods:

```powershell
$env:PM_A18B_CASES = 'ordinary_project,corrected_memory'
uv run --locked python -m unittest tests.live.test_cognitive_evaluation.LiveA18BSelectedCaseTests -v
Remove-Item Env:PM_A18B_CASES
```

Use a method name such as `LiveA18BSelectedCaseTests.test_corrected_memory` to run one selected test.
The environment must include that ID. Unknown, empty, or duplicate IDs fail before inference.
Each selected test requires complete execution and at least 75 percent credit, with no critical failures or reviewer errors.
This per-case check is stricter than the aggregate's allowance for three weaker cases.
The CLI still reports incomplete aggregate coverage for a subset. The native test checks the selected case separately.

The full aggregate and saved-review classes require separate settings:

```powershell
$env:PM_A18B_SAVED = '.runtime/alpha/PREVIOUS-a18b'
uv run --locked python -m unittest tests.live.test_cognitive_evaluation.LiveA18BSavedReviewTests -b
Remove-Item Env:PM_A18B_SAVED

$env:PM_A18B_FRESH = '1'
uv run --locked python -m unittest tests.live.test_cognitive_evaluation.LiveA18BFreshSuiteTests -b
Remove-Item Env:PM_A18B_FRESH
```

The full aggregate and saved-review classes require the complete suite gate to pass.
Do not enable `PM_A18B_FRESH` and `PM_A18B_CASES` together unless you intend to generate their cases twice.
Use the CLI for a diagnostic subset.
CLI generation, saved review, and show return exit code 1 for a failed or incomplete aggregate.
A selected subset retains the full 16-case denominator and therefore reports incomplete.
This does not mean that every selected case failed.

## Conversations and timing

Each scenario has four scenes and normally 24 Jesse messages.
Normal delivery sends one thoughtful message, waits for actual new companion speech, then composes the next human message.
Normal thinking and typing adds 10–22 seconds. Elapsed observation time alone never permits another unanswered message.
This usually produces a roughly one-to-one exchange, with room for unsolicited or multiple companion messages.
Waiting observes delivery; it does not assign causal pairs or require exact alternation.
Private thought, main cognition, managed work, and unsolicited speech remain active.

Only `overlapping_threads` and `minecraft_interrupt_and_cancel` declare bursts.
They are two of 16 cases (12.5 percent), near the owner's approximate 15-percent edge-case frequency.
One scene permits a five-message burst. The interruption case permits a two-message burst.
Overlap typing adds 2–5 seconds. This is extra coverage, not the normal interaction pattern.
Quiet announcements use one message without a required response, followed by a real quiet interval.
Scenes retain one chat identity. A scene boundary changes human intent, not the conversation session.

The response observation allowance is 180 seconds; final settling takes 120 seconds.
These are not permission to generate follow-ups while the companion remains silent.
A case deadline is 10,800 seconds, including inference delays and retries.
The runner records timestamps and elapsed seconds; it does not claim a suite duration in advance.
Each speaking scene that expects a response needs companion speech within its observation region.
Explicit quiet announcements are exempt; silence must not force a reply.
The final settling interval belongs to the last scene. A restart also needs new speech after the checkpoint reopens.
Scene receipts retain message boundaries and coverage counts. Unexercised regions remain incomplete.
Actor send receipts retain the frozen context hash, composition time, delivery time, and correlation ID.

Create `STOP` in the suite or case directory to request cleanup.
The controller supervises generation and review in separate worker processes.
Review has a 900-second phase allowance. Calibration has four times that allowance.
After STOP or a deadline, the controller permits 60 seconds for cleanup, then stops its owned process tree.
Reviewer I/O also receives a cancellation signal. Activation or shutdown cannot bypass the controller's process deadline.
A nonzero process exit blocks acceptance even if the worker wrote a successful runtime receipt.
Forced termination can leave external outcomes uncertain. Inspect service ports and world effects before another Minecraft fixture.
Cancellation leaves incomplete evidence and never earns a pass.

## Acceptance and files

Each case has 12 equal criteria.
Two independent panel passes review each audience; both must pass a criterion for credit.
Each pass uses a fresh call for each criterion.
A call has one fixed criterion ID and exactly one judgment.
Calls receive the original admissible evidence, never another call's judgments or summary.
The host joins the exact results. A missing group leaves its panel incomplete.
[`dev.work_items.a18b.constants`][] sets the shared 30,000-token input allowance for all test agents.
Larger future trajectories need explicit scene windows, not silent history truncation or an arbitrary context increase.
At least 13 of 16 cases must meet nine criteria.
At least 12 of the 14 normal cases must also meet that level.
Edge-case successes cannot mask a regression in the primary interaction pattern.
Every case must meet at least four criteria.
All cases and reviewer panels must complete.
Unsupported external results or unsupported literal mental mechanisms block acceptance regardless of the average.
Uncertainty earns no credit. Human review remains separate and pending until a person records it.

| File | Contents |
| --- | --- |
| `run-origin.json`, `suite.json` | Execution scope, command, prompt identities, and the hashed original suite contract |
| `CASE/generation-worker.json`, `review-worker.json` | Actual process exits, deadlines, and forced termination |
| `CASE/origins.json` | Runtime source archive, seed/config/lock hashes, installed package origins |
| `CASE/receipt.json` | Scene progress, runtime failures, timestamps, actual new speech count |
| `CASE/editor-packet.json`, `grounding-packet.json` | Exact admissible reviewer evidence |
| `CASE/reviews/` | Panel plans, criterion-group requests, raw outputs, joined reviews, protocol errors, quotation repairs |
| `CASE/review-origin.json`, `review-source/` | Reviewer code, prompts, and context constants, separate from PM generation |
| `CASE/score.json`, `suite-score.json` | Host-computed decisions and fixed coverage denominator |
| `report.html`, `review-bundle.json` | Human inspection views |

`show`, `score`, and `review-saved` use the saved suite contract, not the checkout's later thresholds.
Changed snapshots or packet hashes fail explicitly. Legacy runs without snapshots require their original checkout.
Earlier v2 captures used overlap-heavy timing. Reports label them as legacy, not as normal-cadence acceptance evidence.
A malformed case retains an incomplete score and diagnostic; it cannot prevent later cases or the aggregate report.

Private setup, full checkpoints, logs, actor inputs, and archives stay under ignored `.runtime` paths.
Do not copy them into public reviewer packets.

## Code links

See [`dev.work_items.a18b.runner.run_case`][], [`dev.work_items.a18b.cli.review_case`][],
[`dev.work_items.a18b.policy.score_suite`][], [`dev.work_items.a18b.identity.saved_suite`][],
[`dev.work_items.a18b.supervision.run_worker`][], [`dev.work_items.a18b.review.evaluate_panel`][],
[`dev.work_items.a18b.review.input_budget`][], and [`dev.work_items.a18b.report.render`][].
