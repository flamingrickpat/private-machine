# A17 and A18 hardening

## Summary

Human review found incorrect speaker attribution, canned failure speech, narrator leakage, and unreliable turn pairing.
These defects affect the evidence needed for further companion work.
A19 remains on hold while the current runtime receives repairs and whole-conversation review.

Parent: [alpha plan](../alpha-plan/index.md).
Related: [A17 evidence](a17-private-loop.md), [A18 evidence](a18-self-model.md), and [asynchronous review](../../subsystems/async-conversation-review.md).

## Owner review

The owner completed four saved reviews under `.runtime/dev-review/feedback/`:

- `20261015-045957-a17-conversation-review.json`
- `20261015-110410-a18-conversation-review.json`
- `20261015-110410-a18-persona-comparison-retry-personas.json`
- `20261015-171416-a18-rerun-conversation-v2-review.json`

The A17 return case received ratings of 1 and a regeneration request because private thought appeared as Jesse's speech.
The other four A17 cases received mostly 4 or 5, with a contamination concern on the outcome case.
The original A18 capability case received ratings of 1 because runtime failure produced fixed recovery text.
The later A18 capability, feeling, and restart cases also received ratings of 1 and regeneration requests.
The decision explanation received ratings of 5.
Matched-persona feedback was broadly positive, with repetition concerns and explicit limits on factual checking.

These are human observations, not automatic factual labels.
The long absence is a valid scenario fact.
Spamuel can object when Jesse returns after months and immediately asks a technical question.
That reaction alone does not establish a routing or persona defect.
Knowledge of elapsed time remains separate from a claim of execution during shutdown.

The owner also reviewed `20261015-hardening-long-gap-02` and accepted the conversation without regeneration.
The saved ratings are naturalness 5, self-explanation 5, immersion 4, usefulness 5, and groundedness 5.
The owner noted three dandelion references and accepted that repetition for now.
Evidence: `.runtime/dev-review/feedback/20261015-hardening-long-gap-02.json`.

## Repairs

| Boundary | Change |
| --- | --- |
| Stimulus dispatch | Preserve source, kind, correlation, and record ID through the public action payload |
| User speech | Resolve actual user-origin dialogue from accepted records; keep private and system events outside the current user line |
| Proactive speech | Permit generation without a current user line |
| Asynchronous delivery | Permit later user messages and cognitive states while an earlier draft completes |
| Evidence validation | Compare source meaning while excluding delivery bookkeeping; retain actual changes as conflicts |
| Multiple messages | Resolve addressed user stimuli together and deduplicate publication by speaking action |
| Follow-up reports | Keep terminal task speech available after an independent interim reply |
| Unexpected failures | Propagate Python and model protocol errors to application shutdown; retain pending input and tracebacks |
| Expected task rejection | Request an agent-written explanation with the real rejection receipt |
| Tool failures | Catch explicit denials and transport failures; let programming errors and cancellation propagate |
| Narration | Let the existing review select only a verbatim speech span; retain the original draft and finding |
| Cognitive conflict | Retry the complete cycle with new state, retain event identity, and raise after three conflicts |
| Startup evidence | Describe an absent shutdown receipt as an unknown ending |
| Time | Use UTC for new cycle timestamps and restrict offline assertions to the actual offline interval |
| Minecraft state | Add one direct read-only observation for position, health, and surroundings without a managed gameplay task |
| Owned work | Signal and join managed workers before stopping services; publish a clean stop only after successful application cleanup |

The original Spamuel seed and user feedback remain unchanged.
No dependency, model deployment, or source grant changed.
The server fixture uses the selected real Minecraft server with isolated character files.
The runner records a world reset and actual initial observations.

## Engineering boundaries

`tests/unit/workflows/public_speech/test_attribution.py` covers mixed speakers, proactive speech, overlapping delivery, independent follow-ups, and action replay.
It also checks verbatim narration selection, transport-error classification, retry event identity, and application shutdown after broken public input.
The shutdown check uses the real module thread and application error path.
It requires no model response and creates no companion recovery speech.

Existing publication and wiring tests now require valid accepted stimulus records.
Their former fixed recovery expectations were replaced with explicit error and pending-input assertions.
Those checks establish boundaries, not conversational quality.
The new [conversation runner](../../subsystems/async-conversation-review.md) uses real inference and the full preserved history for behavioral evidence.

The recent-history transform retained 11,202 records.
Its measured last activity was approximately 86,402 seconds before the inspection.
Evidence: `.runtime/alpha/hardening-recent-transform-check/history-transform.json` and `.runtime/hardening-transform.log`.
The original seed remains the authority; the shifted dates identify an experimental fixture.

## Retained discovery evidence

`.runtime/alpha/20261015-hardening-long-gap-01/` is the first diagnostic conversation.
It began during development and does not establish the final source revision.
Jesse repeatedly requested a screenshot before a public response arrived.
The companion chose a managed Minecraft project for that request.
The run retained real observations and its task artifacts.
It ended with ten Jesse messages, no companion messages, and two running task receipts.
Its failure exposed missing managed-worker cleanup in direct application use.
Application shutdown now owns the supervisor's stop and join hooks.
Worker and terminal-hook logic failures reach the application instead of disappearing during cancellation or consolidation.
This discovery led to clearer Jesse instructions and a separate direct body-state route.
The final review must keep this attempt separate from later stable-source runs.

The initial foundation attempt also overlapped source edits.
Its imported modules and newly loaded workflow skills used different source versions.
Its failures remain in `.runtime/hardening-unit-first.log`.
They do not replace a stable-source foundation result.

The completed hardening foundation suite passed 715 unit tests and 75 contract tests.
Evidence: `.runtime/hardening-unit-delivery.log` and `.runtime/hardening-contract-delivery.log`.
The narrator probes retained two failures before the third probe passed generation and review.
Their evidence remains under `.runtime/alpha/20261015-hardening-narration-probe*`.

The first recent-history conversation later failed on an expectation resolution with no observation IDs.
The integrator now receives candidate IDs and their text, then validates its proposal before acceptance.
One explicit model repair can correct a rejected proposal; the runtime never invents its replacement.
The speed-development recent run then exposed an unsupported Minecraft-status claim in an unsolicited startup reply.
The reviewer blocked it correctly; no public message escaped.
Story now receives explicit absence-of-observation context and instructions for concurrent incoming speech.
The captured failed input passed a new real generation and review.
Failed receipts remain in `20261015-211528-hardening-recent-01` and `20261015-220900-speed-recent-02` under `.runtime/alpha/`.
Read [inference settings](../../subsystems/inference-settings.md) for the speed changes and their separate evidence.
The [speed verification record](inference-speed.md) retains the later inventory-review failure and its repaired real check.
