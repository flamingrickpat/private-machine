# Beta immersion repair: research and implementation handoff

Status: proposed implementation details; owner-approved behavioral direction. Written 2026-11-16.

Audience: a cheaper coding model working in small, sequential sessions. This is an engineering plan, not a description of implemented behavior.

## Owner intent

The owner wants a usable, immersive, enjoyable companion with freedom to act and realistic social restraint.

- Preserve the distinction between external evidence and private theories in records and prompts.
- Reuse the existing ontology where possible. Investigate a typed feature classification only where existing contracts are insufficient.
- Model inhibition through self-image, intentions, and an uncertain model of the other person.
- Do not require a new external stimulus or an approved reason before autonomous speech.
- Do not install message quotas, fixed cooldowns, or a universal permission gate.
- Allow attention to move elsewhere: poems now, Minecraft and other external activities when available.
- Make review sensitive to the actual conversation and the character's intent.
- Retry recoverable external/model failures until success or cancellation. Fail visibly on internal defects.
- Keep technical failure diagnostics outside cognitive experience. Never disguise a fixable bug as character drama.
- Use repeated month runs as the primary feedback loop. The owner can inspect after roughly six hours and continue or stop.
- After a satisfactory 24-hour run, consider specialized behavioral tests. Do not require them before the next month attempt.

“Cheaper model” means the model executing this plan. It does not authorize changing PM's configured inference models.

## Read first

Read root AGENTS.md, the project-documentation skill, and the simple-english skill. Read the glossary and naming rules before proposing new concepts.

Consult `docs/work_items/process.md`, `docs/workflows/kanban.md`, and `dev/kanban/board.json`. Do not mark existing beta cards complete from this plan.

Preserve unrelated working changes. At plan creation, `.gitignore` was modified. No product files were changed for this plan.

Evidence starts in the ignored root file `investigation-spamuel-monologue.md`. The saved run is `.runtime/simulation/runs/eventful-month-20261115/`.

Use the existing checkpoint read-only. Never overwrite the source persona or resume the defective checkpoint as a clean comparison.

## Current evidence and limits

The report records four Jesse messages and fourteen Spamuel messages. Ten follow the last Jesse input.

Checkpoint inspection established these separate publications:

| Utterance ID | Action ID | Tick | Text |
| --- | --- | --- | --- |
| 12407 | 12400 | 876 | That was real. Filed. |
| 12456 | 12449 | 877 | That was real. Filed. |
| 12548 | 12538 | 886 | That was real. Filed. |

All share user frontier 11592. Each has an empty reply correlation and accepted review. These are separate actions, not one utterance replayed.

Seed record 983 contains “one video good, or did it rot your brain”. Its provenance is Migration and its date is May 24. Historical-context misuse is plausible; the exact generation path remains unproven.

Record 13071 reports a managed note failure: “The requested note bytes did not pass verification.” Record 13102 appraises that failure as a nameless clerk and resentment. Technical failure entered cognitive interpretation.

The `(B)` source, expectation 11652 lifecycle, and full causes of autonomous action selection remain unresolved. A thought claiming that persistence failed is not persistence evidence.

## Current module graph and contracts

This is a source map, not a complete traced execution graph. Research must complete the missing edges before structural edits.

| Responsibility | Existing source and contract |
| --- | --- |
| Evidence classification | `pm/memory/evidence.py`: `EvidenceEnvelope`, `EvidenceData`, `memory_kind(raw) -> str`, `evidence_role(view, raw) -> str` |
| Record provenance and scope | `pm/records/base.py`: `Provenance`, `RecordScope`; `pm/records/enums.py`: `SourceKind`, `FeatureType` |
| Partner beliefs | `workflows/entity_context/schemas.py`: `EntityClaimEvidence`, `EntityClaimDraft`, `EntityContextClaim`, `EntityClaimView` |
| Private thought | `workflows/inner_thought/host.py`, `pipeline.py`, `graph_outcomes.py`, `perspective.py`, and stage prompts |
| Shared decision evidence | `workflows/public_speech/situational_guidance.py`: `SituationalGuidance`, `SituationSource`, `RecentMove`, `build_situational_guidance`, `render_situational_guidance` |
| Main decision | `workflows/main_cognition/phases/decide.py` and `phases/prompts/decide.yaml` |
| Ordinary interaction | `workflows/interaction/host.py`: `Interaction.remaining_inputs`, `handle_batch`; `steps.py` |
| Public review | `workflows/public_speech/speech_review.py`: `SpeechReview`, `ReviewFinding`, `select_spoken_text`; `speech_review.yaml` |
| Durable speech | `workflows/public_speech/publication.py`: `commit_public_reply(...) -> int | None`; `reply_recovery.py` |
| Technical failure | `workflows/note_task/steps.py`; `pm/runs/supervisor.py`: terminal notification and worker error handling |

Observed execution: accepted action -> generated draft -> accepted review -> committed utterance -> published timeline row. Three different actions repeated the same line.

Current publication checks action completion and resolved covered user inputs. It does not establish conversational novelty for autonomous actions without covered inputs.

Current evidence roles already distinguish user reports, companion utterances, world observations, runtime events, internal states, interpretations, and imagined material.

`OBSERVED_EVIDENCE_ROLES` includes internal state and runtime events. “Observed” therefore does not mean “external evidence about Jesse.” Audit the consumers before changing this set.

Current speech findings are advisory. Only `false_task_result` causes a hard block. Preserve that distinction unless a separate design explicitly changes it.

## Research pass: produce evidence before edits

Keep research bounded to these failures. Write findings under a fresh ignored `.runtime/beta-immersion/<timestamp>/` directory.

1. Record revision, dirty paths, run identity, endpoint identity without secrets, and exact inspection commands.
2. Trace all fourteen utterances to action IDs, originating workflow, source IDs, prompt receipts, review, and delivery.
3. Trace record 983 and the `(B)` source through retrieval, rendering, selection, and final speech.
4. Inspect attribution, timestamps, scope, evidence role, and whether each historical line was presented as current input.
5. Trace each duplicate action back to its decision and current intentions. Identify why prior publication did not change the choice.
6. Trace expectation 11652 through actual revisions and accepted update receipts. Separate missing evidence from failed persistence.
7. Trace record 13071 to the note request, produced bytes, verification rule, exception handling, and cognitive ingestion.
8. Trace how thread errors reach the application supervisor. A logged worker exception is insufficient if the application continues as healthy.
9. Inventory every affected renderer and private feature producer. Identify where existing evidence labels disappear or change meaning.
10. Read existing self-image, entity-context, recent-action, quiet-choice, and attention-redirection contracts before designing inhibition.

Print selected named fields, not complete model dumps. Embeddings and expanded metadata make evidence unreadable.

Deliver a compact table: symptom, record/prompt evidence, owning symbol, cause, confidence, smallest repair, remaining uncertainty.

For cognitive research, consult existing repository references first. If external research is needed, use primary papers and cite exact sources.

Research questions: source monitoring; action inhibition from anticipated social consequences; uncertainty in theory of mind; attention redirection without external reinforcement.

For each useful source, record the supported claim, limits, and the proposed computational approximation. Do not claim neuroscience equivalence.

No implementation depends on proving a universal psychological theory. The owner-selected behavior is the engineering target.

## Proposed architecture

Keep the existing package responsibilities. Do not add a parallel conversation controller or another memory store.

### Evidence and interpretation

Separate three questions: who produced a record, what kind of evidence it contains, and when/scope it applies.

A self-produced utterance is a real speech event. An externally supplied report can still be false. A private feeling is real internal state, not proof of another person's reaction.

First repair propagation of existing provenance, evidence roles, scope, and time. Do not collapse them into a binary real/self flag.

If strings permit inconsistent contracts, propose `EvidenceRole(StrEnum)` in the existing evidence owner. Preserve current serialized values where meanings match.

Proposed signature in that case: `evidence_role(view: Snapshot, raw: RecordBase | Mapping[str, object]) -> EvidenceRole`.

Do not add a competing feature-origin field until the inventory demonstrates a missing distinction. Specify its producers, mandatory values, migration, and consumers before implementation.

Prompts must distinguish current messages, historical quotations, private hypotheses, observed internal state, and world observations. Preserve source IDs and dates through summaries.

Repetition of a theory must not increase its external evidential status. Inferences about Jesse remain attributed estimates.

### Character-driven inhibition

Extend shared `SituationalGuidance` only where existing evidence is missing. Reuse actual recent moves, partner claims, self-image, and current intentions.

If needed, propose one typed `SocialConsequenceAssessment` in the existing workflow schema owner. Fields must name candidate action, supporting sources, anticipated impact, uncertainty, and chosen alternative.

Finalize exact field types after reading current action and quiet-choice contracts. Do not create a second action enum or store free-form dictionaries.

The decision considers: what have I already said; has the partner answered; what do I want; how do I want to come across; what might this cost them?

It can choose speech, quiet, or another available activity. It can still choose risky, awkward, playful, or insistent behavior when consistent with the character.

“She might find this needy” is a forecast, not an observed annoyance. Silence alone must not establish rejection or a relationship change.

An illustrative result: “I've sent two follow-ups. Another could feel demanding. I'll work on the poem and leave her room.”

Do not require this wording. Do not make poems a mandatory fallback. Reuse real available activities and existing private-loop scheduling.

Place this reasoning before action selection in both main and ordinary paths where applicable. Retain a compact decision receipt, not a new recurring evaluator loop.

### Conversation-aware review

Supply recent public turns, timing, pending outreach, candidate intent, and the user's explicit tone request.

Review the candidate as the next conversational move. Ask whether it answers an old memory, repeats a completed move, invents a reaction, or exposes technical diagnostics.

Preserve chosen lies, metaphor, teasing, and character flaws. Evaluate continuity and attribution without enforcing generic politeness.

Use advisory findings for social judgment. If a draft reveals a new material problem, propose a bounded return to existing character reconsideration, including a quiet outcome.

That return path changes behavior and requires an explicit contract before coding. It must not become indefinite regeneration until the character sounds agreeable.

Do not use exact-text deduplication as the main fix. Legitimate repeated words can occur in distinct contexts.

### Failure ownership

Recoverable transport/service errors and valid-but-malformed model samples retry at the smallest owning boundary, with delay, visible status, and cancellation.

Invalid schemas, broken internal invariants, and incorrect note verification logic must retain tracebacks and terminate the application through its existing fatal-error path.

Permanent authorization or configuration failures must surface immediately. Repeated calls cannot repair them.

Reconcile uncertain external writes before retrying. Never repeat completed work merely because telemetry failed.

Keep technical receipts in operator diagnostics. Do not convert them into experiential features, appraisal stimuli, or partner beliefs.

Preserve genuine world outcomes. A blocked Minecraft path can inform cognition; an adapter exception cannot masquerade as an in-world rejection.

Do not delete historical failure evidence. Exclude technical records from new cognitive projections, with an explicit migration or classification plan where necessary.

## Ownership, persistence, and asynchronous boundaries

MemoryStore remains the canonical record owner. Existing write batches retain atomic commits and revision checks.

Entity-context ownership remains responsible for accepted partner claims. Inhibition cannot directly rewrite the partner model from its own prediction.

Shared situational guidance remains a read-only projection. Main cognition owns accepted cognitive decisions; ordinary interaction retains its existing limited authority.

Publication still commits before bus delivery. Preserve utterance-ID delivery deduplication and restart recovery.

New decision evidence must carry source revisions through asynchronous generation. Reuse existing stale-context checks when a partner message arrives during deliberation.

Operator errors travel to runtime supervision and diagnostics. Cognitive evidence travels through accepted records. Do not merge these flows.

Record decision source IDs, candidate, alternative, and review findings so the owner can explain speech or quiet without another model call.

## Implementation order and completion gates

Do not create dozens of tasks. Complete these coherent stages, with a handoff after each.

| Stage | Deliverable | Completion evidence |
| --- | --- | --- |
| A: trace | Evidence table and exact target contracts | All three duplicate chains, historical stimulus chain, and note failure chain have receipts; unresolved claims remain labeled |
| B: runtime repair | Correct note behavior and fatal/retry propagation | Actual note bytes inspected; actual defect reaches fatal supervisor; recoverable operations remain cancellable and visible |
| C: evidence boundaries | Typed classification where needed; consistent rendering | Captured real prompts preserve attribution/time; private claims cannot become external observations through summaries |
| D: inhibition | Shared evidence and character decision changes | Real decisions can choose quiet or another activity without a quota; new user input can change the choice |
| E: review | Contextual advisory review and any accepted reconsideration contract | Review identifies the duplicate/historical-context failure without banning persona behavior |
| F: month loop | Fresh month attempt and owner-readable evidence | Honest status at owner inspection; no claim of completion from uptime alone |

Before stages C–E change public types or persistence, complete the exact field/signature diff and migration plan. Present genuinely unresolved design choices to the owner.

Routine local repairs within the selected design need no repeated approval. Do not redesign adjacent subsystems to simplify a stage.

## Verification and month-run workflow

Use configured real model and embedding endpoints. Use a real Minecraft server only when the selected scenario needs it.

Run focused checks for changed contracts and deterministic utilities. Do not run full suites without an explicit owner request.

Use real note execution to verify note repairs. Verify changed fatal propagation without hiding the failure in a passing stub.

Do not build a broad specialized behavioral test suite before the month loop. Targeted implementation checks do not replace owner transcript review.

Read the current README section “Eventful month end-to-end run” before launching. Verify its command and fresh-run behavior against current code.

Use a new run ID, preserve source history, and record the actual configuration and revision. Keep old failure receipts.

At roughly six hours, present elapsed wall time and simulated time separately. Report public turns, unattended follow-ups, repeats, stale-memory replies, and technical leakage.

Include examples of voluntary quiet, alternative activities, and enjoyable exchanges. Counts are diagnostics, not fixed acceptance thresholds.

The owner decides whether to continue or stop. A successful 24-hour run requires acceptable behavior, not merely a live process.

After that milestone, propose specialized tests based on remaining failures. A full month remains the eventual goal.

When a prerequisite is missing, record `NOT RUN — missing prerequisite`, the command, and the missing service. Never substitute fake output.

## Risks and open choices

- Excessive inhibition can make Spamuel inert. Inspect initiative and enjoyable exchanges alongside annoying messages.
- A partner model can reinforce its own guesses. Preserve uncertainty and original evidence rather than repeated inferred confirmation.
- More reasoning can increase latency and cost. Reuse existing calls and projections; measure before adding another model pass.
- Existing seeds lack some modern metadata. Design explicit migration; never infer that missing metadata means external truth.
- Reviewer reconsideration can become a hidden gate. Specify ownership, bounded execution, and quiet semantics before wiring it.
- Fatal propagation spans worker threads and managed runs. Preserve receipts while ensuring the process cannot report healthy continuation.

Owner direction resolves quotas versus inhibition and the month-run strategy. Exact schema additions and review return semantics remain proposed until research fixes their scope.

## Copyable instruction for the implementing model

Read this plan and AGENTS.md. Start with stage A. Inspect current source; do not trust old line numbers or thoughts as evidence.

Work sequentially. Finish one stage with concrete receipts before starting the next. Do not change configured model providers or launch a month run during research.

Keep implementation records here or in a linked work record. Update conceptual docs only when implementation changes the current architecture.

At each handoff report: changed files, established causes, exact checks and outcomes, unresolved choices, evidence paths, and the next bounded step.

Never claim an implemented mechanism or successful run from this plan alone.


## Implementation receipt: 2026-11-16

Evidence: `.runtime/beta-immersion/20261116-repair/`. This is a repair pass, not acceptance of sustained immersion.
The fresh comparison is `eventful-month-immersion-20261116`. The stopped run and seed remain separate.

| Symptom | Established cause and repair | Evidence / limits |
| --- | --- | --- |
| Three identical publications | Three main-cognition actions selected the same move. Earlier delivered speech was available to later choices. Decision instructions now weigh delivery, self-image, and anticipated social cost. | `utterance-chains.json` traces all fourteen public utterances to action, source IDs, review, and publication. |
| Conditional speech sent immediately | Action 12400 said to hold the line until Jesse returned. Routing selected dialogue immediately. Routing instructions now preserve conditions. | Captured action and utterance 12407; direct regression of this exact conditional case remains unverified. |
| Old threads treated as current | Record 983 is historical companion dialogue. Seed records also contain several `(B)` threads. Guidance now carries epistemic roles, and historical public memory explicitly identifies quotations. | Exact originating prompt chains for both historical stimuli remain unresolved. No seed classification migration was needed. |
| Note failures became personal drama | Agent transcription changed whitespace and punctuation. Verification correctly rejected those bytes; the note wrapper converted failures into completed rejected reports. | Actual expected/actual hex retained in old reports. Host now writes exact bytes; real kernel reproduced all 251 bytes. |
| Internal defects continued as task events | The note wrapper caught internal errors before supervisor handling. | Wrapper removed. A real invalid-path invocation reached Application.on_error, set shutdown, and retained interrupted control state. Fatal errors no longer publish their initial task-error world event or consolidate artifacts. |
| Reviewer accepted repeated completed speech | Review lacked situational evidence and could display no conversation when the caller omitted a separate transcript. | Review now receives purpose, grounding, and dated actual turns. An uncached real review flags the captured third duplicate as repeated_move. It remains advisory. |

Exact new note boundaries: `NoteRequest` validates the existing inbox, `NoteSelection` validates model extraction, and `NoteResult` names the kernel result.
The note report retains `pm.note-verification.v1`, expected/actual hex, digest, original request identity, and verification status.
Agent session fields disappear because note copying no longer runs an agent. Existing public reply verification accepts the report without them.
No persistent cognitive schemas, action enums, message quotas, cooldowns, or reconsideration gate were added.

Checks:

- Real configured-model decision and route: four human/AI absence and return cases passed, with no completion-cache hits.
- Captured note request: actual kernel and filesystem preserved all 251 expected bytes.
- Real uncached model selection and note execution: Unicode content preserved exactly, including U+2014; 32 bytes inspected.
- Real managed supervisor fatal path: invalid internal path reached the application error handler and requested shutdown.
- Real uncached review: captured duplicate identified; earlier unsuccessful reviewer receipts remain in the evidence folder.
- Seventeen focused deterministic checks passed. One new check covers review's transcript projection. No full suite ran.
- Configured strict Pyright passed. Explicit note and situational-guidance checks also passed. Expanding strict checking to historically excluded supervisor/reviewer files exposes existing errors; those were not suppressed.
- Strict MkDocs build succeeded. A final rebuild follows the corrected README launch command. HTML inspection is programmatic, not visual.

Documentation changes cover note execution, situational decision evidence, and advisory review. Existing ownership and flow remain intact.
The README command now uses the template positional argument accepted by the current CLI.

Open evidence gaps: expectation 11652's full accepted revision history, exact historical-stimulus prompt chains, and recovery/cancellation under an actual service outage.
These are not established persistence defects. No checkpoint or expectation was rewritten from private thought claims.
The fresh month run is the next quality gate. At six hours, compare simulated and wall time, unattended follow-ups, repeats, historical replies, technical leakage, quiet choices, and enjoyable exchanges.
No beta card is marked complete by this receipt.


Additional trace receipts: `prompt-chains.json` links all fourteen actions to their decision, generation, and review prompt receipts.
The identical short line also occurs in private responses, so its generation candidates retain invocation and timestamp ambiguity rather than claiming uniqueness.
`historical-prompt-excerpts.json` retains the source passages supplied to the video and `(B)` decision/generation/review calls.
`expectation-chain.json` records expectation 11652 as pending, with zero fulfillment and no counted or resolving observations; no accepted successor was found.
A private thought about a failed update still does not establish an attempted accepted mutation.

Early fresh-run observation: the opening reply was selected but deferred because the same cycle changed its partner model.
The normal reconsideration path is running another full cycle. This is latency under inspection, not an example of voluntary quiet.
The dashboard is `http://127.0.0.1:8101/#run/eventful-month-immersion-20261116`.
Six hours after creation is approximately 20:03 Europe/Berlin on 2026-11-16. No automatic six-hour review is claimed.


Final documentation check: clean `mkdocs build --strict` exited 0 (`docs-clean.exit`), with current HTML and local links checked for index, managed work, publication, and situational guidance.
An attempted incremental strict build failed on its dirty-build warning; it was replaced by the successful clean build. No warning was suppressed.
The opening reconsideration released dialogue action 11327 at simulated 08:10:45. The public reply was still pending at 08:11:41.
This startup delay remains a concrete concern; process uptime does not establish acceptable interaction latency.
