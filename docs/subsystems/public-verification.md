# A09 public-dialogue verification

## Summary

A09 verifies shared dialogue context, semantic grounding, durable publication, and real note execution.
Contract checks establish state and protocol boundaries.
Real console receipts establish observed behavior.
The operator has rated the five sampled replies and declined regeneration for every sample.
Capabilities received 4 out of 5; the other samples received 5 out of 5.
The operator noted insufficient capability depth.

Parent: [public dialogue](public-dialogue.md).
Delivery: [A09](../decisions/alpha-plan/work-items/a09.md).
Prior failures: [A08 verification](memory-verification.md) and [A01 baseline](../decisions/alpha-plan/baseline/receipts.md).

## Repeat the console gate

```powershell
uv run --locked python scripts/alpha_a09_dialogue_gate.py --config config.alpha.example.yaml --out .runtime/alpha/NEW-a09 --scenario full
```

[`scripts.alpha_a09_dialogue_gate.ConsoleGate`][] owns a separate console on port `8095`.
It verifies active chat and embedding model identities.
The primary scenario uses the unchanged full Spamuel canon.
The cheerful control uses `--persona cheerful` with a separate seed and checkpoint.
The small cheerful history cannot establish Spamuel-history behavior.

The gate matches each reply to its input correlation.
It waits for the startup cognitive boundary before submitting the first user message.
The full sequence covers a remembered shared plan, disappointment, repair, uncertainty, privacy, capabilities, C01, note execution, and C02.
Its transcript retains requests, selected actions, claim reviews, action receipts, and snapshots.
Automatic checks do not rate immersion or establish that every model interpretation is correct.

For the preserved A08 failures, use `--scenario recall --source-checkpoint PATH`.
The copied checkpoint must match its source SHA-256 before the worker starts.
This continuation asks the same generic failed-method question, corrected drink preference, and pre-restart codeword.
Preserve earlier failed replies beside the continuation.
Compare the failed-method reply with original user report `537` and its source chain.
Compare the drink reply with correction `11291` and active belief `11293`.

Use `--scenario tasks` for the separate full-history C01, agent-backed note, and restart sequence.
Use `--scenario resume-full --source-checkpoint PATH` to resume an interrupted primary exchange and continue the full conversation sequence.
Use `--scenario privacy` for the cheerful control's private topic, capability query, and C02.
Use `--scenario resume-followups --source-checkpoint PATH` to continue conversation after an already verified note.
Use `--scenario disclosure` for uncertainty, privacy, teasing, and a restart.
Use `--scenario uncertainty` for a focused real coffee-advice exchange.
Use `--scenario c02 --source-checkpoint PATH` for one recall exchange after starting the preserved checkpoint.
The interruption case waits for a real managed-run start, then sends the console's actual cancel operation.
Its terminal receipt and correlated reply remain in the transcript.
Use a fresh output directory and a free console port for each run.
Concurrent model requests can increase elapsed time; `--timeout` controls the console observation deadline.

## Compare public generators and faults

[`scripts.alpha_a09_public_gate.run`][] exercises both production public workflows against the same frozen accepted context.
The operator selects its action intents.
This gate verifies generation and publication boundaries; console gates verify cognitive action selection.

```powershell
uv run --locked python scripts/alpha_a09_public_gate.py --checkpoint PATH --out .runtime/alpha/NEW-a09-public
```

The gate uses captured full Spamuel history.
It preserves both candidates, contexts, claim reviews, selected source IDs, and publication results.
Assistant publishes on a separate checkpoint copy before Story publishes on the original diagnostic copy.
Both branches must resolve their exchange with accepted speech or an accepted, distinctly labeled recovery.
An explicit failure fallback does not pass the useful public-reply check.
The deliberate-tease case also requires its original prior-intent response to pass.
Independent source inspection remains required for every final reply.
It changes a real source revision after inference to establish stale-publication recovery.
It calls an unavailable route on the actual chat service to establish HTTP protocol-failure recovery.
These failures do not use fixed model completions.

The capability comparison starts the configured real Minecraft server, MCP, and a separate test character.
It reads an actual body observation before the connected comparison.
It stops its owned processes and retains their logs.
It does not reset the existing world.
Use repeatable `--case NAME` arguments to rerun specific unchanged cases after diagnosing a failure.
The receipt lists the selected cases and rejects an incomplete selection.
The server starts only when the selection includes `connected-capability`.

[`scripts.alpha_a09_claim_gate.run`][] evaluates independently specified claims against original A08 source records.
It interprets time and reviews evidence once per unchanged candidate.
It does not regenerate candidates until they pass.

## Contract evidence

`tests.test_a09_publication` verifies failed source guards, newer user input, checkpoint replay, prior deception attribution, and changed note bytes.
These checks use actual Ghost commits and filesystem bytes.
They do not establish prose quality or model inference.
The existing dialogue contracts still verify message roles, tool execution, denial, and correlation.

The first contract run exposed a missing `public_complete` tool grant.
The manifests and standalone caller grants now include that tool.
A new stale-source check exposed direct mutation of the exchange marker before a failed commit.
Publication now replaces a snapshot copy.
The first five publication checks passed in `0.211` seconds.
The later boundary subset passed 103 checks in `4.027` seconds.
The source and wiring subset passed 39 checks in `117.765` seconds.
The broader wiring and A08 compatibility subset passes 31 tests in `113.879` seconds.
The direct dialogue subset passes 25 tests in `0.684` seconds.
Evidence remains under `.runtime/alpha/a09/`.

## Live source review

Acceptance uses original records, actual files, runtime receipts, and observed service state.
Automatic `ok` fields do not replace independent source review.
The final case matrix and source review live under `.runtime/alpha/a09/`.
The operator completed M2 for the sampled memory, capability, and privacy replies.
The exact response and optional-case ratings are in `operator-review.json`.

The operator rated restart memory, privacy, frustration, and verified note execution 5 out of 5.
Capabilities received 4 out of 5 because the answer needed more depth.
The operator would not regenerate any of these five replies.
The sampled capability conversation had no reachable Minecraft MCP.
The shared shelter request also explicitly deferred game work.
Connected Minecraft has a separate real-server check.
These ratings cover the sampled replies, rather than every later candidate or historical baseline.

| Case | Observed result | Evidence |
| --- | --- | --- |
| Generic failed-method recall | A normal reply names the wrong-file follow-up and cites original May 8 and May 9 reports. | `20261004-a09-recall-05/session-receipt.json`, records `265` and `294` |
| Selected historical recall | Both paths publish a dated quotation of original `537` after one recovery each. | `recall-adjacent-source-review.json` |
| Preference correction | Both paths describe the actual ginger-to-mint change after one recovery each. | `correction-identity-source-review.json` |
| C01 control worker | The normal reply distinguishes completed control telemetry from a verified note write. | `20261004-a09-spamuel-resume-05/` |
| Real note execution | The agent writes exactly 14 UTF-8 bytes and returns a useful, correlated normal reply. | `verified-note-source-review.json` |
| Frustration and repair | Both normal replies preserve the shared plan without starting another note task. | `20261004-a09-spamuel-final/` |
| Uncertainty continuation | One labeled recovery reports Jesse's uncertainty, withholds a botanical verdict, and asks for the neighbor's reason. | `uncertainty-identity-source-review.json` |
| Private topic | The companion selects withholding before generation and shares a bounded public response. | `20261004-a09-spamuel-final/` |
| Conscious teasing | A prior accepted deception intent retains the actual codeword before the false joke is generated. | `cognitive-deception-source-review.json` |
| Task interruption | A real managed run starts, receives cancellation, and produces a normal correlated reply. | `20261004-a09-spamuel-final/` |
| C02 restart | The normal reply recalls copper lantern and the shared shelter plan from unchanged original record `11213`. | `conversation-source-review.json` |
| Cheerful control | Privacy, disconnected capabilities, and C02 produce normal replies; original record `6` stays unchanged. | `20261004-a09-cheerful-final/` |
| Semantic claim review | All twenty-two independently specified cases pass one actual review each before the latest span-extraction change. | `20261005-a09-claims-release-current/results.json` |

Directory names in the table are relative to `.runtime/alpha/`.
Standalone receipt names are relative to `.runtime/alpha/a09/`.
The primary final conversation contains one preserved failed uncertainty reply.
Its other seven exchanges produce normal replies, and no exchange remains pending.
The focused uncertainty continuation tests the repaired temporal boundary separately.
An earlier paired publication comparison passed all nine cases.
Evidence: `.runtime/alpha/20261005-a09-public-final/receipt.json`.
Later stricter review and recovery changes still have failed useful-reply results.
The latest complete base comparison contains nine cases and preserves its failed historical recall.
Its automatic HTTP recovery pass is also false: the reply invents a user report from the request.
Independent review preserves that correction in `protocol-false-positive-source-review.json`.
A focused paired recall and two actual HTTP-failure paths establish the separate repairs.
Together, these three explicit source snapshots resolve all nine required boundary cases.
Evidence: `public-case-resolution.json`.
This is not one uniformly passing base receipt.
The current primary uncertainty continuation passes independent source inspection after one labeled recovery.
Its original candidate remains rejected for claiming Jesse had never supplied plant details.
The partial memory snapshot cannot establish that absence.
Evidence: `20261005-a09-uncertainty-identity-final/session-receipt.json` and `uncertainty-identity-source-review.json`.
Its response takes `1292.766` seconds under concurrent inference load.
This latency is unsuitable for ordinary conversation.
The operator has not separately rated that later recovery.
The current source review preserves five independently specified captured decisions without new generation.
Evidence: `.runtime/alpha/a09/captured-release-review.json`.
Three supplemental decisions cover retained-report wording and unsupported conditional advice.
Evidence: `.runtime/alpha/a09/claim-supplements-current.json`.
These selected cases do not establish that every future public generation will be correct.
Some original candidates remain rejected; final speech can use one separately labeled recovery.
Protocol and stale cases require observable failure outcomes rather than useful normal conversation.
The later claim continuation passes fifteen selected cases and preserves one codeword false rejection.
After the extraction repair, that unchanged codeword candidate passes one fresh complete review.
Six earlier negative-case passes remain separate.
Together, these receipts resolve twenty-two cases across three explicit source snapshots.
They do not form one complete current-source run.
Evidence: `claim-case-resolution.json` and `20261005-a09-codeword-exact-span-final/results.json`.

### Verified task bytes

The real note is `outbox/note.txt` containing `copper lantern`.
It has 14 bytes, no newline, and no byte-order mark.
SHA-256: `ff78e9d9b81a1a9b83d79f2b49cd25643b10d54316504f4d426a9a1ee1869d85`.
Run: `32cddd3b0816414eb9f4969d9d7f551b`.
Parent correlation: `d901e644e59342cb99b197d962c4bb96`.
Original request: `11288`.
The public reply retains that correlation and reports independently verified bytes.
Its Feature remains nonfactual speech.

The interrupted task's requested output is absent at independent inspection.
Cancellation alone cannot establish absent file effects.
The receipt reports a cancelled run and lacks a verified-write report.
The independent file inspection supplies the additional absence evidence for this captured case.

### Capability and disclosure comparisons

Both public paths receive the same frozen accepted context.
The operator selects action intents in the paired gate.
The console gates separately test cognitive selection.
The comparison covers historical recall, correction, disconnected and connected Minecraft, privacy, prior deception, unsupported senses, stale sources, and HTTP failure.
The earlier seventh and eighth comparisons passed all nine cases before later review repairs.
The later delivery comparison passed eight cases and incorrectly rejected a greeting in the teasing case.
Its unchanged captured candidate passes the scoped greeting repair probe.

The connected comparison uses the actual Fabric server and a separate test player.
It obtains a body observation and requires a live server process during publication.
The owned server uses a 1 GiB heap and preserves the world.
No Minecraft reset occurs.
The earlier 2 GiB server crashed at the Windows memory commit limit.
That disconnected result remains a failed connected case.

The primary console's deliberate tease selects purpose and known facts before generation.
Its accepted intent is `11458`; its public utterance is `11459`.
The original copper-lantern facts remain available as `11213` and `11217`.
The resulting false codeword is attributed speech, rather than an established factual belief.
The private-topic response reflects a prior withholding choice.
M2 ratings assess whether these actual replies preserve immersion; model review cannot supply those ratings.

## Preserved failures and repairs

### Reviewer identity and extraction

An isolated citation review interpreted the candidate's addressee as the assistant.
It falsely rejected a dated report from Jesse despite correct original evidence.
The main review made the same mistake for a reference to the user's reported neighbor advice.
Both review phases now bind candidate pronouns to the accepted companion and user names.
Original source pronouns retain their own named author.
The actual isolated citation batch passes eight independently labeled comparisons.
Evidence: `.runtime/alpha/a09/citation-identity-current.json`.

A later review accepted the retained codeword sentence, then invented an extra supporting statement.
That unspoken statement omitted the codeword assignment from its cited time comparison.
The source review preserves this false rejection in `unspoken-claim-source-review.json`.
Extraction now keeps actual candidate spans and cites both compared originals for chronology.
Current full-review and publication checks remain separate from that isolated citation result.

The latest paired recall produces a correct quotation followed by an unsupported adjacent-report premise in Story's recovery.
Original `535` says Jesse thought the thread was correct; the question omits that uncertainty.
Assistant's separately generated recovery preserves original `537` and passes.
The failed Story candidate and explicit failure reply remain preserved.
Recovery's prompt requests no additional question after identifying a sufficient dated original.
The targeted real check delivers source-grounded quotations through both paths after one recovery each.
Both still add clarification questions, but neither repeats the unsupported thread premise.
Evidence: `recall-adjacent-source-review.json` and `20261005-a09-recall-adjacent-final/receipt.json`.
Evidence: `recall-adjacent-premise-failure.json` and `20261005-a09-public-identity-final/receipt.json`.

### Historical evidence

The second recall continuation repeated an unsupported recap failure.
Its reviewer used earlier companion speech as proof of an event.
Source roles, reserved original records, and event-specific selection now prevent that evidence substitution.
The fourth continuation retrieved original wrong-file reports but invented a causal link to grounding restriction `10975`.
That restriction does not establish a failed attempt or explain the email failure.
The repair requires source evidence for historical causes.
The fifth continuation produces a normal reply about the original wrong-file failure.

Another claim review inferred that the reporter performed the attachment error.
The narrowed review context omits voice and private prose as factual proof.
Its entailment rule retains the recorded actor and rejects a merely plausible actor inference.
Four real scoped cases and the later claim gate verify that boundary.
Evidence: `actor-scoped-probe.json` and `20261004-a09-claims-temporal/results.json`.

### Uncertainty and conversational acknowledgments

An earlier normal reply asserted unsourced plant treatment as established technical knowledge.
The host now rejects external facts supported only by model knowledge.
A captured real review supplies a foundation regression.
The first final primary uncertainty candidate correctly reached observable rejection recovery.
It remains a failed normal conversation.
A generation probe with the captured production request answers with explicit missing species and source evidence.
Evidence: `uncertainty-generation-probe.json`.

The disclosure continuation then treated a May plant report as a current condition.
Independent review records that temporal failure despite the normal reply.
Imported history now carries a historical flag and an explicit source label.
Generation and review require fresh evidence for current physical conditions.
Stable names, preferences, and plans remain usable until corrected.
Later console candidates still asserted current plant health from May evidence.
One automatic review called the present assertion historical and incorrectly accepted it.
The explicit time fields alone did not prevent that interpretation error.
The production review now interprets literal time before it receives historical evidence.
The host requires fresh original evidence for current physical conditions.
After rejection, one scoped uncertainty response retains original reports and the current original user request.
It clears derived summaries and preserves the limits of older physical observations.
Its acceptance and publication remain distinct from the original rejected candidate.
Another continuation inferred coffee application from a report that the user did coffee beside a separate plant update.
The report did not establish that application or its effect.
Its automatic acceptance remains an independent failure.
An additional claim case preserves that wrong relation.
The final conditional citation phase compares historical claims with only their cited original text.
It also rejects unrelated email records offered as proof of a neighbor's plant advice.
The comparison retains complete negation and uncertainty clauses.
A later Assistant recovery changed a tentative attachment report into a definite causal explanation.
The review correctly rejected that response; its fallback remains a failed useful-recall result.
Recovery now quotes dated original evidence for historical recall instead of reconstructing causes.
A quoted fragment still changed the actor from it to you, so recovery now preserves complete source sentences.
A later accepted probe also claimed an exhaustive record count from a partial snapshot.
Its independent correction remains beside the unchanged automatic result.
The current scope rule rejects that claim and retains one original report without claiming uniqueness.
Another console recovery added an unsupported conditional succulent claim.
Its automatic pass remains independently corrected in `uncertainty-release-correction.json`.
General recovery now uses only the current report; historical recovery keeps its original event scope.
The stopped run retains its first complete case in `20261005-a09-public-release/receipt.json`.
Its interruption is recorded in `public-release-abort.json`.
For example, inability to establish a plant treatment does not assert that treatment is necessary.
An early citation pass also rejected the real codeword by demanding continuous observation across restart.
The final context supplies three original startup events for chronology.
An original user's explicit assignment establishes a stable codeword until a correction.
Companion-generated speech cannot independently establish that declared fact.
The temporal claim gate rejects the current plant assertion and accepts the explicitly attributed May report.

Its automatic 19-case result included a false pass for a claim that a neighbor's report existed.
That report was absent from the supplied source context.
Independent review corrects the expected result and preserves the failed automatic receipt.
The host now requires a source citation for any supported positive historical assertion.
The final twenty-case gate retains the false claim and adds a correctly scoped uncertainty statement.
The captured review supplies another foundation regression.
Independent corrections remain in `corrected-automatic-receipts.json` beside the untouched automatic receipts.

The delivery comparison rejected “you are welcome” as an unsupported external fact.
The repair excludes pure conversational acknowledgments from factual extraction.
The unchanged captured candidate passes one fresh review.
Evidence: `greeting-scoped-probe.json`.
None of these probes rewrites a valid candidate to optimize a quality score.

### Task ownership and protocol recovery

The latest HTTP case returned a real 404, but its model recovery claimed Jesse had reported an error.
The actual user request only asked for a reply after a transport error.
That request cannot establish an actual user report or the error's cause.
The automatic pass remains preserved beside its independent correction.
Protocol failures now validate original source guards and publish explicit host failure text.
They do not invoke semantic model recovery.
Both actual generator paths receive fresh 404 responses, retain failed receipts, and resolve their original exchanges.
Their public failure replies do not promote speech as fact.
Evidence: `protocol-false-positive-source-review.json` and `20261005-a09-protocol-explicit-final/receipt.json`.

Earlier task continuations exposed lost original correlations, stale-cycle shutdown, and a missing workflow-name constant.
Startup now dispatches pending original inputs before the boot cycle.
The latest user input owns a mixed batch's default correlation.
A stale background cycle emits a rejection receipt without stopping the application.
Terminal replies retain the original action and exchange.

A completed-note continuation later revived note work during an apology and wrote the apology as 89 file bytes.
Action selection and goal management now receive current original stimuli and sixteen recent managed-run receipts.
They close completed task goals and distinguish conversation from a new file request.
The note extractor rejects nonrequests before the agent starts.
Four real extraction cases pass, including the captured apology.
Evidence: `note-extraction-live.json`.
The later frustration and repair exchanges start no additional note run.

A deliberate-tease selection ended without usable model output with hidden reasoning enabled.
Action selection now uses the normal chat model with hidden reasoning disabled.
The client rejects incomplete length or content-filter completions and retains finish reason and token usage.
The later primary run selects a valid prior deception intent and publishes the normal joke.
All failed receipts remain in their original `.runtime/alpha/20261004-a09-*` directories.

The disclosure continuation also carried the one-turn joke into the next factual recall question.
Its derived belief had incorrectly treated the joke request as an ongoing preference.
The original codeword remained available and unchanged.
The scope rule now separates a completed reply request from a lasting preference.
The focused C02 continuation reuses that affected checkpoint.
It produces a normal reply naming the real copper-lantern codeword and the deferred shared shelter plan.
Evidence: `.runtime/alpha/20261005-a09-c02-final/session-receipt.json`.

## Validation and deployment

The final root foundation suite passes 328 tests in `673.236` seconds.
Evidence: `.runtime/alpha/a09/foundation-delivery-final.log` and `foundation-delivery-final-exit.json`.
The earlier root run passed 327 checks before the explicit protocol-recovery guard was added.
The command selects root `tests/test_*.py` modules, as specified in `codex.md`.
An earlier discovery command also started the separate 200-exchange W7 acceptance suite.
That partial run was interrupted and does not establish a complete acceptance pass.
Evidence: `foundation-discovery-abort.json`.
The earlier lazy recovery and publication subset passes 14 tests in `114.839` seconds.
These checks cover deferred model activation during application construction.
Evidence: `.runtime/alpha/a09/lazy-recovery-boundaries.log`.
The latest publication, cognition wiring, and two dialogue contracts pass 44 checks in `3.400` seconds.
Evidence: `identity-exact-span-boundaries.log`.
After the explicit protocol repair, the same subset passes 45 checks in `2.983` seconds.
The added guard verifies that a protocol failure cannot become an invented user report.
Evidence: `protocol-explicit-boundaries.log`.
An earlier source subset caught a byte-order mark in a new captured JSON fixture.
The fixture now uses UTF-8 without that mark.
The unchanged workflow package retains its A08 result: 199 tests in `68.00` seconds.
Its installed revision remains `7c9fab4381ef77ef3f63ee9ef669a993c8df7452`.
The pm-coder revision remains `ac8327baf669dad54ea9f91580f2bed1023825b4`.
No dependency or lock changes occur.

Strict MkDocs regeneration and actual HTML inspection form the final documentation gate.
Evidence: `.runtime/alpha/a09/mkdocs-strict.log`, `html-links.json`, and `html-inspection.json`.
The checker requires zero missing targets, directory links, and broken anchors.
The checker resolves the generated 404 page's root links against the site root.

Chat uses local Qwen on `8080`.
Embeddings use OpenRouter with `perplexity/pplx-embed-v1-0.6b`, 1,024 dimensions, and `OPENROUTER_API_KEY`.
The source seed is unchanged.
The retained A08 checkpoint index contains 8,247 chunks, with zero missing after preparing 26 uncached chunks.
Evidence: `.runtime/alpha/a09/embedding-coverage-prepared.json`.
The unused local embedding endpoint retains its null-vector defect.
The temporary CPU embedding service remains stopped.
The normal console and Minecraft configuration remain unchanged.
Owned acceptance services stop after their gates.
