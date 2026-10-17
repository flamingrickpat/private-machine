# A18C wait and control verification

## Summary

The owner accepted A18C for alpha on 17 October 2026 after review of this record.
Implementation and focused real-service checks are complete.
Final-source validation passed 896 unit tests and 83 contract tests.
The complete A18B evaluation remains unverified. Detailed human quality ratings remain unassigned.

Parent: [A18C](../alpha-plan/work-items/a18c.md).
Implementation: [inference waits](../../subsystems/inference-wait.md).
Remaining design: [deliberate control](../alpha-plan/deliberate-control.md).
Historical failure: [A18B stall](a18b-errors/stall.md).

## Source and preserved work

The resumed base is `f7092a7b2a216131d6f394869d74fd08379da387`.
Separate commits retain the earlier A18B singleton repair and the A18C implementation.
No dependency changes are required by the wait path.
Installed origins remain in `.runtime/a18c/origins-before.json`.
The original implementation handoff remains `.runtime/a18c/HANDOFF.md`.

Earlier A18B failure receipts are unchanged.
The resumed work does not claim that silence was a persona error.
It also does not establish a complete 16-case evaluation pass or A19 readiness.

## Owner acceptance

The owner reviewed this verification record and judged the implementation sufficient for the current alpha.
This acceptance closes A18C delivery. It does not supply separate recall, voice, usefulness, or immersion ratings.
Historical pending-review statements below describe the status at each evidence capture.
The complete A18B repertoire and its retained failures remain separate from this acceptance.
The [future direction](../alpha-plan/deliberate-control.md#future-direction) records the owner's possible agent and MCP design.

## Focused checks

```powershell
uv run --locked python -m unittest tests.unit.workflows.cognitive_control.test_wait -b
uv run --locked python -m unittest tests.unit.pm.memory.test_lineage -b
```

`control-lineage-final2.log` records 27 passing checks in 2.746 seconds: 22 wait checks and five lineage checks.
Earlier focused passes remain in `wait-unit-05.log` and `lineage-unit-01.log`.
These paths are under `.runtime/a18c/`.

The wait checks cover failed checkpoints, stale Sleep decisions, pending-input races, duration validation, restart, and timeout replay.
They also cover late private commits, old queued thoughts, zero work during waits, and duplicate timer delivery.
Real assembly construction verifies that paused user sources do not redispatch after restart.
It also verifies that timer records cannot enter public speech recovery.
The lineage checks cover checkpoint preservation, missing references, hidden source exclusion, and immutable original history.

Episode checks in `tests/unit/workflows/inner_thought/test_episode.py` use the real store and wait controller.
They cover durable episode identity and sequence, lineage, the acknowledgment gate, generation interruption, the step limit, stalled progress with parking, parked-source trigger suppression, and checkpoint reopen.
`episode-focused.log` records 70 passing inner-thought checks in 1.748 seconds.
Pipeline checks cover an unknown continuation, a missing or unknown next operation, an unknown progress kind, and a downgraded rephrased-progress claim.

Focus checks in `tests/unit/workflows/cognitive_control/test_focus.py` cover the current-cap clamp, expiry, the separate rejection receipt, accepted advice, and urgent reduction protection.
Mental-work checks in `tests/unit/workflows/focus_management/test_mental_work.py` cover the report write, host commit, stale-source rejection, and the inspected-set boundary.
They use a fake agent driver, so they are boundary tests, not real recall quality.
`focus-mental-focused.log` records 10 passing checks in 0.835 seconds.
Those boundary tests also found a real defect: the mental role passed a mapping to `write_report`, which requires a typed model. The role now writes the mapping directly.

The first focused attempt failed because test tool grants omitted the newly declared retrieval tool.
It also exposed a local `json` import that shadowed the module import.
Those failed checks remain in `resume-focused-initial.log`.
The next attempt exposed three missing required `source` arguments in the new tests.
That failure remains in `resume-focused-02.log`.
Subsequent targeted passes do not erase those receipts.

## Foundation and real gate

The first resumed unit suite passed 854 checks in 639.356 seconds under `unit-resume-01.log`.
The first resumed contract suite passed 80 checks in 4.870 seconds under `contract-resume-01.log`.
Those results precede the last restart-race checks.
The next unit repeat passed 863 checks in 694.512 seconds under `unit-resume-final.log`.
After the timer-shutdown repair, the final unit repeat passed 864 checks in 670.911 seconds under `unit-resume-final2.log`.
The final contract repeat passed 80 checks in 5.265 seconds under `contract-resume-final2.log`.
The episode and mental-work source then changed the suite.
The reopened unit repeat passed 888 checks in 525.760 seconds under `unit-episodes-final.log`.
The reopened contract repeat passed 82 checks in 4.852 seconds under `contract-episodes-final.log`.
Both final exits are 0. No native-exit anomaly occurred in these completed runs.
Compilation alone does not replace either suite.

The focused live command is:

```powershell
uv run --locked python dev/work_items/alpha_a18c_wait_gate.py --config config.alpha.openrouter.yaml --out .runtime/a18c/live-wait-01 --deadline 1800
```

This uses the real concurrent assembly and full Spamuel canon.
It starts with ordinary conversation before testing deliberate quiet.
It does not inject fixed action-selection output.
Its timeout configuration is 45 seconds by default, capped at 60 seconds for this diagnostic.
The production default remains 1,800 seconds.

The first live attempt exposed a gate defect: unsolicited startup speech satisfied its initial reply counter.
Its sequence therefore cannot establish ordinary back-and-forth acceptance, even if its final receipt reports success.
The corrected gate waits for the submitted correlation and actual public-delivery acknowledgment.
It also measures routine stage entry independently of already authorized speech.

The first attempt still supplies narrow observed facts.
It accepted wait record `11325` for 45 seconds at `00:19:22 UTC` on 17 October 2026.
During its five-second observation, chat requests stayed at 65 and accepted cycle count stayed at 815.
Outside input ended that wait at `00:19:41 UTC`.
The first run later completed its mechanical checks in 1,576.5 seconds and returned `passed: true`.
Its restart preserved private record IDs and the accepted wait.
Timeout record `11387` retained `elapsed_seconds: 45.448311` and reached `wake_status: consumed` through main cognition.
The initial cadence defect still prevents full-gate or conversation acceptance from that run.
Its source also precedes the later restart-race and timer-shutdown repairs.

The corrected fresh run is `.runtime/a18c/live-wait-02/` with outer log `live-wait-02.log`.
It uses its own writable embedding cache and preserves `source.zip` with tracked and untracked sources.
`source-files.json` records file hashes.
Source and effective configuration hashes remain separate.
It retains `facts.json`, `bus.jsonl`, replies, kernel artifacts, and `ghost.json`.
Its terminal receipt passed in 1,641.719 seconds with exit code 0.
This establishes the focused wait sequence, not complete A18C or companion quality.
Its runtime snapshot precedes the final timer-shutdown helper extraction.
That boundary has separate final-source tests.

The corrected run accepted wait `11346` for 45 seconds.
Across its five-second quiet observation, routine stage entry stayed at 59, chat requests stayed at 84, and cycle count stayed at 816.
New source `11347` reached main cognition and received its correlated public reply.
Restart preserved wait `11414` and the private record IDs without redispatching its paused sources.
System wake `11417` retained elapsed seconds `45.294732` and committed `wake_status: consumed` through main appraisal.
Prompt and provider logs are copied into each run's `llm-logs/` folder from the logged request IDs.
An active-run capture can contain pending requests.
After termination, repeat the collection.
Human immersion review remains pending regardless of the mechanical gate result.

The first strict HTML build passed in 108.55 seconds under `docs-resume-01.log`.
Chromium opened eight local pages and followed five generated API links without broken HTML targets.
Receipts and the inspected screenshot remain in `.runtime/a18c/html/`.
Final regeneration uses `docs-resume-final.log`.
The final browser inspection uses `html-inspection-final.log` and refreshes `html/receipt.json`.

## Remaining acceptance

Real recall, focus influence, and parked concerns now have full-canon evidence below.
Urgent competition, focus expiry, and accepted-pause observation passed separate fresh gates below.
The final continued-episode gate also passed below.
Ordinary private-thought goal changes now remain suggestions for main appraisal.
Beta clustering, fragmentation metrics, and dreaming remain out of scope.
Strict HTML regeneration and local link inspection must accompany the final source state.

This session changed sources after the last live wait gate snapshot.
The wait gate receipts remain valid for the wait path only; they do not cover the new episode, focus, or mental-work code.

## Third resumed session

The first real mental-agent run is `live-mental-03/` under `.runtime/a18c/`.
It completed 13 real agent requests and revised failed recall queries to inspect original playlist and mood-word conversations.
Its host boundary then failed with `KeyError: 'mental_inspected'` because the manifest omitted that tool grant.
The failed receipt and completed agent trace remain visible.
The repaired boundary tests now use the manifest's actual tool grants.

`live-mental-timeout-01/receipt.json` passed in 54.703 seconds with all 11,202 seed records.
A real one-second agent deadline produced `TimeoutError` and a failed action result with no plan.
Checkpoint reopen queued that result for appraisal without another agent session.
`live-mental-provider-01/receipt.json` passed in 59.875 seconds with the same full seed.
It used an explicitly invalid test credential against the actual provider.
The resulting `PermanentProviderError` retained a failed result and passed the same restart check.
These narrow checks establish failure handling, not recall quality or a full conversation.

The normal rerun is `live-mental-04/`.
Its real agent completed 11 requests and committed proposal `11204` from inspected original conversations.
It used revised queries and retained uncertainty about the old history.
Its terminal receipt passed in 931.968 seconds.
Main appraisal accepted the proposed ledger goal and a focus strength of 0.35 for 900 seconds.
Seven real nominations received the 0.35 bias. The direct stimulus retained its reserved attention place.
The private loop then chose `rumination_notice` and suspended its concern.
Its next request retained parked sources `1264`, `11206`, and `11225`.
This run establishes actual recall execution, focus influence, and parking. It does not establish a continued episode or human recall quality.

Main context initially rejected repeated malformed ID lists with an ellipsis.
The rejected answers and `ModelFormatError` remain in the run.
The context prompt now requires complete, unique integer IDs without abbreviations.
The later appraisal completed after that prompt repair, so the source snapshot and final prompt differ.
Fresh follow-up gates retain the repaired prompt and later scheduling source.

The A18B driver now records runtime protocol `a18c.pause.v1`.
A committed wait for the current pending exchange ends generation with `accepted_pause`.
No further actor message or invented public reply follows that outcome.
Unfinished scenes retain incomplete coverage and earn no aggregate acceptance.
Idle cycles, expired waits, and another exchange's wait do not qualify.

The host also repairs closed-episode parking, acknowledged continuation without another trigger, and generation preservation across restart.
`episodes-resume-03.log` records 73 passing private-loop checks in 2.829 seconds.
`control-resume-03.log` records 45 passing control, mental-work, and harness checks in 5.195 seconds.
The first foundation attempt was intentionally stopped after these scheduling repairs superseded its source.
Its interruption is recorded in `unit-resume-03-interrupted.json`; it is not a native-exit anomaly or a pass.
Final-source foundation results use `unit-resume-03-final.log` and `contract-resume-03-final.log`.

The next foundation repeat passed 894 unit checks in 678.526 seconds and 82 contract checks in 5.303 seconds.
After the next boundary additions, `unit-resume-03-delivery.log` passed 896 checks in 678.145 seconds.
`contract-resume-03-delivery.log` passed 83 checks in 4.823 seconds.
The final private-publication replay source uses another repeat in `unit-resume-03-source-final.log` and `contract-resume-03-source-final.log`.

`live-pause-03/receipt.json` passed in 846.078 seconds.
It received an ordinary correlated public reply, then accepted wait `11345` for pending source `11293`.
During the quiet sample, model requests stayed at 81, routine stage entries at 58, and accepted cycles at 814.
The gate sent no input after the pause request and retained the pending source.

`live-focus-05/receipt.json` passed in 651 seconds from an isolated copy of the completed recall checkpoint.
The old focus stayed active while a new urgent alarm report reached attention.
Real scoring assigned the direct stimulus urgency 1.0, selected it, and set the danger flag.
The same run observed actual expiry of the 900-second focus on the real clock.
It used no synthetic clock advancement and did not rerun the recall agent.

`resume-checks-03.json` also replays the captured real attention scores without bias.
The selected set changes from `n1/n3/n5/n12/n13` to `n1/n12/n13/n15/n17`.
This comparison establishes the bias's effect on those scores, not a human quality judgment.
The direct stimulus keeps its reserved place in both results.

The first private-comparison attempts rejected `next_operation: exploration`.
The thought prompt requested a catalogue operation but omitted the catalogue itself.
The role now receives the complete invocation catalogue.
`operation-catalogue-contract.log` records six passing role checks, and the rejected real outputs remain visible.
The fresh `live-episodes-06` gate includes that repair and the final private-publication replay path.

The second real fault runs use actual `ActionSchemaKnoxel` records and `arguments.instruction`.
`live-mental-timeout-02` passed in 58.141 seconds; `live-mental-provider-02` passed in 55.047 seconds.
Both reopen the checkpoint and submit the original action again.
Neither startup recovery nor duplicate delivery repeats the completed failed session.
The private result remains available for main appraisal with no accepted plan.

The earlier A18B singleton files and the Spamuel canon remain unchanged from the supplied handoff.
The recovery audit records their hashes separately from this session's changes.

The final-source unit suite passed 896 checks in 634.738 seconds under `unit-resume-03-source-final.log`.
The final-source contract suite passed 83 checks in 4.572 seconds under `contract-resume-03-source-final.log`.
Both exits are 0, with no native-exit anomaly.

`live-episodes-05/receipt.json` passed in 1,234.516 seconds after the catalogue repair.
It developed three acknowledged steps under episode `b3a8a3b5b5ec42ecbeb1c6d65a871946`, then suspended and parked a later concern.
Its source changed during the repair, so final-source acceptance uses the next fresh run.

`live-episodes-06/receipt.json` passed in 866.859 seconds with the final runtime snapshot.
Thoughts `11296`, `11317`, and `11333` retain episode `39136d9ee45a40b68db8c08d4aa88704` with sequences 0, 1, and 2.
They developed a narrative, checked feasibility, and proposed concrete alternatives through main appraisal.
The host prevented advance before each acknowledgment.
The third step still requested `continue`, but the configured three-step limit closed the episode explicitly.
Private steps changed no accepted goals before main appraisal.
This establishes bounded execution with real content. Human review of the alternatives' usefulness remains pending.

Strict HTML regeneration and local inspection use `docs-resume-03-final.log` and `html-inspection-resume-03-final.log`.
The browser follows the changed guides and their generated API links, including both private replay methods.
