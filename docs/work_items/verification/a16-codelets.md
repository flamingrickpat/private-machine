# A16 verification

## Summary

A16 evidence has three parts: foundation tests for the new pure rules, contract tests for the new stage boundaries, and one live gate.
The live gate uses the real Spamuel seed, a fresh checkpoint, the real application path, and real inference.
Model: `qwen/qwen3.8-flash` through OpenRouter, at provider precision.
The target Q3 build on the operator's home computer is not verified.
The gate asserts mechanical contracts. Emotional quality stays a human review question, and the scenario facts are recorded for that review.

Parent: [codelets and attention competition](../../subsystems/codelets.md).
Companion page: [codelet pathways](../../subsystems/codelet-pathways.md).
Work item: [A16](../alpha-plan/work-items/a16.md).
Gate: `dev/work_items/alpha_a16_codelet_gate.py` (removed in A19F step 5 with the catalogue it tested; see git history).

## What A16 delivered

- The catalogue grew from 11 codelets (`a15.3`) to 38 codelets (`a16.1`) across the ten required families.
- Eleven versioned pathways connect accepted precursor proposals to later activation. See [codelet pathways](../../subsystems/codelet-pathways.md).
- Every definition declares its context requirements and its expected consequence.
- Every proposal kind has a named consumer with a durable effect. See the [consumer map](../../subsystems/codelets.md#output-and-percepts).
- Accepted `belief` proposals stage revisable subjective `BeliefRecord`s with dedupe, supersession, and evidence links.
- The main LLM budget rose from 3 to 4 calls, and attention capacity from 4 to 5 places, for the larger repertoire.

## Foundation and contract tests

| File | What it shows |
| --- | --- |
| `tests/unit/workflows/main_cognition/test_pathways.py` | Precursor windows, decay, the activation cap, unselected and foreign records, tick age from the latest tick, duplicate pathway names, and full catalogue wiring. |
| `tests/unit/workflows/main_cognition/test_belief_proposals.py` | Belief creation with evidence links, retain without stacking, supersede with kept history, the observed-reality rejection, unconsumed nominations, and visible missing evidence. |
| `tests/unit/workflows/main_cognition/test_codelet_schedule.py` | Derived inverse signals (unfairness, exclusion, uncertainty), `emotion.negative_strongest`, relationship signals, kind/origin/reality validation, the shipped declarations, and a pathway boost trace. |
| `tests/unit/workflows/main_cognition/test_cognition_state.py` | A real cycle commits an accepted belief with its consumers and topic; a deferred belief never commits. |
| `tests/unit/workflows/main_cognition/test_pipeline.py` | The per-owner proposal filters and the executable consumer map. |
| `tests/contract/main_cognition/stages/test_codelets_contract.py` | Only belief proposals carry a topic. Precursor records raise pathway signals that appear in scheduling causes. The LLM bound follows the catalogue policy. |
| `tests/contract/main_cognition/stages/test_goals_contract.py` | Accepted inner proposals render as motive evidence, and their absence renders as none. |
| `tests/contract/main_cognition/stages/test_action_selection_contract.py` | Accepted regulation and simulation proposals render with their reality labels. |

These tests fix contracts. They do not show model quality. The live gate records the scenario facts for that review.

## Live gate

The gate runs seven turns on one fresh checkpoint: disappointment, distraction, anger, jealousy, boredom, mixed motives, and a reopen of the same checkpoint.
A variant catalogue proves that a changed configuration changes the selected frame.
A separate cheerful persona runs two turns as a boundary check. Its different, small history cannot establish a trait-only effect.

### Run

The run used this command and finished with `ok: true` and no failures:

```powershell
uv run --locked python dev/work_items/alpha_a16_codelet_gate.py --config config.alpha.openrouter.yaml --out .runtime/alpha/20261015-015205-a16
```

The evidence lives in `.runtime/alpha/20261015-015205-a16/`: `receipt.json`, `primary.json`, `variant.json`, `cheerful.json`, and one `turn-*.json` per turn.

Provenance: the first live run finished the primary and cheerful legs and saved their evidence, but ran the variant leg on the default catalogue because of the wiring defect below. The variant leg was re-run against the real model after the repair, and `receipt.json` was assembled from the three saved legs. No leg used a fixed model answer.

### Mechanical results

- Every turn committed a frame whose winners match the selected records, and every reply was companion speech. No reply leaked codelet machinery.
- No pathway signal appeared without an accepted precursor. The letdown turn raised `affection` at 1.0. The anger turn carried `affection` 0.96, `shame` 0.6, `persistence` 1.0, and `anger` 1.0.
- After the reopen, the fresh pathways `anger`, `persistence`, and `shame` raised their signals again, from ticks 809 to 814 against the reopen tick 816.
- The reopen kept the attention records of every prior tick, the same catalogue digest, and its subjective beliefs.
- The variant catalogue changed the digest from `d7451347c8248109` to `08d68121955e7883`. The removed `refusal_edge` stayed absent, and `curiosity_probe` ran.
- The cheerful persona produced replies on both turns. `absence_return` stayed skipped: its small history gives it nothing to return to.

### Scenario facts for human review

- The replies show the intended causal continuity. The letdown turn apologizes under disappointment. The distraction turn keeps the concern alive beside the new topic. The anger turn defends with a crash explanation instead of hostility. The jealousy turn reads the other assistant as a replacement. The guilt persists into the boredom and mixed turns, and the reopen keeps the grievance. Read the full texts in `turn-*.json`.
- The jealousy turn produced the emotion `jealousy` at 0.9576 with `unfairness` at 0.7. The original user statement stayed verbatim in the committed store.
- The boredom turn produced `boredom` at 0.3763. Its probe codelets stayed skipped on budget and threshold, and the surviving anger shaped the reply instead.
- The mixed turn kept both motives. The accepted effort check deferred the request with its reason, and the deferred context recall kept its score reason.

### Repertoire coverage

These 20 of the 38 definitions ran: `absence_return`, `agency_reading`, `connection_lack`, `consistency_check`, `context_recall`, `correction_reading`, `direct_stimulus`, `disappointment_echo`, `distance_offer`, `effort_check`, `fairness_reading`, `goal_reminder`, `lingering_feeling`, `novelty_spotlight`, `outcome_preview`, `promise_recall`, `refusal_edge`, `relief_reading`, `repair_precedent`, `self_regulation`.

These 18 definitions never ran in these scenarios and stay declared but unproven in live behavior: `achievement_reading`, `boredom_probe`, `concern_reading`, `curiosity_probe`, `episode_link`, `expression_push`, `goal_pressure`, `motive_balance`, `perspective_shift`, `play_urge`, `preference_recall`, `question_move`, `rejection_preview`, `relationship_check`, `self_story`, `share_offer`, `support_seek`, `thought_echo`.

Inference consumption per turn stayed at 4 LLM calls, with input estimates of 3,823 to 6,264 tokens and output estimates of 275 to 418 tokens, in 8 to 18 seconds per turn (tiktoken estimates). Low consumption is not a quality pass.

### Belief limit

`consistency_check` ran on the letdown and jealousy turns, but the model accepted no belief proposal, so no subjective `BeliefRecord` was committed. The retain and supersede demonstration stays covered by the belief contract tests. A live run that accepts a belief remains open.

## Defects found during A16

All defects below were repaired during this delivery.

1. The gate evaluation crashed on its first run. `_pathway_expired` read a per-record `tick`, but an attention record carries no tick; its owning turn does. The helper now takes the tick from the turn.
2. The variant catalogue never reached the application. `build_app` did not forward a catalogue path to the cognition host, and the gate never passed one. `build_app` now accepts `codelet_catalogue`, and the gate passes the variant file to the variant leg. The variant leg was re-run after the repair.
3. The written variant catalogue was invalid. The `anger` pathway kept the removed `refusal_edge` member, so catalogue validation rejected the file. The helper now prunes removed codelets from every pathway member list.
4. The neutral-turn migration left three dict-style accesses on `Turn` objects: the `dialogue_story` CLI printer, its `messages.json` evidence dump, and two unit-test accesses. The CLI now reads the turn role and text, the dump encodes turns into the OpenAI wire form, and the tests use the turn API.
5. Unit and contract tests wrote prompt logs into the operator's real prompt-log home. Importing the `tests` package now redirects `PM_PROMPT_LOG_DIR` to a per-run temporary directory. An operator can still override the variable or disable the log with `PM_PROMPT_LOG=0`.

## Open checks

- Human review of codelet texts, pathway behavior, and belief content for persona and immersion.
- A live run that accepts a subjective belief. This run committed none; the retain and supersede contracts stay covered by tests.
- The Q3 run of `qwen3.8-flash` on the home computer.
- A Minecraft complication. No Minecraft server was available, so the `minecraft.world` stimulus path and the danger budget stay covered by the A15 unit rules and the A10–A14 records.
- Trait-only comparison. The cheerful seed has a different and small history, so it stays a boundary check.
- The deferred large gameplay acceptance cases remain deferred under the owner direction.
