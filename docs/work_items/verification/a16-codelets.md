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
Gate: `dev/work_items/alpha_a16_codelet_gate.py`.

## What A16 delivered

- The catalogue grew from 11 codelets (`a15.3`) to 38 codelets (`a16.1`) across the ten required families.
- Eleven versioned pathways connect accepted precursor proposals to later activation. See [codelet pathways](../../subsystems/codelet-pathways.md).
- Every definition declares its context requirements and its expected consequence.
- Every proposal kind has a named consumer with a durable effect. See the [consumer map](../../subsystems/codelets.md#proposal-consumers).
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

GATE_RESULTS

## Defects found during A16

DEFECTS

## Open checks

- Human review of codelet texts, pathway behavior, and belief content for persona and immersion.
- The Q3 run of `qwen3.8-flash` on the home computer.
- A Minecraft complication. No Minecraft server was available, so the `minecraft.world` stimulus path and the danger budget stay covered by the A15 unit rules and the A10–A14 records.
- Trait-only comparison. The cheerful seed has a different and small history, so it stays a boundary check.
- The deferred large gameplay acceptance cases remain deferred under the owner direction.
