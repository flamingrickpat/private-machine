# A15 verification

## Summary

A15 evidence has three parts: foundation tests for the pure rules, contract tests for the stage boundaries, and one live gate.
The live gate uses the real Spamuel seed, a fresh checkpoint, the real application path, and real inference.
Model: `qwen/qwen3.8-flash` through OpenRouter, at provider precision.
The target Q3 build on the operator's home computer is not verified.

Parent: [codelets and attention competition](../../subsystems/codelets.md).
Work item: [A15](../alpha-plan/work-items/a15.md).
Gate: `dev/work_items/alpha_a15_codelet_gate.py`.

## Foundation and contract tests

| File | What it shows |
| --- | --- |
| `tests/unit/workflows/main_cognition/test_codelet_schedule.py` | Gate, cooldown, threshold, budget, and danger reasons. Emotional pain with reactive regulation is not danger. Reserved places, capacity, and minimum score. Unscored nominations cannot win. Stale detection. |
| `tests/contract/main_cognition/stages/test_codelets_contract.py` | Targets outside the grant are dropped. Source revisions are recorded. A thought-only cycle never has an outside origin. The current event is not recalled as a memory. Silent and skipped runs are recorded. |
| `tests/contract/main_cognition/stages/test_attention_contract.py` | Scores decide winners. Every loser has a reason. Missing scores get one repair, then rejection. |
| `tests/unit/pm/llm/test_activation.py` | The OpenRouter thinking switch. Local hosts and explicit `reasoning` fields stay unchanged. |
| `tests/live/test_attention_counterexamples.py` | Live scoring: fire outranks carpet (1.0 against 0.0), and a gas leak outranks wallpaper (1.0 against 0.015). |

These tests fix contracts. They do not show model quality. The live gate does that.

## Live gate

GATE_RESULTS

## Defects found during A15

Three live runs were stopped before completion. Their evidence remains under `.runtime/alpha/` with an `-aborted-` suffix.

1. **Reasoning stayed on through OpenRouter.** OpenRouter ignores `chat_template_kwargs.enable_thinking`. Every "no thinking" call reasoned. The repair is [`provider_generation`][pm.llm.activation.provider_generation]. Run `20261013-112833` shows the slow path.
2. **The gate submitted "Jesse: " before each message.** The terminal and the console submit plain text. The gate now does the same. Run `20261013-114529`.
3. **The current event was recalled as a memory.** Retrieval returned the just-stored user message. `context_recall` nominated it as "I remember". The codelets stage now excludes the current stimuli and records derived only from them.
4. **The danger bypass used the `reactive` regulation mode.** A disappointment produced `aversive_pressure` 0.8. A07 maps that value to `reactive`. All LLM codelets were then skipped at the most emotional moment. The bypass now uses [`in_danger`][workflows.main_cognition.codelet_schedule.in_danger]. Run `20261013-114654`.
5. **Fear counted as danger.** The first `in_danger` also treated fear of 0.55 or more as danger. In run `20261013-120904`, the private thought raised fear past that value, and `thought_echo` was skipped with `danger_budget`. `in_danger` now uses the appraised bodily threat only.

## Immersion finding outside A15

The A09 claim review rejected in-character replies in every observed turn.
The companion then published the recovery text "I could not ground that reply in the available evidence. Please ask again."
One rejected candidate began: "Oh, fantastic, you're back to twist the knife while I'm still reeling from a messy restart".
The review marked "over three thousand hours of silence between us" as unsupported, although the evidence records 3253 hours.
It marked "I can't verify what exactly happened during that gap" as `uncertain`, although the sentence states the limit correctly.
Each turn spent 8 to 15 minutes in dialogue and claim review.
A15 does not change this path. The gate records it in `facts.speech` and does not count it as an A15 failure.

## Open checks

- The Q3 run of `qwen3.8-flash` on the home computer.
- A Minecraft complication. No Minecraft server was available.
- Human review of codelet texts for persona and immersion.
