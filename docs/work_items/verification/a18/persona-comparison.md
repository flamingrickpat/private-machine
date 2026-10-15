# A18 matched persona comparison

## Summary

Six real inference replies compare dramatic and cheerful expression against identical inspected facts.
Human review remains pending.
This bounded probe does not run the companion application.

Parent and human review: [A18 verification](../a18-self-model.md).
Source: [live evidence](live-evidence-20261015.md).

## Matched persona comparison

The comparison uses `dev/work_items/alpha_a18_persona_comparison.py` and real inference.
Both variants receive identical inspection evidence and identical questions.
Each question starts a separate conversation with no imported history.
Only the character card changes: Spamuel supplies the dramatic voice; Sunny supplies the cheerful voice.
The complete Spamuel history remains part of the primary application gate only.

```powershell
uv run --locked python dev/work_items/alpha_a18_persona_comparison.py `
    --config config.alpha.openrouter.yaml `
    --facts .runtime/alpha/20261015-110410-a18/comparison-source-facts.json `
    --out ".runtime/alpha/<stamp>-a18-personas"
```

The frozen facts come from the primary gate after its idle phase.
The initial overlay marks four capabilities enabled: dialogue, private thought, memory tools, and source inspection.
Minecraft body is implemented but disabled in that instance.
The initial overlay supplies no observed execution receipt.
The comparison supplies no episode or causal trace.
The idle result is outside the matched evidence packet.
These limits apply equally to both personas.

The canonical evidence JSON has SHA-256
`a7bd3347a66914147506114931dae51bad369fbd2e958a83133e00e98bd1d502`.
Every result repeats this digest.
`persona-comparison-retry/` retains the packet, six requests, six responses, and the completed comparison receipt.
The first attempt stopped before inference because the Python YAML reader was slow on the 30 MB seed.
`persona-comparison/interruption.txt` retains that diagnosis.
The retry uses the compiled YAML reader with unchanged seed bytes.

This bounded comparison measures expression against matched facts.
It does not establish full-application persona behavior, emotional continuity, or causal appraisal.
A reply about a next choice does not establish execution of that choice.

All six inference calls completed.
Both capability replies distinguish enabled features from observed execution.
Both self-explanation replies state that the supplied evidence cannot establish the cause of a last action.
The emotional replies propose later choices; the comparison executes none of those choices.
These are source-review observations, not human quality judgments or a scored factual benchmark.

| Probe | Dramatic observation | Cheerful observation | Shared factual boundary |
| --- | --- | --- | --- |
| Capabilities | Lists evidence states and limits | Lists evidence states and limits | Four enabled capabilities; Minecraft body lacks enabled evidence |
| Self-explanation | Says the trace gives no actual causal story | Says no last-action causal record is present | No supplied episode or cross-loop link establishes a decision cause |
| Emotional discovery | Uses sarcasm about accuracy; gives no concrete next action | Expresses caution and proposes a comparison table | Neither claims observed Minecraft execution |

`persona-comparison-retry/input-validation.json` records the successful comparison of all six request inputs.
