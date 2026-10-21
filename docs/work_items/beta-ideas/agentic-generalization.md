# Agentic generalization

## The idea

Alpha uses deterministic rules for attention selection, routing, and public review.
The rules are simple, testable, and easy to diagnose.
Beta replaces a rule with an agentic equivalent when a tested reason exists.
The rule stays as the regression baseline.
A replacement must match or beat the rule on the same cases.

This turns the "explicit logic" concern into a method.
The deterministic stage is not a defect. It is the reference behavior.

## Rough scope

For each rule:

1. Record the rule, its files, and the cases that exercise it.
2. Keep the rule and its tests as the baseline.
3. Build the agentic candidate behind the same interface.
4. Compare the two on the recorded cases and on new cases.
5. Keep the rule when the candidate is not better.
6. Keep the agentic candidate versioned and reversible.

Candidate rules:

- Attention winner selection.
- Action routing.
- Public speech review.
- Codelet activation scoring.

## Open questions

- How do we compare an agentic candidate to a rule without a hand score?
- How many cases are enough to trust a replacement?
- What happens when the agentic candidate is sometimes better and sometimes worse?
- Do we keep both and let the companion choose?

## Notes

Read the [generalization ladder](../alpha-plan/index.md#generalization-ladder).
A20 records the rule baselines for this step.