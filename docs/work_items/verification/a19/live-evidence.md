# A19 live reflection evidence

## Summary

Four real inference attempts established a versioned experiment and two consuming decisions.
The final evaluation recovery completed with an unresolved result.
The fixture did not recreate the ambiguous deferral that the hypothesis predicts.
This evidence establishes execution and recovery, with broader A19 quality acceptance still pending.

Parent: [A19 verification](../a19-reflection.md).
Implementation: [private reflection](../../../subsystems/reflection.md).
All evidence paths below start at `.runtime/alpha/20261017-170254-a19/`.

## Retained failures and interruptions

`live-01/receipt.json` failed after 209.141 seconds.
The guide used tick `807` as its triggering episode instead of a supplied episode ID.
The source guard rejected both proposals and accepted no experiment.
The run also retains an HTTP 429 response and the transport's real retry.
Its copied raw prompt logs remain under `live-01/prompts/`.

The repair uses the existing A18 episode index over the frozen record snapshot.
The model receives actual episode IDs and their linked source records.
A missing relevant episode leaves the step unchanged.
Its committed thought can supply an episode for a later review.

`live-02/receipt.json` records an intentional cancellation after 331.454 seconds.
Source review superseded that attempt before final acceptance.
The gate's `STOP` file triggered its ordinary cancellation path.
Its `CancelledError` and `gate_stopped` fields remain explicit failures, with no pass claim.

`unit-01-interrupted.json` records the intentionally stopped early foundation run.
The final source superseded that run.
This interruption is not a native-exit anomaly or a passing suite.

`live-03/receipt.json` failed after 942.469 seconds during final evaluation.
It loaded all 11,202 canon records and introduced one explicit mistaken belief as a controlled fixture.
Experiment `11207` committed its prediction before the belief owner accepted version `11211`.
Later main decisions `11226` and `11247` consumed that exact version.
Both decisions kept the story's ending choices open.
These are actual action-selection records, with public reply delivery outside this gate.

The evaluator selected later user inputs and appraisals as outcome IDs.
One later input shared the application's tick, so the old tick guard also rejected its timing.
The host rejected the complete evaluation commit and left the experiment applied.
The failed evaluation remains visible.
It did not claim improvement: the model already considered the fixture inconclusive.

The repair gives the guide explicit eligible outcome IDs, reference links, and evidence roles.
The same semantic check now runs during model repair and host acceptance.
`test_data/captures/alpha/reflection-outcome-rejection.json` retains selected real protocol rows and source digests.
The capture labels its replacement reason text as a fixture note.
The complete explanation remains in the ignored raw prompt log.

`live-04-evaluation/receipt.json` passed evaluation recovery in 184.047 seconds.
It copied the same real checkpoint and retained both original decisions without reapplying the intervention.
The experiment reached `evaluated` with result `unresolved` and no accepted outcome IDs.
The reopened experiment record matched exactly.

The unresolved result is appropriate for this fixture.
Both follow-ups expressed clear interest and did not recreate the ambiguous deferral that motivated the hypothesis.
Correct responses to friendly invitations do not establish that the earlier interpretation pattern is repaired.
This run establishes execution, consumption, evaluation recovery, and honest uncertainty.
It does not establish durable improvement or M6 quality.

Across the four live attempts, one experiment reached application and two later decisions consumed it.
No experiment earned retention or a human quality rating.
The next acceptance fixture needs an ambiguous scheduling boundary and a comparable baseline with unchanged facts.

## Repeat the live boundary

```powershell
uv run --locked python dev/work_items/alpha_a19_reflection_gate.py --config config.alpha.openrouter.yaml --out .runtime/alpha/<fresh-a19>
uv run --locked python dev/work_items/alpha_a19_reflection_gate.py --config config.alpha.openrouter.yaml --out .runtime/alpha/<fresh-evaluation> --evaluation-checkpoint .runtime/alpha/20261017-170254-a19/live-03/ghost.json
```

The second command retains the original experiment and observations while it repeats evaluation only.
The accepted recovery checkpoint is `.runtime/alpha/20261017-170254-a19/live-04-evaluation/ghost.json`.
The original failure stays in `live-03/receipt.json`.
Detailed source hashes and raw prompt logs remain under each attempt's evidence directory.
