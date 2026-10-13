# Codelet discovery

## The idea

A16 delivers a fixed repertoire of codelets.
Beta can let the companion grow the repertoire itself.
The companion proposes new codelets, simulates their effect, critiques them, and keeps
the survivors.
This is the old codelet-discovery pipeline.

A growing repertoire lets the companion develop behavior its designer did not write.
It is one path to novelty and long-term change.

## Rough scope

- Generate a candidate codelet from an observed need.
- Simulate the candidate over recorded situations.
- Critique the candidate with a separate step.
- Keep the survivors and retire the failures.
- Version the catalogue and keep a reversal path.

Read `pm_next/experiments/codelet_discovery/` and
`private-machine/pm/subsystems/codelet_discovery/`.
The old driver has a missing import. Repair or rewrite it.

## Open questions

- What is a safe candidate? A codelet cannot change state or command the body.
- How do we prevent runaway growth or duplicate codelets?
- Who reviews a candidate before it activates?
- How does a discovered codelet meet the no-role-play evidence rules?
- Do we allow a candidate to change the scheduler rules too?

## Notes

This depends on A16 and on the grounded prompt style.
The old catalogue prompts invented senses and past events. The new ones must not.