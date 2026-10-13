# Workspace momentum

## The idea

Today the workspace is rebuilt each cycle.
A human mind does not rebuild.
It carries a working state with it, and that state decays.
The companion should keep some momentum between cycles.
Attention, mood, and a salient thought should persist for a while instead of resetting.

Momentum makes the companion feel continuous.
It also makes a distraction feel like a distraction, because the previous focus is still
present and can return.

## Rough scope

- Carry the last workspace frame into the next cycle.
- Decay the carried content with elapsed time.
- Let a strong or unresolved item keep its activation across cycles.
- Let an interrupting event win attention without deleting the carried state.
- Persist the carried state across a restart with the same decay rules.

This is the Current Situational Model from the old generations.
Read `pm_next/pm_core/model/csm.py` and the coalition code in `private-machine/pm_lida.py`.

## Open questions

- Which fields carry, and which stay per-cycle?
- How fast does each field decay?
- Does momentum belong in the workspace record or in the cycle state?
- How does momentum interact with the A17 shard contract?
- How do we measure "feels continuous" without inventing a metric?

## Notes

A17 must design its shard contract so this idea can be added without a rewrite.
The user asked for this in beta, not alpha.