# Compare simulation results

Use the same scenario and starting memory to compare changes to a companion.
The [overnight Spamuel baseline](../workflows/spamuel-baseline.md) includes a
conversation, profiler spans, and final memory from one completed run.

- [Run Jesse and Spamuel in Minecraft](../workflows/simulation-jesse-spamuel-minecraft.md)
- [Inspect a local run](../workflows/simulation-workspace.md)
- [Read the execution timeline](timeline.md)

The four numbered launchers use versioned scenarios and keep each run’s evidence
under `.runtime/simulation/runs/`. Use a distinct run ID for each candidate.
Compare the visible behavior before comparing speed. A short wait does not show
that the companion understood the request or completed a world action.

The [campaign controller][simulation.baseline.BaselineCampaign] also groups
the four scenarios for admission, metrics, and evidence export.
