# K002 simulation harness

## Summary

K002 adds a separate simulation application under `simulation/`.
The application connects real PM processes through public APIs and controls virtual time.
It records scenarios, conversation, execution telemetry, and reports.
The runtime retains raw causal history and restores its accepted view by default.

Parent: [local Kanban board](../../workflows/kanban.md).
Design discussion: [causal memory and recovery](causal-memory.md).
Source report: [contributor A/B test](../../ab_test_2026-10-23.md).
Process: [work-item development](../process.md).

## Accepted owner requirements

These requirements come from the K002 card and the owner discussion.
They describe requested behavior. They do not establish an implementation result.

| Area | Required behavior |
| --- | --- |
| Ownership | The harness and its simulated participants live under `simulation/`. PM retains its real runtime and workflows. |
| Personas | Create new persona homes from the unchanged Spamuel canon. The owner removed the old homes. |
| Browser | Browse characters and history, fork a checkpoint or load canon, enter a scenario, and inspect its generated plan. |
| Operation | Edit the scenario plan, start, pause, stop, resume, and inspect the report. |
| Participants | Support several PM processes now. Each process owns one persona. |
| Connections | The harness connects participants. A participant sees another entity through ordinary public input. |
| Identity | Transport does not reveal whether the partner is human, simulated, or another PM instance. |
| Scenario | Scripted facts are mandatory. Early termination occurs only on failure. |
| Models | Use the default model. Add no artificial cost, call, turn, or elapsed-time budgets. |
| Minecraft | Disable the body and its workflows for these simulations. |
| Evidence | Capture actual prompts, outputs, workflow activity, memory inputs, causal links, and failures. |
| Files | Keep pm-coder sessions and persona diagnostics under their owning persona home. |
| Review | Replace the development review desk with the simulation report and review path. Preserve captured evidence. |
| K003 | Keep capture and viewer interfaces modular. K002 does not implement the future memory debugger. |
| Recovery | Startup opens the clean causal view and retains raw history for debugging. |
| Memory identity | Every logical knoxel change creates a new knoxel ID. Existing knoxel payloads remain unchanged. |
| Mental events | Thoughts, appraisals, and other occurred mental features remain immutable. Persistent goals and intentions can acquire successor records. |
| Shutdown | Drain running work and save workflows parked for outside input. Forced interruption remains separate. |

## Public interface

The owner changed the proposed completion endpoint to asynchronous stimulus delivery.
The request accepts the final message content. PM owns conversation history.
The HTTP response acknowledges delivery without waiting for companion speech.
A WebSocket carries later replies, including several replies to one stimulus.

This contract uses an OpenAI-shaped request body with PM response semantics.
The contributor compatibility routes remain a separate contract.
The inspected routes cover start, advance, send, environment events, snapshots, restore, inject, state, trace, and stop.

The implementation uses stable stimulus IDs, output IDs, and event cursors.
These identifiers permit reconnection and delivery deduplication.
The harness records routing receipts separately from a persona's public speech.
Control endpoints expose shutdown requests and shutdown progress.

## Persistence requirements

Main cognition arbitrates acceptance of new internal and external stimuli.
A completed older action can remain valid during a newer unfinished cycle.
Its result must retain every actual input dependency.
An unfinished selected action must remain available for restart.
Clean failure can finish an action. A process crash does not prove clean failure.

The owner requires saving all activity and opening its clean causal view by default.
Raw provisional history remains available for debugging.
The [causal memory contract](causal-memory.md) separates stored history from the accepted view.

## Planned checks

The implementation needs focused storage, action recovery, clock, protocol, and routing checks.
Real acceptance uses the full Spamuel canon and the configured model.
The cat scenario must cross midnight and reach its mandatory morning event.
A second scenario connects separate PM processes through the public transport.
Recovery must retain valid completed speech and remove excluded stimuli from the restored persona view.
Interrupted approved dialogue must resume without another action-selection call.
Reports must retain failed attempts and exact source revisions.

## Current evidence

The source audit is under `.runtime/kanban/k002-20261027-202608/`.
Its `source-audit.json` records repository revisions and installed dependency origins.
The contributor checkout is at `542143f4b01dfc75add7fff66bc9efac939cbb85`.
The current runtime audit started at `aa57839e517b19d28776083dac5e17238a717690`.
The initial strict documentation build passed in 102.08 seconds.

Implementation is in progress. Focused recovery, transport, and workflow checks have passed.
Real acceptance is still running. No companion-quality rating has been supplied.

Operation: [simulation and review](../../subsystems/simulation.md).
Verification: [K002 execution evidence](verification.md).
