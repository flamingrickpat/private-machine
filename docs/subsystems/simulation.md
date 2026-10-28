# Simulation and review

## Summary

The simulation browser runs real PM companions in separate processes.
Each participant owns a home, ghost, workflow files, logs, and clock.
The harness owns scenario actors, public message routing, and coordinated checkpoints.

Parent: [engineering manual](../index.md).
Requirements and evidence: [K002](../work_items/k002/index.md).
Recovery rules: [causal memory](../work_items/k002/causal-memory.md).
Capture format: [record packages](record-packages.md).

## Start the browser

Start the local model and embedding services first.
The selected configuration must name their actual endpoints and models.

```powershell
uv run --locked python -m simulation --config config.alpha.example.yaml --port 8095
```

Open `http://127.0.0.1:8095/`.
The default homes directory is `~/.pm/private-machine`.
The default evidence directory is `.runtime/simulation/`.
Use `--homes` or `--data` to select different directories.

1. Select an existing home, or load the unchanged Spamuel canon into a new home.
2. Inspect its history and select a checkpoint before creating a fork.
3. Select the primary character and an optional connected character.
4. Enter a scenario and a start time with an explicit UTC offset.
5. Generate the plan, inspect its facts, and edit its JSON when necessary.
6. Create the run, then select **Start / resume**.
7. Inspect conversation, mandatory facts, and the attempt report.

Each run forks its source homes before startup.
Connected personas retain their own identities and see the partner as Other Spamuel.
The endpoint does not announce whether that entity is human or another PM.
The backend accepts a graph of participants; the browser currently selects two.
Minecraft and its managed workflows remain disabled for simulation participants.

## Scenario execution

The planner first extracts mandatory facts, then creates the editable plan.
A timed fact must occur at its stage time.
The actor generates a draft. A focused check verifies its identity and current stage facts before delivery.
Rejected drafts remain available for debugging.
The actor sends ordinary entity messages through the public API.
The director checks the delivered conversation against the stage facts.
Each completed fact cites actual actor delivery IDs. Timed facts must retain their specified time and recipient.
It cannot end the scenario successfully while mandatory facts remain missing.
Companion disagreement does not cancel the actor's required commitments.

No harness quota limits turns, model calls, cost, or execution time.
The configured serving context still limits an individual prompt.
Invalid model output retains a receipt before repair.
A context overflow, process failure, or contradictory scenario execution fails the attempt explicitly.

Virtual time advances through semantic wake and private-thought deadlines.
Model transport, process supervision, and diagnostics use real time.
Fast scheduler polls do not create artificial inference steps.
Clock receipts retain the virtual origin and timer deadlines for resume.

## Pause, stop, and resume

Pause and stop fence new routing, drain runnable work, and retain parked workflows.
Both operations retain the plan position and pending deliveries for resume.
A forcefully interrupted process remains a failed attempt, with its evidence retained.
Main cognition determines which inputs survive recovery.
The controller can resend an excluded delivery through its stable delivery ID.
A failed single-participant run retains its causally accepted progress.
A failed multi-participant run restores all participants and routing state from the latest coordinated barrier.
Every new run records an initial barrier before any public exchange.

A coordinated checkpoint contains every participant snapshot, the shared time, and the transit ledger.
A participant snapshot alone cannot retract speech that another participant already accepted.
Completed attempts and their source copies remain immutable evidence.
Each later attempt receives a separate source copy, report, and failure receipt.
The source receipt records the repository revision, local file hashes, and installed dependency origins.

## Public companion API

POST `/v1/completion`, `/v1/completions`, or `/v1/chat/completions` with an OpenAI-shaped body:

```json
{"messages":[{"role":"user","content":"Hello"}],"delivery_id":"client-message-1"}
```

Only the final message content supplies the stimulus.
PM owns conversation history; earlier supplied messages do not become memory.
The endpoint returns HTTP 202 after durable delivery, without waiting for a reply or an embedding call.
An optional `user` names the speaker.
An optional `entity_ref` distinguishes connections that share a display name.
Reusing a delivery ID with different content raises an error.

Connect to `/v1/events?after=SEQUENCE:ORDINAL` for public companion speech.
Each event includes the knoxel output ID and its durable cursor.
Speech can include several messages or unsolicited messages.
Reconnect with the last cursor and deduplicate output IDs.
The cursor follows commit order because IDs can be reserved before other commits finish.

## Control API

The controlled participant starts with `python -m simulation.participant`.
The harness supplies its home, configuration, and readiness file.
These controls remain separate from the asynchronous public API.

| Endpoint | Behavior |
| --- | --- |
| `/control/start` | Start real cognition with `clock_start`; state reports loading and errors. |
| `/control/state`, `/control/status` | Read clock, activity, timers, parked work, and the output cursor. |
| `/control/send` | Deliver `text`, optionally after advancing to `ts`. |
| `/control/advance` | Advance through semantic deadlines and return buffered public outputs. |
| `/control/advance-step` | Advance one coordinated clock step without waiting for resulting work. |
| `/control/env` | Queue observed environment events at their recorded times. |
| `/control/freeze`, `/control/thaw` | Fence intake and autonomous timers for a checkpoint barrier, then reopen them. |
| `/control/snapshot`, `/control/restore` | Save or restore a named ghost and its clock receipt. Names cannot overwrite prior snapshots. |
| `/control/delivery` | Check whether a delivery exists and whether cognition accepted it. |
| `/control/trace/ID` | Read the exact output record and its creating transaction. |
| `/control/shutdown`, `/control/stop` | Fence intake, drain running work, save parked work, and exit. |
| `/control/inject` | Reject experimental controls. The owner deferred them to another work item. |

The ordinary PM chat server also exposes runtime lifecycle controls.
Process startup and restore belong to the participant controller.
Compatibility does not suppress runtime errors or overwrite mental state.
Dependent controls wait for startup. Status remains available during loading.
Legacy output and trace responses include empty gateway-call lists with an explicit unavailable-linkage reason.
PM prompt IDs remain in persona logs; they are not contributor gateway IDs.

## Evidence and human review

Attempt reports link conversation, private activity, GWT cycles, transactions, and captured prompt files.
Ready review packages use the accepted causal view.
Raw branch records remain separate debugging evidence.
Record links follow captured IDs through appraisal, codelets, attention, decisions, and public replies.
Expand a record to inspect all captured fields except embeddings.
Uncaptured references remain explicit.

Human feedback is separate from immutable captures.
Automated fact checks do not supply companion-quality ratings.
Register offline runtime packages with:

```powershell
uv run --locked python -m simulation.review register-package .runtime/debug/chat.json --id example --title "Example review"
```

The command prints a direct browser URL.
`python -m simulation.review feedback` reads saved owner feedback when the owner requests it.
The old development review server is retired.
Its registrations remain in `simulation/review/legacy-registry.json`.
Existing capture files under `.runtime/dev-review/` remain unchanged.

## Code ownership

- [`simulation.engine.RunEngine`][] owns scenario progress, routing, and attempt evidence.
- [`simulation.participant.Participant`][] owns one controlled PM process.
- [`simulation.review.catalog.register`][] registers immutable offline captures.
- [`pm.app.control.RuntimeControl`][] supplies ordinary runtime controls.
- [`pm.app.transport.public_events`][] supplies replayable public speech.

The simulation imports the runtime and workflow assembly.
Product runtime code does not import the simulation application.
K003 can add richer inspection through the capture and presentation interfaces.
