# Wait for a stimulus

## Summary

`Sleep` now commits an actual pause of routine cognition in real mode.
Outside evidence or a deadline ends the pause.
A spoken reply also arms a short quiet window, so the next cycle can read an answer.
A timeout supplies a private scheduling stimulus, not a forced public reply.
The controller checks deadlines without a model call.
Full A18C acceptance remains incomplete.

Parent: [private loop](private-loop.md).
Work item: [A18C](../work_items/alpha-plan/work-items/a18c.md).
Evidence: [verification](../work_items/verification/a18c-control.md).
Runtime: [`workflows.cognitive_control.host.CognitiveControl`][].

## Action semantics

| Choice | Effect |
| --- | --- |
| No decision, or a route to `Ignore` or `Idle` | No external action now. Routine private thought can continue |
| Quiet private work | Active inference without public speech |
| `Sleep` | Suspend routine main and private inference until a qualifying stimulus or deadline |
| A spoken reply | Suspend routine inference for `post_reply_seconds`, then run the next cycle |

A decision to wait can route to `Sleep`. The route call can supply `arguments.timeout_seconds`.
[`RouteAnswer`][workflows.main_cognition.schemas.RouteAnswer] rejects nonnumeric, nonfinite, boolean, zero, negative, or excessive durations.
The decision text becomes the reason of the wait. See [decisions, routing, and expectations](decisions-and-routing.md).
Allowed values span 1 through 86,400 seconds.
Invalid stage output enters the existing format-repair boundary.
The host also enforces the operator maximum before committing a wait.

`Sleep` dispatches no workflow.
Already authorized speech and external work remain active.
A terminal or relevant progress event can wake cognition about that work.
An accepted pause does not complete an unanswered public exchange.

## Quiet window after a reply

A cycle that dispatches a `Reply` or a `ToolCallAndReply` arms a wait of
`post_reply_seconds`. The companion has just spoken, so the next thing worth
reading is her partner's answer rather than another of her own thoughts. This
makes a slow cycle cheap to live with: she answers, then waits, then runs one
cycle that has had time to accumulate private thought and world observation.

[`workflows.main_cognition.phases.commit.wait_request`][] builds the request.
A `Sleep` route keeps its own duration and wins over the window. A cycle that
said nothing arms nothing, and a rejected reply arms nothing because it never
reached the partner. Set `post_reply_seconds` to 0 to disable the window.

A wait does not stop a managed run. A reply that also started Minecraft work
therefore leaves that work running through the window.

## Assembly and configuration

[`workflows.companion.build_app`][] supplies `cognitive_control_config` in real mode.
It accepts [`workflows.cognitive_control.schemas.ControlConfig`][] or a matching dictionary.
`wait_seconds` defaults to 1,800 seconds.
`maximum_wait_seconds` defaults to 86,400 seconds.
`post_reply_seconds` defaults to 600 seconds.
The controller checks persisted deadlines once per second.
The private timer retains its usual 30-second default.

The `cognitive_control` section of the model configuration supplies these limits.
[`pm.llm.config.ModelConfig`][] keeps that section as a plain mapping, because the
runtime must not import `workflows`. [`ControlConfig`][workflows.cognitive_control.schemas.ControlConfig]
validates it during assembly, so an unknown key fails at startup instead of being
ignored.

For a short diagnostic run, construct the application with:

```python
cognitive_control_config={"wait_seconds": 45, "maximum_wait_seconds": 60}
```

This construction argument is not yet an operator-console setting.
Episode fields in `ControlConfig` bound real private episodes.
`maximum_focus_bias` bounds attention influence, and `mental_timeout_seconds` bounds each mental-agent session.
These limits are not quality thresholds.

## Commit and interruption boundaries

Main cognition stages its ordinary result and optional wait in one write batch.
[`workflows.cognitive_control.host.CognitiveControl.commit`][] holds the store lock through acceptance.
A failed checkpoint accepts neither result nor wait.
No model or network call runs inside this boundary.

Fresh outside input advances the control generation before mailbox delivery.
Accepted cycles retain its watermark across restart.
Pending input also prevents a wait between mailbox drain and run start.
An older `Sleep` result then records a superseded decision instead of hiding the input.
Accepted waits and fresh input cancel the active private run cooperatively.
A generation check at private commit rejects a late result even if cancellation arrives too late.
Main intake discards queued private output from an older generation.

Two tiers of evidence exist. The first tier ends a wait at once:
`user.input`, `workflow.question`, and the `workflow.event` states `failed`,
`cancelled`, `interrupted`, and `waiting_for_input`.
The second tier is held for the next cycle instead: `minecraft.world` and the
`workflow.event` states `completed` and `gameplay_progress`.
Held evidence stays in the store. The controller parks its envelope and
publishes it again when the wait ends, so the body poller still sees its
observation as unclaimed and clears it later.
Without this split, the 120 second world poll ended a 600 second quiet window
after two minutes.
Self-generated frames, private thoughts, and ordinary running telemetry do not end a wait.
A private-only main result also fails its final generation check after newer input.

## Durable records and restart

The wait Feature carries `control_kind: inference_wait` and `status: waiting`.
It records `started_at`, `wake_at`, the accepted duration, and original source IDs.
A wake changes its status to `awake` with an attributed cause and time.
A rejected late decision uses `control_kind: inference_wait_rejected` and `status: superseded`.

Timeout acceptance creates one Feature with `control_kind: inference_wake` and `wake_status: pending`.
The record retains the wait source and actual elapsed seconds.
The bus then publishes `cognition.wake` to main cognition.
Its stimulus kind remains `cognition.wake`, not a user report or workflow result.
Main consumption changes `wake_status` to `consumed` in the same commit as appraisal.

A crash before delivery leaves a pending wake for replay.
A crash after main acceptance leaves a consumed wake that needs no new inference.
Duplicate envelopes for consumed wakes are discarded before a model call.
Cancellation releases a pending wake for another delivery.
The bus itself remains nondurable.

Wake records deliberately avoid the public `publication_status` field.
Public recovery must never publish elapsed-time evidence as companion speech.
Recovery also excludes original sources already handled by an accepted wait.
Those user records retain `exchange_status: pending`.
They are not new outside input after restart.

## Timing and diagnostic limits

Deadlines use UTC wall time because they survive process stops.
A backward clock adjustment delays expiry.
A forward adjustment can expire a wait early.
Elapsed-time text never claims offline cognition or unobserved activity.
Use one active application owner per checkpoint.

The A18B driver recognizes a committed wait tied to the current pending exchange.
It records `accepted_pause` and stops generation without another user message.
Unseen scenes remain incomplete, and the case earns no aggregate acceptance.
The [authoring guide](cognitive-evaluation-authoring.md) describes this runtime protocol.
An ordinary quiet cycle is not proof of a deliberate wait.
A bounded private episode now owns continuation outside the wait path.
It closes on a step, time, or stalled-progress limit and parks the concern for review.
The [remaining design](../work_items/alpha-plan/deliberate-control.md) describes what is still pending.

## Run the focused real gate

```powershell
uv run --locked python dev/work_items/alpha_a18c_wait_gate.py --config config.alpha.openrouter.yaml --out .runtime/a18c/NEW-RUN
```

Use a fresh output directory.
The gate imports the full Spamuel seed and uses real configured inference.
It checks conversation, selected waiting, external wake, restart, and timeout appraisal.
It retains `receipt.json`, `facts.json`, `bus.jsonl`, replies, kernel artifacts, and the checkpoint.
It does not establish agentic focus, private episode quality, or the complete A18B repertoire.
