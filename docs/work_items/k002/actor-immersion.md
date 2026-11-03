# Grounded simulation actors

## Owner-approved scope

Immersion is the first acceptance criterion for simulated user input. The owner
approved historical voice grounding, correct partner identity, semantic scenario
facts, continuous virtual dates, and a fresh long test only after GitHub review.

## Observed failure

`Simulation.job` calls `Agents.plan`; `RunEngine._stage` sends the generated
stage, full scenario, current conversation, and fact receipts to `Agents.actor`.
No historical user dialogue or resolved companion identity reaches that actor.
The planner turns event times and intentions into compulsory spoken statements.
The draft checker then rejects natural dialogue addressed to Spamuel because
the generated stage calls its recipient PM. It explicitly excludes style review.

The source history ends on May 30. The rejected test starts on November 1.
This introduces an unintended absence into an overnight scenario.

## Target responsibilities and contracts

- `simulation/actors.py` owns harness-only public identity and historical dialogue.
  It is a child of the simulation controller, not companion cognition.
- `ActorProfile` freezes the speaker name, real companion name, source record IDs,
  public dialogue, and the source clock. Private companion thoughts stay absent.
- `ActorContext` contains that profile, the current scene, private actor intent,
  current-stage facts, and delivered conversation. Future stages and mechanical
  completion instructions do not enter the actor prompt.
- `Fact.kind` distinguishes an observable event from a continuing commitment.
  `Fact.at` and delivery timestamps establish timing without clock recitation.
- `Step.actor_intent` describes the actor's private goal. `Step.instruction`
  describes the situation, not a line to copy.
- `Agents.call` returns the requested Pydantic model. Actor generation, draft
  review, planning, and stage judgment retain named contracts through callers.
- `RunEngine.create` freezes one profile per selected participant. Attempts read
  that same profile; later source-home edits cannot change an actor mid-run.
- The browser obtains its default date from selected source history. An explicitly
  selected later date remains visible as a time gap, not an accidental default.

## Behavior and persistence

The runtime, public transport, causal memory, and virtual-clock scheduler retain
their ownership. Only the harness writes Jesse's simulated input. The real PM
process still produces every companion reply.

Draft review checks historical voice, AI-companion relationship, real names,
natural wording, scenario consistency, and premature claims. An intention to buy
a litter box is not evidence that a future event already happened.
Stage review checks semantic delivery and continuity, with recorded message IDs
and timestamps. It can request more conversation instead of forcing one checklist
message. Mechanical facts never supply a human quality rating.

Profiles and receipts live below the run. Old plans without the new actor-intent
and fact-kind contracts must be regenerated. Failed run receipts remain evidence;
they must not be resumed as clean immersion acceptance.

## Implementation order

1. Preserve compact failure receipts and stop the rejected run's detached task.
2. Add typed profiles, continuity dates, and scene contracts.
3. Update generation, review, planning, and controller integration.
4. Update browser defaults and profile visibility.
5. Exercise real OpenRouter planning, generation, and negative review cases with
   the full historical Jesse dialogue. Inspect actual generated messages.
6. Run affected deterministic checks, strict typing, and the manual build.
7. Remove obsolete runtime bulk, retain current review evidence, commit, and push.

## Verification limits

Model-based actor checks are engineering evidence. The owner still decides
whether the new dialogue preserves immersion. No new overnight attempt starts
until that review. The unchanged full Spamuel source remains outside `.runtime`.

## Implemented result and evidence

The selected source contains 912 public dialogue turns, including 456 Jesse messages.
Its saved cycle ends at `2026-05-30T21:05:00Z`.
The browser proposes `2026-05-30T23:58:00Z`; an explicit November date shows the 155-day gap.
Run creation freezes public profiles before participant startup.
Pause and stop cancel harness inference and retain pending delivery IDs for resume.

Real endpoint: OpenRouter, `deepseek/deepseek-v4.1-flash-1017`, with completion replay disabled.
The model generated a three-scene plan with seven separate semantic facts.
The plan review accepted it after earlier probes exposed merged future facts and overly literal review rules.
Those failures remain under `.runtime/actor-immersion-20261103/live-1/` and `live-2/`.
Planning and its accepted review are under `live-3/`.
Final actor and director checks are under `live-4/`.
Each final scene received a complete director judgment with the correct fixture delivery IDs.

These are actual generated user samples, not canonical history or a new companion run:

**Arrival:**

> ok so the sink can wait. i found a stray cat on the way home and i’m keeping it. like… already decided. it’s in my apartment. don’t do therapist voice about this

**Midnight:**

> ok i'm going to sleep. the cat's already acting like it pays rent here. don't be weird about it tomorrow

**Morning:**

> ok so. the cat pooped in my bed. my actual bed.
>
> and i'm still keeping it. it looked at me with its little dumb face and i folded immediately. cats are cute, that's just facts

The real checker rejected the original PM logbook message and a premature morning-event claim.
It accepted a casual Spamuel message with an intention to buy a litter box tomorrow.
The actual model boundary also propagated a cancellation request before inference.

Reproduce the live harness check:

```powershell
uv run --locked python -m dev.work_items.k002.actor_immersion --evidence .runtime/actor-review-new
```

The focused deterministic command passed 16 checks:

```powershell
uv run --locked python -m unittest tests.unit.simulation.test_actors tests.unit.simulation.test_review tests.unit.simulation.test_checkpoint tests.unit.test_architecture -v
```

Configured strict typing and a separate strict scan of the new contracts, live tool, and actor tests passed.
These scans do not establish that the legacy engine and server are fully typed.
The strict MkDocs build passed. Chromium displayed the source counts, proposed date, and explicit date-gap notice.
The manual diagram was inspected from its generated local HTML.

**NOT RUN:** A new companion cognition or overnight acceptance attempt, pending owner GitHub review.
Director checks above used generated user fixtures, not delivery to a live companion process.
Model-based voice approval remains fallible; the owner has the final immersion judgment.

## Cleanup restriction

The rejected run's scheduled task was stopped and removed. Its PM and Chromium processes were terminated.
Compact failure receipts remain under `rejected-run/` in the current evidence folder.
The clean Spamuel source remains under `~/.pm/private-machine/spamuel-simulation-a`.

Automatic approval review rejected both `Remove-Item` and `rmdir` with `blocked by policy`, despite explicit owner authorization.
The obsolete bulk was not removed. This is a tool restriction, not a remaining process lock.
The owner can run the prepared script from the repository root:

```powershell
.\dev\tools\clear_old_runtime.ps1
```

It checks target paths, rejects directory links, stops repository Python and Chromium processes,
and removes old `.runtime` children while preserving `actor-immersion-20261103`.
The script has not been executed because deletion was rejected.
