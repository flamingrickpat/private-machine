# Follow one exchange

## Summary

A user input becomes accepted memory and a correlated bus envelope.
Cognition produces a committed GWT and typed actions.
Dialogue produces accepted reply memory and a public reply.

Parent: [manual](../index.md).
Context: [cognition](../subsystems/cognition.md).

## Path

```text
submit -> input Feature -> user.input envelope
       -> cognitive mailbox -> twelve typed phases
       -> GWT turn and cycle state commit
       -> typed action envelope -> dialogue mailbox
       -> reply Feature commit -> dialogue.reply
       -> user output and console notification
```

## Important steps

[`workflows.public_speech.user_input.UserInputProxy.submit`][] stores a dialogue feature with the current user as its source.
It assigns a correlation ID and publishes the input.
The bus gives each matching subscriber a detached JSON copy.
Each module processes its own mailbox batch.

[`workflows.main_cognition.host.MainCogLoop`][] invokes the real parent workflow.
[`workflows.main_cognition.steps.run`][] runs the ordered phases.
The parent commits accepted state before action dispatch.
The selected dialogue module commits its reply before public output.
The Story request retains the accepted action intent beside the current user message.
Its selected memory texts preserve attribution and original evidence IDs.
Read [experienced memory](../subsystems/experienced-memory.md) for the retrieval and commit boundaries.

## Evidence

The network view shows actual recipients, topics, correlation IDs, and causation IDs.
Workflow events show exact requests and reports.
Model events show actual prompt messages and completion results.
The memory snapshot shows accepted records and GWT turns.

## Failures

A model call can fail before a reply exists.
The application preserves its first thread error and requests shutdown.
A checkpoint error prevents acceptance of that batch.
The console retains the trace and worker log for analysis.
Restart restores accepted memory, not incomplete proposals.

## Go deeper

Read [Ghost commits](../ghost.md), [workflow runtime](../runtime.md), and [evidence and quality](../concepts/evidence.md).
