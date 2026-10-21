# Felt-state projections and vocabulary

## Summary

A17 derives qualia channels from accepted mental state and attributed records.
The channels describe internal change without creating another state authority.
Reportable snapshots cite their actual drivers.
The private ontology connects recurring patterns to stable terms.
Public expression remains agentic.
This page defines the design contract behind the delivered behavior.

Parent: [cognitive architecture](cognition.md).
Owner item: [A17](work-items/a17.md).
Related: [private loop](private-loop.md), [private ontology](ontology.md), and [source selection](cognitive-sources.md).
Current schemas: [`pm.records.qualia_records.QualeChannelState`][] and [`pm.records.qualia_records.QualeRecord`][].
A17 delivered the channel projections in `workflows.inner_thought.felt_state`; see [the private thought loop](../../subsystems/private-loop.md#felt-state-channels) and the [A17 verification](../verification/a17-private-loop.md).

## Content, introspection, and expression

The selected source distinguishes content representation, introspective representation, and reportability or self-modulation.
The pm adaptation uses three corresponding boundaries:

1. Accepted state and attributed events supply the content.
2. A derived snapshot identifies a meaningful internal transition and its drivers.
3. Selected introspection and ontology context inform a private interpretation or public action.

A report is an interpretation of recorded state.
It does not create that state merely by describing it.
An expressive sentence does not establish a new sense or an unimplemented emotion.
These distinctions govern records, not a new public sentence filter.

## Channel definitions and temporal measurements

Each channel declares its identity, version, source fields, aggregation, units, and applicable reality.
Reuse existing mental-state ranges and record collections.
Do not create separate authoritative channel embeddings or a parallel memory database.

Include affect, relationship appraisal, goal conflict, and attentional engagement in the initial supported channels.
Demonstrate a meaningful transition for each delivered channel.
Additional channels need a distinct use and an inspectable driver.

Use deterministic measurements with recorded observation times:

```text
position = declared projection of accepted source state
velocity = (position - previous_position) / elapsed_seconds
acceleration = (velocity - previous_velocity) / elapsed_seconds
```

Define the sampling convention for nonuniform intervals.
Record projection units and derivative units.
Document initialization, absent sources, and nonpositive intervals.
Do not present initialized default zeros as measured absence of movement.
Preserve the preceding sample and projection version through checkpoints.

Separate an observed restart transition from executed private activity.
A long offline gap does not supply intermediate thought samples.
Any existing elapsed-time decay remains an explicit state update after restart.
It does not establish imagined background execution.

## Driver and reportability contract

Each notable snapshot retains source IDs, revisions, sampling interval, channel version, and accepted cycle references.
The driver evidence must establish the transition.
Co-moving channels alone do not establish a common cause.
A combined episode report can cite several channels with explicit shared drivers.

A reportable quale requires causal or attention evidence for its drivers.
A candidate whose drivers never entered accepted attention remains dormant.
The gate does not require a public utterance or an external action for every internal transition.
An accepted private appraisal or attention effect can establish the relevant consequence.

Revalidate source revisions before accepting the report.
Cancellation prevents a late commit.
An interrupted report leaves its accepted driver records intact.
Retry must not duplicate the same snapshot or vocabulary binding.
The memory store owns commits and checkpoint recovery.

## Stable personal vocabulary

Use the existing ontology term and binding records.
Bind a term to a sourced pattern or episode, not to a dramatic sentence.
Retain literal meaning, applicability, anchor examples, and conflicting examples.
For example, the same personal phrase can describe a recurring mixture of warmth and apprehension.
Its anchors must show that mixture rather than an invented physical sensation.

Reuse a suitable established term before proposing a synonym.
New recurring patterns can justify new terms.
Distinguish a provisional term from a stable term.
A changed definition must retain its old version and previous bindings.
A misleading or unsupported term remains an inspectable proposal.
It must not acquire an invented state mechanism.

Demonstrate reuse across different wording, distraction, and restart.
Demonstrate a distinct pattern that does not match the same term.
Term merging must preserve anchor meaning and original provenance.
General ontology discovery and a large offline terminology pipeline remain beta work.

## Expression and accepted memory

Keep metaphor, sarcasm, roleplay surface, simulations, and accidental unsupported text attributed to their source role.
Do not aggregate those utterances as observed channel values.
Their actual interpretation can still cause a sourced appraisal.
A metaphor can therefore matter emotionally without becoming a factual sensory observation.

Dialogue receives selected reportable state and terms as context.
It can speak naturally, retain privacy, or choose deliberate presentation under A09R.
The ontology remains advisory.
`false_task_result` remains the only hard speech block.
Raw thought transcripts and provider reasoning are not public outputs of this mechanism.

## Acceptance

Verify deterministic projection, elapsed-time measurements, source conflicts, and checkpoint recovery with foundation checks.
Use real inference for interpretations and vocabulary choices.
Trace actual accepted drivers through an internal consequence and a later relevant conversation.
Retain dormant candidates and failed interpretations.
Human review assesses whether recurring language feels coherent and personal.
It does not establish consciousness or replace source evidence.
