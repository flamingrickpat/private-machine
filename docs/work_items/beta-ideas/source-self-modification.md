# Source self-modification

## The idea

A19 lets reflection change goals, beliefs, attention, and workflow configuration.
Beta can let reflection change its own source.
The companion reads its code, forms a hypothesis, edits a file, tests the result, and
keeps or reverses the change.

The user allows this, in the private deployment and inside the sandbox.
The change must be versioned, reversible, and evidence-backed.

## Rough scope

- A full test run before and after the change.
- A versioned change with a written explanation.
- A reversal command.
- A limit on which files the companion may change.
- A guard that keeps the evidence store and the checkpoint untouched.
- A human review gate for a change to the runtime or the model configuration.

## Open questions

- Does the companion edit the runtime, the workflows, or both?
- Who decides that a change is safe?
- What stops a bad change from breaking the next startup?
- How do we compare the changed behavior with the baseline?
- Do we keep the change when the tests pass but the behavior feels worse?

## Notes

Read the A19 reflection contract and the generalization ladder.
A source change is a stronger form of the same monitor-assess-guide loop.
It follows the [sandboxed host access](sandboxed-host-access.md) boundary.