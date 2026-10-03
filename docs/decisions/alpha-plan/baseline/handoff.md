# Alpha session handoff contract

## Summary

Each later session identifies its source snapshot, changed contracts, commands, results, and remaining limits.
Behavioral claims require real endpoint receipts and reviewed acceptance facts.
Source review establishes implementation evidence only.
The frozen A01 matrix remains the starting comparison.

Parent: [A01 baseline](index.md).
Procedure: [local commands](commands.md).
Results: [baseline receipts](receipts.md).

## Before implementation

1. Read the selected work item and its linked design.
2. Inspect repository instructions and all relevant working trees.
3. Record full revisions, dirty files, lock hashes, and installed package origins.
4. Record configuration, persona, model, and fixture identities.
5. Resolve the item's explicit dependencies against existing receipts.
6. If a required gate failed, record a concrete diagnosis before dependent work.

## Required session record

```yaml
item: A02
started_at: "UTC timestamp"
source:
  baseline_commit: "full Git revision"
  repositories: {}
  dirty_source_archive: "path and sha256, if applicable"
  dependency_locks: {}
configuration:
  model_config_sha256: "digest"
  persona_sha256: "digest"
  services: {}
  fixture:
    directory: "actual world or evidence directory"
    seed: "configured seed"
    initial_inventory: "receipt path"
    reset_command: "exact command"
changed_contracts: []
verification:
  - command: "exact command"
    result: "passed | failed | unavailable | not_run"
    exit_code: 0
    evidence: "path and sha256"
    diagnosis: "required for failed gates"
human_review:
  routing_labels: "reviewed | pending | not_applicable"
  immersion: "rating receipt or pending"
limits: []
next_item: A03
```

Credentials and full inherited memory stay outside the published handoff.
Model errors, logic errors, expected gameplay failures, and unavailable services remain distinct.
The report includes sample counts beside rates and keeps holdout exposure visible.

## A01 changed contracts

The application, Ghost, workflow engine, managed-run schema, and existing model choices retain their behavior.
A01 adds a local model configuration, source snapshot command, console smoke, real-server gate wrapper, and development-only LLM comparison.
The command output directories must be new descendants of `.runtime`.
The real-server reset requires `--reset-world` and rejects removal targets outside the selected server directory.
The routing fixtures record pending human review and separate development from holdout.

## Next item

A02 must verify typed Laya inference and expose the public pm-coder client.
The current service advertises `laya` on port 8082.
Service discovery is not an inference result.
Existing workflows remain usable without the decision service.
Future A04 comparisons need reviewed case facts and recorded input limits.
