---
name: note_task
driver: {kind: python}
human_resolver: {mode: forbid}
failure_policy: {max_attempts_per_phase: 1}
roles:
  note:
    skill: workflows/note_task/steps.py
    tools: [note_spec]
    result_contract:
      type: json
      schema:
        status: {enum: [done]}
        summary: string
phases:
  - name: note
    kind: role
    role: note
    on_status: {done: stop}
    on_invalid: {action: stop}
---

# Agent-backed note

The Python role leases the configured chat connection for one pm-coder session.
The agent writes the requested note in the task's outbox.
The host compares the actual file bytes with the requested UTF-8 bytes.
The final report records the instruction, expected bytes, digest, and verified result.
