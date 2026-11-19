---
name: note_task
driver: {kind: python}
human_resolver: {mode: forbid}
failure_policy: {max_attempts_per_phase: 1}
roles:
  note:
    skill: workflows/note_task/steps.py
    tools: []
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

# Exact note write

The configured model selects content when the request has no accepted note specification.
The Python role writes the accepted text directly as exact UTF-8 bytes in the task outbox.
The host compares the actual file bytes with the requested UTF-8 bytes.
The final report records the instruction, expected bytes, digest, and verified result.
