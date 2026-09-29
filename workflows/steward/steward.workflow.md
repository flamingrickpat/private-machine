---
name: steward
driver: {kind: python}
human_resolver: {mode: forbid}
state_policy: {mutation_model: append_only}
roles:
  steward:
    skill: workflows/steward/steps.py
    tools: [steward_status, steward_answer, steward_update, steward_terminate]
    result_contract:
      type: json
      schema:
        status: {enum: [done]}
        summary: string
phases:
  - name: steward
    kind: role
    role: steward
    on_status: {done: stop}
    on_invalid: {action: stop}
---

# Steward

One deterministic steward operation: status, answer, update, or terminate. It
stays dormant until explicitly activated. It reads committed progress and
returns the evidence it read; it never edits a file or a prompt.
