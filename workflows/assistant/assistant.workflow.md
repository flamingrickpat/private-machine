---
name: assistant
driver: {kind: python}
human_resolver: {mode: forbid}
state_policy: {mutation_model: append_only}
roles:
  execute:
    skill: workflows/assistant/steps.py
    result_contract:
      type: json
      schema:
        status: {enum: [done]}
        summary: string
phases:
  - name: execute
    kind: role
    role: execute
    on_status: {done: stop}
    on_invalid: {action: stop}
---

# Private assistant session
