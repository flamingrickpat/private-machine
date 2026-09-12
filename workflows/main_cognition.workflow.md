---
name: main_cognition
driver: {kind: python}
human_resolver: {mode: forbid}
state_policy: {mutation_model: append_only}
roles:
  work:
    skill: workflows/roles/main_cognition.py
    result_contract:
      type: json
      schema:
        status: {enum: [done]}
        summary: string
phases:
  - name: work
    kind: role
    role: work
    on_status: {done: stop}
    on_invalid: {action: stop}
---
