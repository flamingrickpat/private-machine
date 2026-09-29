---
name: self_management
driver: {kind: python}
human_resolver: {mode: forbid}
state_policy: {mutation_model: append_only}
roles:
  manage:
    skill: workflows/self_management/steps.py
    tools:
      - managed_start
      - managed_status
      - managed_list
      - managed_update
      - managed_terminate
      - managed_answer
    result_contract:
      type: json
      schema:
        status: {enum: [done]}
        summary: string
phases:
  - name: manage
    kind: role
    role: manage
    on_status: {done: stop}
    on_invalid: {action: stop}
---

# Self management

One deterministic managed-run operation: start, status, list, update,
terminate, or answer. It contains no prompts and requests no model call. The
same operation functions are exposed to Assistant dialogue through MCP.
