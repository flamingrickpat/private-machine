---
name: main_cognition
driver: {kind: python}
human_resolver: {mode: forbid}
state_policy: {mutation_model: append_only}
roles:
  work:
    skill: workflows/main_cognition/steps.py
    tools: [run_stage]
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

# Main cognitive cycle

Runs the eight ordered children (triage, appraisal, context, codelets,
attention, goals, workspace, action selection) over one request and one shared
proposal batch. Codelets nominate; attention scores the nominations and runs
the competition. The parent assigns action IDs and releases typed dispatch
records. The module, not this role, commits the GWT, the attention records,
and publishes dispatches.
