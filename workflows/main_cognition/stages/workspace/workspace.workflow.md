---
name: workspace
driver: {kind: python}
human_resolver: {mode: forbid}
state_policy: {mutation_model: append_only}
roles:
  work:
    skill: workflows/main_cognition/stages/workspace/steps.py
    tools: [llm_complete]
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

# Workspace

Assemble one coherent attention frame from the selected attention candidates and
the current goals. Choose the focus ids, the control mode, and a brief
first-person subjective experience. Focus only on ids that were supplied and,
when attention made a selection, only on selected ids. It never writes memory.
