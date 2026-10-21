---
name: focus_management
driver: {kind: python}
human_resolver: {mode: forbid}
state_policy: {mutation_model: append_only}
roles:
  work:
    skill: workflows/focus_management/steps.py
    tools: [mental_memory, mental_goals, mental_inspected]
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

# Deliberate mental work

Run a private pm-coder session with scoped memory and goal reads.
Return advice for main appraisal. The session cannot commit a workspace.
