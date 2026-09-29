---
name: triage
driver: {kind: python}
human_resolver: {mode: forbid}
state_policy: {mutation_model: append_only}
roles:
  work:
    skill: workflows/main_cognition/stages/triage/steps.py
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

# Triage

Classify each incoming stimulus by relevance, urgency, and input kind. This
stage decides how much cognitive effort the rest of the cycle spends. It never
writes memory.
