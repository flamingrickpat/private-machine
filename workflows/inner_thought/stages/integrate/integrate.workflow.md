---
name: integrate
driver: {kind: python}
human_resolver: {mode: forbid}
state_policy: {mutation_model: append_only}
roles:
  work:
    skill: workflows/inner_thought/stages/integrate/steps.py
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

# Private integration

Turn one performed thought into accepted consequences: goal lifecycle
changes, one expectation lifecycle step, an optional focus intention for the
next main cycle, and a disclosure choice. Every consequence cites its
evidence. The stage never writes memory; the host commits.
