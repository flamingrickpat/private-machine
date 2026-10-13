---
name: codelets
driver: {kind: python}
human_resolver: {mode: forbid}
state_policy: {mutation_model: append_only}
roles:
  work:
    skill: workflows/main_cognition/stages/codelets/steps.py
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

# Codelets

Schedule a relevant subset of the codelet catalogue and collect their typed
proposals for attention competition. Pure codelets run in Python. Each LLM
codelet makes one short model call. A proposal names its origin, its reality
label, and the source revisions it read. It never writes memory or starts an
action.
