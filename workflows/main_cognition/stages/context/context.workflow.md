---
name: context
driver: {kind: python}
human_resolver: {mode: forbid}
state_policy: {mutation_model: append_only}
roles:
  work:
    skill: workflows/main_cognition/stages/context/steps.py
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

# Context

Assemble the relevant context for the current cycle. The stage selects only
from the supplied memory candidates, adds relevant active goals and capability
facts, and reports the selected candidate IDs. It never fabricates a memory or
a source ID and never writes memory.
