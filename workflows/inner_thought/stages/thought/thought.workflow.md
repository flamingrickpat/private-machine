---
name: thought
driver: {kind: python}
human_resolver: {mode: forbid}
state_policy: {mutation_model: append_only}
roles:
  work:
    skill: workflows/inner_thought/stages/thought/steps.py
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

# Private thought

Perform the chosen thought operation on the private candidates. Keep the
observed evidence separate from the interpretation, label the reality of the
thought, record the alternatives that were weighed, and end with an explicit
continuation decision. The stage never writes memory.
