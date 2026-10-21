---
name: trigger
driver: {kind: python}
human_resolver: {mode: forbid}
state_policy: {mutation_model: append_only}
roles:
  work:
    skill: workflows/inner_thought/stages/trigger/steps.py
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

# Private trigger

Pick one fired trigger and one of its suggested thought operations, or choose
no operation. The stage never invents a trigger, an operation, or a candidate.
It never writes memory.
