---
name: appraisal
driver: {kind: python}
human_resolver: {mode: forbid}
state_policy: {mutation_model: append_only}
roles:
  work:
    skill: workflows/main_cognition/stages/appraisal/steps.py
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

# Appraisal

Appraise the triaged situation from the companion's point of view. Produce a
one-sentence feeling, signed emotion and need deltas, reasoning, and evidence
ids. This stage never writes memory.
