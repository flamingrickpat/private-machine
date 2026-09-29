---
name: attention
driver: {kind: python}
human_resolver: {mode: forbid}
state_policy: {mutation_model: append_only}
roles:
  work:
    skill: workflows/main_cognition/stages/attention/steps.py
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

# Attention

Nominate and rank the candidate knoxels that matter for this cycle. Weigh
urgency, novelty, goal relevance, and reportability against the appraisal and
retrieved context. Dangerous or goal-blocking items outrank benign detail, and
repetition does not by itself defeat salience. Select only ids that were
supplied as candidates. It never writes memory.
