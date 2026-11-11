---
name: input_triage
driver: {kind: python}
human_resolver: {mode: forbid}
state_policy: {mutation_model: append_only}
failure_policy: {max_attempts_per_phase: 1}
roles:
  classify:
    skill: workflows/input_triage/steps.py
    tools: [llm_complete]
    result_contract:
      type: json
      schema:
        status: {enum: [done]}
        summary: string
phases:
  - name: classify
    kind: role
    role: classify
    on_status: {done: stop}
    on_invalid: {action: stop}
---

# Early input triage

Classify committed partner input before the full main cycle. The host accepts
the assessment and coordinates the critical publication barrier. This workflow
does not appraise, choose actions, change intentions, or publish speech.
