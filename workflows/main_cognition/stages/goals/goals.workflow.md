---
name: goals
driver: {kind: python}
human_resolver: {mode: forbid}
state_policy: {mutation_model: append_only}
roles:
  work:
    skill: workflows/main_cognition/stages/goals/steps.py
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

# Goals

Review the current goals against the appraisal and the assembled context. This
stage decides whether each goal is retained, added, completed, or
reprioritized. It proposes justified goal changes only, never edits code,
prompts, or hidden files, and never writes memory.
