---
name: action_selection
driver: {kind: python}
human_resolver: {mode: forbid}
state_policy: {mutation_model: append_only}
roles:
  work:
    skill: workflows/main_cognition/stages/action_selection/steps.py
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

# Action Selection

Turn the current workspace frame, the active goals, the available tool
workflows, and the companion's real capabilities into zero or more action
intents for this cycle. Each intent states the goal of an action, never a
finished companion reply. The stage may choose no action at all. It never
writes memory and it never invents a workflow or a capability.
