---
name: minecraft_project
driver: {kind: python}
human_resolver: {mode: external}
state_policy: {mutation_model: append_only}
roles:
  planner:
    skill: workflows/minecraft_tasks/steps.py
    tools: [minecraft_call, minecraft_endpoint, minecraft_reason, managed_instructions]
    result_contract:
      schema:
        status: {enum: [planned, survey, invalid, waiting, blocked]}
        summary: string
  surveyor:
    skill: workflows/minecraft_tasks/steps.py
    tools: [minecraft_call, minecraft_endpoint, managed_instructions]
    result_contract:
      schema:
        status: {enum: [observed, replan]}
        summary: string
  reviewer:
    skill: workflows/minecraft_tasks/steps.py
    tools: [minecraft_call, minecraft_endpoint, minecraft_reason, managed_instructions]
    result_contract:
      schema:
        status: {enum: [accepted, inspect, refine, revise, waiting, blocked]}
        summary: string
  selector:
    skill: workflows/minecraft_tasks/steps.py
    tools: [minecraft_call, minecraft_endpoint, managed_instructions]
    result_contract:
      schema:
        status: {enum: [selected, aggregate, complete, replan, waiting]}
        summary: string
  specialist:
    skill: workflows/minecraft_tasks/steps.py
    tools: [minecraft_call, minecraft_plan, minecraft_list_tools, minecraft_tool_schemas, minecraft_endpoint, minecraft_lease, managed_instructions]
    result_contract:
      schema:
        status: {enum: [candidate, progress, replan, waiting]}
        summary: string
  verifier:
    skill: workflows/minecraft_tasks/steps.py
    tools: [minecraft_call, minecraft_endpoint, minecraft_reason, managed_instructions]
    result_contract:
      schema:
        status: {enum: [accepted, replan, waiting, blocked]}
        summary: string
  final_verifier:
    skill: workflows/minecraft_tasks/steps.py
    tools: [minecraft_call, minecraft_endpoint, minecraft_reason, managed_instructions]
    result_contract:
      schema:
        status: {enum: [done, replan, waiting, blocked]}
        summary: string
phases:
  - name: plan
    kind: role
    role: planner
    on_status: {planned: review, survey: survey, invalid: replan, waiting: wait, blocked: stop}
    on_invalid: {action: stop}
  - name: survey
    kind: role
    role: surveyor
    on_status: {observed: replan, replan: replan}
    on_invalid: {action: stop}
  - name: review
    kind: role
    role: reviewer
    on_status: {accepted: select, inspect: finish, refine: replan, revise: replan, waiting: wait, blocked: stop}
    on_invalid: {action: stop}
  - name: select
    kind: role
    role: selector
    on_status: {selected: act, aggregate: verify, complete: finish, replan: replan, waiting: wait}
    on_invalid: {action: stop}
  - name: act
    kind: role
    role: specialist
    on_status: {candidate: verify, progress: select, replan: replan, waiting: wait}
    on_invalid: {action: stop}
  - name: verify
    kind: role
    role: verifier
    on_status: {accepted: select, replan: replan, waiting: wait, blocked: stop}
    on_invalid: {action: stop}
  - name: replan
    kind: role
    role: planner
    on_status: {planned: review, survey: survey, invalid: replan, waiting: wait, blocked: stop}
    on_invalid: {action: stop}
  - name: wait
    kind: human
    question_from_result: summary
    next: plan
  - name: finish
    kind: role
    role: final_verifier
    on_status: {done: stop, replan: replan, waiting: wait, blocked: stop}
    on_invalid: {action: stop}
---

# Durable Minecraft project

The planner retains a generic goal/story/leaf tree. An independent review can
refine one oversized node. The selector releases prerequisites only after
independent verification. Each selected leaf gets a fresh native actor child.
The verifier has read-only world tools. Waiting retains all task evidence.
Replanning preserves world effects and unaffected accepted work.
