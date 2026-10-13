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

Score the codelet nominations of this cycle and choose the winners. The model
gives four integer scores per nomination: urgency, novelty, goal relevance,
and reportability. Python then runs the inspectable competition in
`workflows/main_cognition/attention_competition.py`: reserved places,
capacity, a minimum score, and a danger bonus for urgency. Every nomination
receives a status and a reason. Target ids come only from supplied
candidates. It never writes memory.
