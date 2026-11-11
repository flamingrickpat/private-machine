---
name: interaction
driver: {kind: python}
human_resolver: {mode: forbid}
state_policy: {mutation_model: append_only}
failure_policy: {max_attempts_per_phase: 1}
roles:
  select:
    skill: workflows/interaction/steps.py
    tools: [llm_complete, embed_texts]
    result_contract:
      type: json
      schema:
        status: {enum: [done]}
        summary: string
phases:
  - name: select
    kind: role
    role: select
    on_status: {done: stop}
    on_invalid: {action: stop}
---

# Ordinary character interaction

Choose a local conversational move from accepted direction and current input.
Consequential choices return to main cognition. The existing speech workflow
owns public generation, review, and durable delivery.
