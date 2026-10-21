---
name: inner_thought
driver: {kind: python}
human_resolver: {mode: forbid}
state_policy: {mutation_model: append_only}
roles:
  work:
    skill: workflows/inner_thought/steps.py
    tools: [run_stage, retrieve_memory]
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

# The private thought loop

Run one private cycle: evaluate the deterministic triggers, perform the chosen
thought operation, and integrate its consequences. A cycle with no trigger
performs nothing. Accepted work is committed by the host into the inner GWT
shard and published to the main loop as a self-origin event.
