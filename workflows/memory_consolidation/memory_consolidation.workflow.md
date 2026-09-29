---
name: memory_consolidation
driver: {kind: python}
human_resolver: {mode: forbid}
state_policy: {mutation_model: append_only}
roles:
  consolidate:
    skill: workflows/memory_consolidation/steps.py
    tools: [consolidated_artifacts, list_consolidations]
    result_contract:
      type: json
      schema:
        status: {enum: [done]}
        summary: string
phases:
  - name: consolidate
    kind: role
    role: consolidate
    on_status: {done: stop}
    on_invalid: {action: stop}
---

# Memory consolidation

One evidence-backed consolidation of a run's real text artifacts. It proposes
outcome, observation, and experience memory, records the inventory hash in its
receipt, and reuses that receipt for a repeated request. It archives nothing.
