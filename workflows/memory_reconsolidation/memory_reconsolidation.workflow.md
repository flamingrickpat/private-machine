---
name: memory_reconsolidation
driver: {kind: python}
human_resolver: {mode: forbid}
failure_policy: {max_attempts_per_phase: 1}
state_policy: {mutation_model: append_only}
roles:
  reconsolidate:
    skill: workflows/memory_reconsolidation/steps.py
    tools: []
    result_contract:
      type: json
      schema:
        status: {enum: [done]}
        summary: string
phases:
  - name: reconsolidate
    kind: role
    role: reconsolidate
    on_status: {done: stop}
    on_invalid: {action: stop}
---

# Memory reconsolidation

The character can sort through earlier experiences during conversation or
sleep while the work runs. The worker prepares complete memory units.
Main cognition publishes the ready units when it commits a cycle. During
memory sleep, a waking event or the end of this task publishes the units
and saves the character. The report counts prepared units awaiting or
already past that boundary.
