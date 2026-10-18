---
name: memory_reconsolidation
driver: {kind: python}
human_resolver: {mode: forbid}
failure_policy: {max_attempts_per_phase: 1}
state_policy: {mutation_model: append_only}
roles:
  reconsolidate:
    skill: workflows/memory_reconsolidation/steps.py
    tools: [reconsolidate_memory]
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

The companion sorts through its memories: the character decided to "sleep on
it" or to go through what happened. The Python role calls
`pm.memory.reconsolidation.reconsolidate` on the live memory store. It builds
topical and temporal summaries, extracts facts, and refines the narratives.
Each finished group commits through the normal store path, so the live loop
sees the new memory at its next cycle. The report lists the created record ids.
