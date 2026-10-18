---
name: main_cognition
driver: {kind: python}
human_resolver: {mode: forbid}
state_policy: {mutation_model: append_only}
failure_policy: {max_attempts_per_phase: 1}
roles:
  phase:
    skill: workflows/main_cognition/steps.py
    tools: [llm_complete, embed_texts, codelet_agent]
    result_contract:
      type: json
      schema:
        status: {enum: [done]}
        summary: string
phases:
  - name: intake
    kind: role
    role: phase
    on_status: {done: triage}
    on_invalid: {action: stop}
  - name: triage
    kind: role
    role: phase
    on_status: {done: appraise}
    on_invalid: {action: stop}
  - name: appraise
    kind: role
    role: phase
    on_status: {done: retrieve}
    on_invalid: {action: stop}
  - name: retrieve
    kind: role
    role: phase
    on_status: {done: codelets}
    on_invalid: {action: stop}
  - name: codelets
    kind: role
    role: phase
    on_status: {done: structures}
    on_invalid: {action: stop}
  - name: structures
    kind: role
    role: phase
    on_status: {done: attention}
    on_invalid: {action: stop}
  - name: attention
    kind: role
    role: phase
    on_status: {done: experience}
    on_invalid: {action: stop}
  - name: experience
    kind: role
    role: phase
    on_status: {done: goals}
    on_invalid: {action: stop}
  - name: goals
    kind: role
    role: phase
    on_status: {done: decide}
    on_invalid: {action: stop}
  - name: decide
    kind: role
    role: phase
    on_status: {done: route}
    on_invalid: {action: stop}
  - name: route
    kind: role
    role: phase
    on_status: {done: commit}
    on_invalid: {action: stop}
  - name: commit
    kind: role
    role: phase
    on_status: {done: stop}
    on_invalid: {action: stop}
---

# Main cognitive cycle

One cycle runs as one sequence of typed phases. All phases share one
`CycleWork` object (see `workflows/main_cognition/cycle_work.py`). Each phase
validates it at entry and writes it back. Every model call of the cycle is
an unconscious story and choice simulator: it starts with the shared story
prefix and ends with its instruction.

- intake groups the stimuli by entity, workflow, or self.
- triage rates each group; appraise checks the open expectations and
  appraises the events.
- retrieve and the codelets fill the current situational model (CSM). The
  structure builders bind its items, and attention moves them into the
  workspace shards.
- experience writes the subjective experience into the self shard tail.
  goals reads it.
- decide makes zero or one in-character decision for each significant
  group. route maps each decision to an action, checks conflicts, and
  records the expected reactions.
- commit writes the final report. The module, not this workflow, commits
  the write batch and publishes the dispatches.

A role with a Markdown skill runs as a pm-coder agent, so a later item can
insert an agent phase between two phases.

Each phase runs once per cycle. Invalid model output gets repair turns inside
the phase. Exhausted repairs stop the run, and the module retries the whole
cycle from fresh state. Other errors stop the application.
