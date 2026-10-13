---
name: dialogue_story
driver: {kind: python}
human_resolver: {mode: forbid}
state_policy: {mutation_model: append_only}
roles:
  story:
    skill: workflows/dialogue_story/steps.py
    tools: [story_complete, public_complete]
    result_contract:
      type: json
      schema:
        status: {enum: [done]}
        summary: string
phases:
  - name: story
    kind: role
    role: story
    on_status: {done: stop}
    on_invalid: {action: stop}
---

# Story dialogue

One Story generation. The role builds the three-message sequence (story
writer, derived context, verbatim transcript with the open companion prefill),
continues the prefill, cleans the continuation, and reviews it once with the
agentic persona-aware review. The review returns a judgment, a reason, and
advisory findings. Only a false task result blocks the reply; there is no
regeneration loop. The workflow requests no tool call from the generation
model. It writes no memory; the caller decides what to do with an accepted
candidate.
