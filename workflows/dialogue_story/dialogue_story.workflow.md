---
name: dialogue_story
driver: {kind: python}
human_resolver: {mode: forbid}
state_policy: {mutation_model: append_only}
roles:
  story:
    skill: workflows/dialogue_story/steps.py
    tools: [story_complete]
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

One standalone Story generation. The role builds the five-message sequence,
continues the companion prefill, cleans the continuation, and applies the
private-ontology boundary. The workflow requests no tool call from the
generation model. It writes no memory; the caller decides what to do with an
accepted candidate.
