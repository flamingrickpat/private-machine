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

The role builds three messages: the writer instruction, derived context, and
the verbatim transcript with an open companion prefill. Empty output permits
one recovery instruction after the full context. Each valid draft receives one
review. A false task result receives correction feedback until a draft passes.
After ten follow-ups, generation starts again from the original context.
Retries and resets print to stdout. The review returns a judgment, a reason,
and advisory findings. Only a false task result blocks the draft. The model
receives no tools. The workflow writes no memory. The caller handles the
accepted draft or the declared model-output error.
