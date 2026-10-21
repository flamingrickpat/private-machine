---
name: dialogue_assistant
driver: {kind: python}
human_resolver: {mode: forbid}
state_policy: {mutation_model: append_only}
roles:
  assistant:
    skill: workflows/dialogue_assistant/steps.py
    tools:
      - assistant_complete
      - public_complete
      - minecraft_observe
      - ghost_get
      - ghost_search
      - ghost_gwt
      - managed_start
      - managed_status
      - managed_list
      - managed_update
      - managed_terminate
      - managed_answer
    result_contract:
      type: json
      schema:
        status: {enum: [done]}
        summary: string
phases:
  - name: assistant
    kind: role
    role: assistant
    on_status: {done: stop}
    on_invalid: {action: stop}
---

# Assistant dialogue

One tool-capable Assistant exchange. The role builds a separate Assistant
message sequence, runs the model's tool calls, appends their real results, and
returns one final assistant utterance. A tool receipt, not the model's memory,
supports any claim that a management operation happened. It writes no memory;
the caller decides what to do with an accepted reply.
