---
name: local_task_question
driver: {kind: python}
human_resolver: {mode: external}
state_policy: {mutation_model: append_only}
roles:
  work:
    skill: workflows/local_task/steps.py
    result_contract:
      type: json
      schema:
        status: {enum: [done]}
        summary: string
phases:
  - name: ask
    kind: human
    question: "Which artifact should the local task produce?"
    next: work
  - name: work
    kind: role
    role: work
    on_status: {done: stop}
    on_invalid: {action: stop}
---

# Local task with a blocking question

The same local worker, but it suspends on one human question first. A
correlated answer resumes the run into the work phase.
