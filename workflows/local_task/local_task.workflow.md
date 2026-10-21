---
name: local_task
driver: {kind: python}
human_resolver: {mode: forbid}
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
  - name: work
    kind: role
    role: work
    on_status: {done: stop}
    on_invalid: {action: stop}
---

# Local task

One small real local workflow. It writes observations and a result into its
task folder and stops. It performs no model call and no world action. It exists
to exercise managed-run concurrency, status, updates, and cancellation.
