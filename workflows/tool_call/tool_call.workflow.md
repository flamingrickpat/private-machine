---
name: tool_call
driver: {kind: python}
human_resolver: {mode: forbid}
state_policy: {mutation_model: append_only}
roles:
  execute:
    skill: workflows/tool_call/steps.py
    tools: [tool_call_settings, managed_instructions]
    result_contract:
      type: json
      schema:
        status: {enum: [done]}
        summary: string
phases:
  - name: execute
    kind: role
    role: execute
    on_status: {done: stop}
    on_invalid: {action: stop}
---

# One tool goal

The Python role opens one sustained pm-coder session for one companion goal.
The goal is the companion's decision text, unchanged. The context comes from the cycle that chose the action.
The session gets the persona, that context, and every configured MCP server.
The host keeps tool images, checks local artifact bytes, and writes `pm.tool-call-report.v1`.
The report records success claims, failed attempts, blockers, or uncertainty.
The supervisor's terminal hook consolidates the report. The reply path reads it.
See `docs/workflows/tool-call.md`.
