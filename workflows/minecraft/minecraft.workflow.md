---
name: minecraft
driver: {kind: python}
human_resolver: {mode: forbid}
state_policy: {mutation_model: append_only}
roles:
  body:
    skill: workflows/minecraft/steps.py
    tools: [minecraft_call, minecraft_plan, minecraft_list_tools, minecraft_tool_schemas, minecraft_endpoint, minecraft_lease, managed_instructions]
    result_contract:
      type: json
      schema:
        status: {enum: [done]}
        summary: string
phases:
  - name: body
    kind: role
    role: body
    on_status: {done: stop}
    on_invalid: {action: stop}
---

# Minecraft body

The default role runs one sustained pm-coder specialist through a scoped MCP
gateway. The session retains one body lease and checks instructions before
each tool call. It writes an actor journal, screenshots, a world record, and
a task handoff. The configured text model cannot interpret screenshot pixels.
Explicit W8 actions retain the narrow boundary-check path. The MCP must be running.
