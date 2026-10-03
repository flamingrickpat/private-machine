---
name: minecraft
driver: {kind: python}
human_resolver: {mode: forbid}
state_policy: {mutation_model: append_only}
roles:
  body:
    skill: workflows/minecraft/steps.py
    tools: [minecraft_call, minecraft_plan, minecraft_list_tools, minecraft_tool_schemas, managed_instructions]
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

One bounded Minecraft body run. The role observes the real world through the
`minecraft_call` MCP tool, chooses one allowed action, executes it, and writes
real evidence: text observations, a structured `outbox/world.yaml`, screenshot
files, and a final report. Screenshot pixels become evidence files. They do
not enter the planning prompt or embedding text. The MCP must already be running.
