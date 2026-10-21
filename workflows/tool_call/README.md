# tool_call

One managed workflow: the companion uses its own MCP tools for one goal, in one native pm-coder session.

Read the guide: [docs/workflows/tool-call.md](../../docs/workflows/tool-call.md).

| File | Content |
| --- | --- |
| `tool_call.workflow.md` | The manifest: one Python role |
| `host.py` | `build_workflow_spec` and the pinned source context |
| `steps.py` | The session, the tool images, and the terminal report |
| `schemas.py` | Context, request, agent assessment, evidence, and report contracts |
| `report.py` | Report files, the interrupted report, and the reply-time file check |
| `consolidation.py` | The outcome memory record with tool images |
| `catalogue.py` | The tool list for the route prompt |
| `resources/execute.md` | The work instructions of the session |

The configuration type is `pm.llm.mcp_config.ToolCallConfig`, in the `tool_call` section of the model config.
