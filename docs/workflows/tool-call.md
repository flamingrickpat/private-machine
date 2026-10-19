# Use MCP tools for one goal

Parent: [managed work](../subsystems/managed-work.md).
Related: [decisions and routing](../subsystems/decisions-and-routing.md), [public publication](../subsystems/public-publication.md), and [images in records](../subsystems/multimodal-records.md).

## Summary

The `tool_call` workflow gives the companion its own MCP tools.
Main cognition decides in character, for example "make him that skiing cat picture".
Routing sends that decision to `tool_call`.
One native pm-coder session then uses every configured MCP server until the goal ends.
The host keeps a terminal report and any tool images.
Consolidation turns the report into a memory record.
A `ToolCallAndReply` action then gets a reply in the companion's own voice.

API: [`workflows.tool_call.host.build_workflow_spec`][], [`workflows.tool_call.steps.run`][], and [`pm.llm.mcp_config.ToolCallConfig`][].

## Configure the MCP servers

The MCP servers are in the `tool_call` section of the model configuration file.
That is the file that you give with `--config`, in the terminal mode, with `--home`, and in the console.
Each setup or character home can use its own file, so each companion gets its own tools.
The character home folder holds no MCP file.

```yaml
tool_call:
  timeout_seconds: 1800        # deadline of one session, in seconds
  gwt_turn_limit: 8            # recent GWT turns, up to the source cycle
  mcp_servers:
    - name: image_gen          # unique; also the tool prefix in the session
      url: http://127.0.0.1:9877/mcp
      headers:
        Authorization: Bearer ${IMAGE_MCP_TOKEN}
      request_timeout_seconds: 600
```

| Field | Default | Meaning |
| --- | --- | --- |
| `model_connection` | `Default` | The chat connection of the session |
| `timeout_seconds` | `1800` | The session deadline, in seconds |
| `gwt_turn_limit` | `8` | The number of recent GWT turns in the session context |
| `mcp_servers[].name` | required | A unique name. It is the tool prefix, for example `image_gen_paint` |
| `mcp_servers[].url` | required | An HTTP or HTTPS MCP endpoint |
| `mcp_servers[].headers` | `{}` | HTTP headers. `${NAME}` refers to an environment variable |
| `mcp_servers[].request_timeout_seconds` | `600` | The deadline of one tool request, in seconds |

A header variable comes from the process environment first and from the ignored `.env` file second.
If a variable has no value, the application start stops with `McpConfigError`.
No request goes out without the header.

Without servers, the application does not register `tool_call`.
Routing then never offers it, so the companion cannot claim a tool that it does not have.
Only HTTP MCP servers are supported. A stdio MCP needs an HTTP adapter.

## How the companion chooses a tool

1. The decide phase writes the decision. Its prompt names no tool and no workflow.
2. The route phase reads the workflow list. The `tool_call` description lists each server and its tools.
   [`ToolCatalogue`][workflows.tool_call.catalogue.ToolCatalogue] discovers the tools with a 5-second deadline and keeps a successful list.
   A server that does not answer is shown as "unavailable now". It is asked again after 60 seconds.
3. The route answer is `ToolCallAndReply` (speak about the result) or `ToolCall` (private result), with `workflow_name: tool_call`.
   It can add `arguments.tool_goal`, one short concrete goal.
4. The dispatch removes routed `instructions`. Self-management starts the run with the decision text, unchanged, as its instruction.

## Pinned source context

Self-management stores the action record id in the run arguments (`_public_action_schema_id`).
That record holds the frozen action request of the cycle that chose the action.
At launch, [`capture_context`][workflows.tool_call.host.capture_context] reads that record through the `dispatch_context` hook of the specification.
The context then holds the decision, its stimuli, its correlation, the cycle tick, the feeling, and the workspace shard story of that cycle.
The GWT turns end at that cycle.
A run without a source action, for example an operator start, gets the newest context at launch.
Its `context_source` is `launch`, never `dispatch`.

## The session

The session prompt starts with the companion: the character card, its decision in its own words, and what led to it.
These parts are story-framed, because the session is the companion's own work.
The tool rules are plain work instructions in `workflows/tool_call/resources/execute.md`.

One model lease and one native session last for the whole goal.
The session sees every configured server for the whole time.
It can call tools, look at results, and try again.
The session writes notes to `scratch/task-card.md`, `scratch/observations.md`, and `scratch/result.md`.
It returns one JSON assessment: outcome, summary, artifacts, observations, its own quality judgment, and uncertainty.
The MCP file of the session holds resolved headers. It is deleted when the session ends.

## Report

After the session, the host writes `outbox/report.yaml` with the schema `pm.tool-call-report.v1` ([`ToolCallReport`][workflows.tool_call.schemas.ToolCallReport]).

| Field | Meaning |
| --- | --- |
| `outcome` | `succeeded`, `failed`, `blocked`, `uncertain`, `cancelled`, or `interrupted` |
| `accepted` | The agent reported success, and every claimed local file passed host inspection |
| `verified` | Always false. No generic verifier exists for a tool goal |
| `agent_outcome` | The agent's own assessment, unchanged |
| `artifacts` | Each claimed file or URI, with host evidence: size and SHA-256 of local bytes |
| `tool_calls` | The MCP calls in the session record |
| `tool_images` | The images that tools returned, kept as `outbox/tool-image-<digest>.<ext>` |
| `action_id`, `action_schema_id`, `source_tick_id`, `correlation_id`, `stimulus_ids` | The links to the source action |

The supervisor maps `accepted: false` to a failed receipt. The report keeps the more specific outcome.
Every terminal state goes through the terminal hook and consolidation.
A cancelled session writes a `cancelled` report and then stops.
A run that a restart finds `interrupted` gets an `interrupted` report from [`retain_interrupted_report`][workflows.tool_call.report.retain_interrupted_report].

## Images

A tool can return an image in its MCP result.
The host reads the native session record (`messages.json`) after the session and keeps each image once as an outbox file.
A tool can also write a file into the outbox, when the agent gives it a host path.

[`tool_call_outcome`][workflows.tool_call.consolidation.tool_call_outcome] stores these images on the outcome record through [`pm.records.media`](../subsystems/multimodal-records.md).
Each image has `provenance.source: tool`, the server, the tool, the run id, and the artifact path.
The record content ends with the `[image:<id>]` tags, so the companion can see the images later.
The same bytes from a tool result and a saved file are one image.

## Memory and reply

[`Consolidator`][workflows.memory_consolidation.artifact_consolidation.Consolidator] recognizes the report schema.
It creates one outcome record with the outcome, the summary, the agent's own judgment, the uncertainty, the tools used, the full report in `metadata.tool_call_report`, and the source action links.
The consolidation key is the run and the digests of its files.
A repeated terminal hook or a restart therefore creates no second record.

The terminal event carries the report summary as its detail, so a private `ToolCall` result reaches cognition in words.
For `ToolCallAndReply`, [`ReplyRecovery`][workflows.public_speech.reply_recovery.ReplyRecovery] builds the reply request.
[`reply_evidence`][workflows.tool_call.report.reply_evidence] reads the report again and checks the bytes of every reported file.
The reply can name a local file truthfully when its bytes did not change.
`task_verified` stays false: unchanged bytes do not prove the content or the quality of an image.

## Recovery and external effects

Automatic resume is off (`resumable: False`).
A restart does not run an interrupted goal again, because an MCP call can have a real effect, such as a paid image.
After a timeout or a cancellation, inspect the remote result before a new attempt.
The session sees instruction updates only at its start and at its end.
A changed instruction during the session makes the result `uncertain`.

## Test with the demo server

`dev/tools/demo_mcp.py` is a real MCP server with two tools: `roll_dice(sides)` and `make_test_image(color, save_path)`.
`dev/tools/tool_call_check.py` runs the workflow once through the real supervisor and consolidator, without a cognition cycle.

```powershell
uv run --locked python dev/tools/demo_mcp.py --port 9881
$env:PYTHONPATH = "."
uv run --locked python dev/tools/tool_call_check.py --config config.alpha.openrouter.yaml `
    --mcp demo=http://127.0.0.1:9881/mcp --instruction "roll a 20-sided die and make a red test image"
```

The check writes `.runtime/toolcall-test/run-<time>/summary.json` with the receipt, the report, and the records.
The demo server logs each real call to `.runtime/toolcall-test/demo-mcp-calls.jsonl`.

## Limits

- A native compaction replaces older messages with a summary. Images in those messages are then not in the session record. Ask tools to save files into the outbox for long sessions.
- The workflow has no per-call journal and no rollback of external effects.
- A shared resource, such as a Blender session, has no admission control here. Do not add a protected endpoint, such as the Minecraft body, to this list.
- The initial context is not clipped. An oversized context still needs a check against the model limit.

## Deferred acceptance

These checks need real endpoints and a full cognition cycle. They have no receipts yet.

| Case | Required evidence |
| --- | --- |
| Normal skiing cat goal | Routed action, pinned context, real MCP calls, image bytes, consolidation, and a useful reply |
| Several calls for one goal | One continuous session across generation, inspection, and revision |
| Two configured MCPs | Tools from both servers in the same session |
| Unsatisfactory images | Retained attempts, a subjective judgment, a failed goal, and an honest reply |
| A remote artifact | The URI stays unverified as a host path |
| A missing claimed local file | The host rejects the success claim and keeps the agent's assessment |
| Endpoint or model failure | A failed or uncertain report, consolidation, and a useful failure reply |
| Cancellation during a call | Native cleanup, a cancelled receipt, and no automatic repeat |
| Changed instructions | Distinct instruction revisions and an uncertain result |
| Restart after terminal completion | No repeated goal, memory, or reply |
| Oversized launch context | Model-limit evidence and kept necessary sources |
