# Ordered implementation work items

## Summary

Each work item ends with observable evidence and a usable checkpoint.
Write the relevant failing tests before the implementation.
Do not redesign the completed contracts during a later item.
Report a genuine conflict instead of silently changing the plan.

Parent: [implementation handoff](../../IMPLEMENTATION_PLAN.md).
Acceptance details: [test requirements](tests.md).
Commit and package procedure: [GitHub delivery](delivery.md).

Current checkpoint: W0 baseline rechecked, W1 delivered in version 0.2.0, and W2 delivered in version 0.3.0.
Read the [W1 delivery record](../work-items/W1.md) and [W2 delivery record](../work-items/W2.md) for evidence and limits.
The next unfinished item is W3.
Its [current checkpoint](../work-items/W3.md) provides real agent target binding and inference cancellation in application version 0.5.0.

## Source map and observed gaps

These paths describe source inspected during planning, not completed future features.

| Source | Reuse or required change |
| --- | --- |
| `shell.py` | Keep direct `step()` and common thread lifecycle. Isolate delivered payloads. |
| `workflow_adapter.py`, `runtime/workflow_runner.py` | Shared runner and typed results are available. Per-run cancellation remains unfinished. |
| `ghost.py`, `ghost_mcp.py` | Keep real models and scoped reads. Add proposals, capabilities, atomic commits, and persistence. |
| `working_target.py` | Keep lazy projection, batch registration, and exact file text. |
| `../pm-workflows/pm_workflows/runtime.py` | Extend per-run context without Ghost coupling. |
| `../pm-workflows/pm_workflows/kernel.py` | Reuse child phases. Replace shared child environment with scoped child namespaces. |
| `../pm-workflows/pm_workflows/drivers/minimal_agent.py` | Target, model connection, and cancellation are available. Complete scoped dynamic MCP integration. |
| `../pm-coder/pm_coder.py` | Explicit workspace discovery and live-test failure reporting are available in 0.9.1. Complete production failure classification. |
| `../pm-coder/pm_bash_machine.py` | Preserve batch lazy-file registration and raw-output support. |
| `../pm-workflows/pm_workflows/dryrun.py` | Extend existing graph checks and simulations. |
| `../pm_next/pm_core/subsystems/context/` | Adapt planner and render helpers. Do not import the old orchestration layer. |
| `../pm_next/pm_loop_engine/services/embedding_cache.py` | Reuse legacy cache lookup and metadata format only. |
| `../pm_next/pm_core/system/llm/llm_disk_cache.py` | Reuse binary vector reading, not the whole old LLM framework. |

`PythonDriver` and `PmCoderDriver` receive runtime context.
The current kernel passes the same runtime object into a child.
Role MCP filtering currently depends on the parent allowlist path.
Test top-level roles as well as children. Do not permit unrestricted top-level discovery.
W2 now provides accepted-state persistence. The application still lacks live cognition.

## W0: establish the baseline

1. Inspect instructions, Git status, and branch names in every affected repository.
2. Preserve existing changes in `pm-workflows` and `pm-coder`.
3. Use the Python 3.12 `.venv` and GitHub dependency revisions from `uv.lock`.
4. Run the current application tests and demonstration.
5. Run affected dependency tests.
6. Record commands, revisions, and actual results.

Useful existing commands, from this repository:

```powershell
uv sync --locked
uv run --locked python -m unittest discover -s tests -v -b
uv run --locked python demo.py
```

The real-folder demonstration needs the working Git Bash on PATH, not the unusable WSL launcher.
The delivery baseline recorded 34 application tests, 55 workflow tests, 30 Bash Machine tests, and 21 demonstration checks.
The [delivery record](delivery.md#bootstrap-releases) identifies the tested setup. Repeat these checks when implementation starts.

Gate: known baseline, preserved user changes, and no dependency changes in the running Hadal environment.
For dependency development, use their source checkouts. Final application validation uses their pushed GitHub packages.

## W1: model connections and existing embedding cache

Implement `runtime/models.py`, explicit activation, and `pm_core/embeddings.py`.
Add credential-free example config and ignored local secret config.
The delivery setup ignores `.env` and local config. Keep those exclusions before adding secret files.
Implement both chat startup forms and the single embedding connection.
Reuse legacy cached vectors without rewriting their directory.
Correct cosine handling for non-unit vectors.
Keep network access out of constructors and seed loading.

Gate: resolver tests, credential-redaction tests, real cache-hit evidence, and live generation and embedding requests.
Do not regenerate the cache or change the shared seed as a shortcut.

Delivered: [model guide](../models.md), [config](../../config.example.yaml), and [W1 evidence](../work-items/W1.md).

## W2: Ghost, bus, and durable accepted state

Move responsibilities into `pm_core`, `bus`, and `runtime` incrementally.
Implement independent envelope delivery and preserve synchronous module behavior.
Add scoped snapshots, `GhostWriteBatch`, reserved IDs, revisions, and commit receipts.
Implement the versioned snapshot and atomic checkpoint replacement.
Preserve every required field in the existing seed and GWT models.
Keep proposed changes invisible to other runs until commit.

Gate: direct bus tests, full-seed round trip, concurrent batch tests, conflict rejection, and failed-save recovery.
No model call holds the Ghost lock.

Delivered: [Ghost and bus guide](../ghost.md) and [W2 evidence](../work-items/W2.md).

## W3: complete the workflow-to-agent boundary

Extract the shared runner from the current adapter.
Extend `WorkflowRuntime` and both Python and pm-coder driver paths.
Use `run_auto_with_bash_machine` or its asynchronous equivalent for virtual targets.
Keep the existing real-folder path available.
Pass the session store explicitly through pm-coder paths that currently rely on the global active session.
Make every file tool obey the supplied target and its access rules.

Add dynamic MCP binding, role grants, child namespaces, per-run events, and cancellation propagation.
Keep Python tools scoped without claiming that arbitrary Python is sandboxed.
Use per-run MCP config files. Do not mutate shared files or environment variables during concurrent runs.
Preserve plain Python workflows as a first-class path.

Gate: real Python and agent runs on both targets, scoped child tests, MCP cleanup, concurrent isolation, and cancellation.
This is the critical infrastructure gate. Do not hide an incomplete agent path behind dummy cognition.

## W4: standalone Story dialogue

Create `workflows/dialogue_story` with its manifest, prompts, scripts, schemas, and direct tests.
Adapt existing context selection and message rendering.
Use the selected embedding service and direct generation client.
Implement the private-ontology acceptance boundary.
Provide one command that runs a fixed Spamuel situation without module threads.

Gate: exact message-role tests, real accepted dialogue, persisted result, and ten saved quality trials.
The workflow must not call pm-coder merely to obtain a text completion.

## W5: real cognitive children and conversation wiring

Create the seven child workflows in the agreed order.
Add each stage's direct tests before connecting it to the parent.
Implement typed action dispatch records and dependency receipts.
Connect UserInputProxy, MainCogLoop, and Story dialogue through the real bus.
Remove dummy character policy from the real application path.
Keep old demonstrations only when clearly labeled and separate from acceptance.

Gate: real user input produces an accepted GWT, action intent, and correlated reply.
Attention counterexamples and all child contracts pass.
No cognitive keyword rules appear in the generic runtime.

## W6: Assistant dialogue, management, and stewardship

Create separate Assistant prompts and tool-capable sessions.
Implement one managed-run table and shared start/status/update/terminate operations.
Expose those operations through scoped MCP tools and SelfManagement.
Implement the dormant steward with code-only cancellation and explicit update boundaries.
Use a small real local workflow to exercise concurrent execution and blocking questions.
Implement basic report-to-knoxel conversion and evidence-backed final consolidation for its text artifacts.

Gate: tool receipts support dialogue claims, updates reach the intended run, status reflects evidence, and cancellation preserves artifacts.
Repeated action or consolidation requests do not duplicate accepted state.
No doctor repair or prompt editing enters this item.

## W7: first-delivery acceptance

Extend the existing dryrun checks and add finite routing simulations.
Run the direct and threaded integration suites.
Run ten conversation scenarios with twenty exchanges each.
Activate the evaluator only after generation finishes.
Save every result and apply the fixed quality gates.
Perform the extension test without bus or runtime edits.

Gate: all hard contracts pass, quality thresholds pass, and evidence identifies every model and tested revision.
Report evaluator errors separately from quality failures.
Do not declare success from a foundation-only run.

## W8: Minecraft body and richer consolidation

Inspect the current Minecraft MCP and its existing isolated-server tests.
Add the Minecraft workflow with its own prompts, scripts, capabilities, and real-world evidence.
Reuse the managed-run and steward contracts unchanged.
Extend consolidation for spatial records and base64 screenshot evidence.
Preserve task folders until durable consolidation succeeds.

Gate: a real isolated server demonstrates observation, an action, status, update, termination, and memory recovery.
Do not substitute a conversation simulation for this result.

## Handoff after each item

Add `docs/work-items/<item>.md` with the structure of the [W1 delivery record](../work-items/W1.md).
Record changed files, executed commands, evidence paths, and remaining failures.
Keep the README accurate about current behavior rather than planned behavior.
Commit and push tested work under the user's standing authorization.
Publish changed dependencies before refreshing, testing, and pushing the application lockfile.
Include package versions, commit links, and actual test results in the completion report.
Do not add internal thought, a doctor, containers, a broker, or an optimization framework to fill spare time.
