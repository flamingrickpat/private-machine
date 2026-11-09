# Month-long simulation workspace

Status: owner-approved implementation, 2026-11-04.

## Current path

`simulation.server.Simulation` creates browser jobs and `RunEngine` instances.
`Agents.plan` creates `ScenarioPlan`, whose ordered `Step` objects drive actor
generation, delivery, settling, and director checks. `RunEngine._report` captures
immutable attempt evidence. `pm.tools.timeline.api.TimelineApi.get` reads that
evidence through `EvidenceStore` and `AttemptView`. Its standalone server serves
a separate frontend. The simulation frontend uses a separate dark style.

`MainCogLoop.publish_output` calls `MemoryStore.backup` after every accepted
cycle. The current checkpoint is saved separately by memory commits.

## Approved target

One light-default simulation website owns setup, execution, analysis, and the
debugger. The debugger stays independently usable. Its existing static frontend
and read-only API are mounted under `/debugger/`; no second server is required.
The embedded debugger retains its own DOM and shares the theme with its parent.

`simulation/templates.py` owns a versioned `SimulationTemplate`, named participant
and connection contracts, and `DirectorSettings`. Templates contain a brief,
director settings, analysis settings, and an optional exact `ScenarioPlan`.
The browser and `python -m simulation.templates` use the same JSON contract.
`ScenarioPlan` adds director settings and a minimum turn count per scene.
Long-plan generation works one day at a time with an editable combined result.
Actors see current scene circumstances and public history, never future plot.

`simulation/analysis.py` owns typed rating settings, cited findings, and analysis
reports. Two starter criteria assess conversational sense and immersion as an
AI companion. `analyze_run(root: Path, config: str) -> Path` reads completed
public deliveries, rates bounded sections, and writes a separate analysis job.
Each result retains its exact input, coverage, prompt, and model output.
Failed analysis never changes a completed companion run into a failed run.
Owner feedback stays separate. Analysis can run again with changed criteria.

`ModelConfig.ghost_backups: bool = False` controls automatic cycle copies.
`MainCogLoop` receives this flag from the assembly. The main checkpoint and
explicit recovery snapshots remain required. Existing backup files are retained.

## Ownership and failures

Runtime code never imports the simulation. The debugger remains an evidence
reader in `pm/`. The simulation imports its public API. No runtime modules move.
Templates and browser requests validate at their external boundary. Internal
contract errors propagate. Real model calls use the existing visible retry and
cancellation paths. Required context must fit; no silent evidence truncation.

## Steps and verification

1. Add template, director, and rating contracts and command-line entry points.
2. Add daily planning and minimum-turn execution without changing cognition.
3. Mount the debugger and build light-default setup and review controls.
4. Add automatic post-run ratings and explicit reanalysis.
5. Add the backup switch, update conceptual docs, and build MkDocs.
6. Run focused deterministic checks, real model planning/rating checks, and a
   real short companion run. Inspect the browser and resulting evidence.
7. Create an eventful adult Jesse/Spamuel month template and start a detached
   real run on OpenRouter, bound to `0.0.0.0`, with retained logs and status.

## Limits and risks

Virtual months still execute the runtime's real wake and thought schedule.
Duration in wall time and inference cost can be large. Three runs are an owner
review method, not proof of statistical significance. Rating agents judge
observable behavior, not consciousness. Richer private introspection criteria
can follow later. No sandbox escape or filesystem-access scenarios are in scope.

## Implementation evidence

Evidence root: `.runtime/simulation-workspace-20261104/`.

- Strict configured Pyright: zero errors, including the three new contract modules.
- 25 focused deterministic checks passed. These cover validation, causal
  checkpoint handling, record review, timeline reads, real backup files, and
  import boundaries. `checks.log` and `types.log` retain the output.
- The legacy scripted cognition-wiring tests were updated for the new backup
  default. They were not used as live model evidence.
- Real OpenRouter daily planning produced the short acceptance plan.
- Both evaluator agents rated all 14 deliveries in the retained completed cat
  run. Their exact quotes validated. Scores were 2/5 for conversational sense
  and 3/5 for immersion; these are model judgments, not owner acceptance.
- Browser checks passed template import/export, the integrated debugger, and
  matching light backgrounds without JavaScript errors. Screenshots are retained.
- The strict MkDocs build passed. The final build output is in `docs-build.log`.
- The exact month template contains 30 days, 90 scenes, and a minimum of 360
  user messages. CLI validation and creation passed; no model planner is needed
  to execute that supplied plan.
- `dev/simulation-themes.txt` is an independent, non-executable idea register.

The fresh real two-turn companion run and its automatic analysis remain in
progress until their final receipts are recorded below. The source home remains
unchanged. Current checkpoints remain present and no automatic backup copies
appeared during accepted cognition.

The participant logs contain `httpx2.AsyncClient.aclose` event-loop cleanup
warnings. The same diagnostic exists in the older cat run. It did not become
an attempt failure in the observed run; its original traceback remains intact.

## Live completion and month launch

The short real run `20261104-211346-7f064a4b` completed two user turns and two
companion replies with no attempt failures. Automatic analysis completed both
criteria with exact supporting citations. `verified-artifacts.json` records the
actual files: the original source hash is unchanged, the current checkpoint is
154,231,915 bytes, no automatic ghost copies exist, and the coordinated snapshot
and attempt report are present. The scores are model judgments, not owner ratings.

Additional real browser checks passed month-template import, 30 day groups,
scene editing, citation navigation, reanalysis, and standalone debugger use.
The browser's reanalysis created a new retained report through the server job
path. Three generated manual pages opened locally; 552 local link targets existed.
The existing Mermaid blocks did not become SVG diagrams during the local-file
inspection. That manual-rendering limitation remains visible; the strict build
passed and the simulation dashboard does not depend on Mermaid.

The prepared exact month run is `20261104-eventful-month`. A hidden Windows
scheduled task, `PM-Eventful-Month-20261104`, owns its dashboard and monitor.
The task uses the owner's interactive logon and has no execution time limit.
S4U registration was denied by Windows; interactive registration succeeded.
The server listens on `0.0.0.0:8080`. The participant is running against frozen
source and OpenRouter with completion replay disabled.

- Dashboard: `http://127.0.0.1:8080/#run/20261104-eventful-month`
- SSH-readable status: `.runtime/simulation-workspace-20261104/long-status.json`
- Dashboard log: `.runtime/simulation-workspace-20261104/dashboard.log`
- Task monitor: `.runtime/simulation-workspace-20261104/long-launch.log`
- Completed result: `long-result.json`, written only after the run and analysis finish.

The two temporary verification servers on ports 8095 and 8096 drained and exited.
The month is running, not completed or quality-approved. It retains the ordinary
runtime schedule; no cognition, private thought, or quiet interval was bypassed.
