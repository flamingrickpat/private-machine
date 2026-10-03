# Manual milestones and acceptance evidence

## Summary

Seven milestones let the user test visible behavior before later work begins.
Contract tests establish engineering properties.
Real endpoints establish behavioral evidence.
The user evaluates immersion.
Failed or unavailable live gates remain visible in the handoff.

Parent: [alpha plan](index.md).
Baseline: [current-state audit](current-state.md).
Delivery items: [ordered plan](index.md#ordered-delivery).

## M1: decisions over real artifacts

**After A05.** Run the local Laya service and the delivered classification demo through PM.
M1 adds the client and node capability, plus a new demonstration workflow.
It does not require changing existing PM or coding-workflow manifests to use Laya.
Produce several real test reports in a host workspace.
Let Laya select the relevant report from filenames and metadata, then classify its captured contents.
Repeat with a fake filesystem and a changed report.

Inspect both selection and classification receipts, the digest, INVALID handling, and destination.
Try missing data, a stale producer attempt, an unavailable endpoint, and an oversized report.
These conditions must have distinct failure evidence.
Existing coding workflows must still compile and run.
Their existing model choices remain the default, and they run without the decision service.

A02 supplies a documented service command using the downloaded GGUF.
A04 supplies an executable demo for both environments.
A05 adds optional PM connection wiring and runs that demo through PM.
No agent browsing is necessary on the normal Laya path.
Inspect a scoped coding-agent fallback for an oversized or ambiguous input.

## M2: a continuous companion

**After A09.** Discuss a planned shared activity, then introduce disappointment and later repair.
Discuss another topic before returning to the activity.
Restart the process and continue the conversation.

Inspect persistent affect, active goals, relevant memory, and causal links.
The companion should remember what happened and respond in its established voice.
Its emotional behavior can evolve without an instant reset.
Inspect startup/shutdown evidence and its reaction to the elapsed offline gap.
It must not describe unexecuted activity during the stopped interval.

Ask what it can currently perceive and do.
Compare its answer with active configuration and available tools.
New Minecraft capabilities should be described as game capabilities.

## M3: a finished house and iron pickaxe

**After A13.** Request a small usable house and an iron pickaxe from self-mined iron.
Use an isolated Babymode world with no supplied iron or pickaxe.
Do not obtain iron by taking village supplies.

Inspect the generic goal tree and its fresh specialist sessions.
Verify actual resource gathering, mining, smelting, crafting, and construction.
Observe a finished usable house and the crafted pickaxe in inventory.
Preserve the ore-to-ingot-to-pickaxe evidence chain.

Interrupt with conversation and change a building detail while work is active.
Confirm that accepted instructions reach the actor at an action boundary.
Ask for its inventory and verify direct Assistant tool use without another managed task.
Run cancellation and restart variants using the same fixture setup.

A10's five-tree task remains an earlier developer check.
The first user Minecraft milestone requires the complete house-and-pickaxe result.
Call counts are descriptive evidence.
Requiring exactly 150 calls would reward inefficient behavior.

## M4: adaptive shared survival

**After A14.** Extend or adapt the earlier house.
Change its footprint while materials are being gathered.
Work beside the companion.
Obstruct a route or remove an assumed resource.

Then request a diamond expedition in a recorded fixture.
Introduce a failed search, low supplies, and a hazard.
Inspect revised subtasks, local negative-search evidence, recovery, and return planning.

Completion requires actual acceptance evidence.
Test a self-origin goal and an appraised refusal using recorded persona state.
House and diamond fixtures must exercise the generic planner, rather than dedicated hardcoded workflows.
A safe, explained partial result can be correct when conditions prevent completion.
It does not pass a fixture that was designed to allow success.

Separate tests cover body disconnect, process restart, unknown action outcome, and concurrent actor attempts.
The companion must reobserve before retrying an uncertain world mutation.

## Later milestones

[M5–M7](cognition-verification.md) verify the second conscious loop, self-reflection, and sustained integrated use.
The [evaluation contract](evaluation.md) defines measurements and session evidence for every milestone.
