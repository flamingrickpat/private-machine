# K002 causal memory and recovery

## Summary

This implementation retains immutable knoxels and complete workflow input manifests.
Main cognition admits new stimuli into an accepted history.
Approved actions can finish against that history without another cognition cycle.
Recovery reconstructs an accepted view from stored activity.
The runtime implements this contract. Live acceptance status is recorded on the K002 page.

Parent: [K002 simulation harness](index.md).
Current behavior: [public publication](../../subsystems/public-publication.md) and [runtime files](../../subsystems/runtime-files.md).

## Behavior observed before K002

The earlier `WriteBatch.replace` accepted a replacement under an existing record ID.
The commit increased that ID's revision counter and installed its new value.
The normal checkpoint retained the newest value without a general payload history.

Examples included goal completion, goal priority, exchange resolution, and delivery acknowledgment.
Belief supersession created a new belief and replaced the old belief to mark it inactive.
The base record had no universal tombstone field.

The base record had tick and sub-tick fields.
Runtime creation left the sub-tick at zero.
Main and inner cognition shared the tick allocator.
Record lineage and reply metadata supplied partial dependencies, without a complete recovery graph.

Main cognition already saved selected requests with pending dispatch metadata.
Admission ended some pending dispatches before their workflow finished.
K002 extends this mechanism with explicit terminal outcomes and direct replay of unfinished approved actions.

## Immutable knoxels and supersession

The owner requires a new knoxel ID for every logical change.
Each knoxel remains immutable after creation.
A successor names the knoxel it supersedes. The original does not change to point forward.
Deletion becomes an appended tombstone with its own dependencies.
The original payload remains available for historical views.

Normal queries select the visible heads of supersession chains in their current view.
Historical and causal references identify the exact original knoxel ID.
Historical inspection can show superseded knoxels and excluded branches.
A deletion in an excluded branch cannot hide a record in the recovered view.
The new checkpoint schema includes an explicit migration path.
An older checkpoint can establish a baseline, but missing historical payloads cannot be reconstructed.

Most mental records describe an event that happened once.
A later workspace entry references the thought or appraisal. It does not rewrite that event.
Persistent intentions and goals can acquire successor records.
Execution and delivery receipts describe lifecycle transitions separately from the original cognitive content.
Normal reads and derived cache updates do not create successor mental records.
Concurrent successors need an explicit head guard to prevent an unintended conflict between current states.

## Transaction dependencies

Each completed transaction records its input view, actual inputs, produced knoxel IDs, invocation identity, and outcome.
Inputs include exact knoxel IDs, prompt history, tool results, artifact versions, configuration, and clock observations.
Query scope and view identity cover collection reads, including an empty result.
Model-session summaries retain the dependencies of the history they replace.

The runtime captures these inputs mechanically.
A model's explanation of its sources cannot define the dependency manifest.
Untracked live memory reads must fail or acquire an explicit dependency.
Structural references can contain cycles. They remain distinct from causal dependencies between transactions.
Atomic transaction membership keeps jointly created records together.

This contract describes software input dependencies.
It does not identify which prompt tokens determined a model's answer.
Legacy activity with incomplete dependencies cannot establish a verified clean recovery before its baseline.

## Acceptance and completed action results

Successful main cognition admits the stimuli and records of its accepted transaction.
Its selected actions retain that admission as their authority.
A completed action result can join accepted history when its memory and stimulus dependencies belong to accepted history.
Observations and results from its authorized execution accompany that result.
Recorded timestamps do not independently require cognition admission.
New external input, private thought, and timer stimuli require main-cognition admission.
An output's relationship to an old tick alone does not establish acceptance.

For example, cognition 10 selects reply action A.
Action A completes using only admitted inputs while cognition 11 remains unfinished.
Recovery retains cognition 10 and the completed reply from A.
It excludes the provisional stimuli and results of cognition 11.

If A reads a later unaccepted message B, its reply also depends on B.
Recovery must exclude that reply until main cognition admits its dependencies.
Fast dialogue can keep its current behavior, but every later input must enter the dependency manifest.

Recovery preserves previous accepted history and verifies reference closure.
It does not collect records solely by their connection to the newest thought.
A missing dependency prevents publication of a supposedly clean view.

## Durable pending actions

Main cognition appends the selected request and its input references before dispatch.
An action remains pending during execution.
An atomic terminal transaction records its result or declared clean failure and ends its pending status.
Audit history remains after the action leaves the pending queue.

A crash without that terminal transaction leaves the action pending.
Interrupted dialogue retries the accepted request directly after restart.
Actions with external effects follow their existing recovery contract.
They cannot infer successful or absent effects from a process crash.

The result, exchange resolution, and queue completion must belong to one transaction.
Public delivery follows durable recording of that transaction.
Cognitive admission can remain pending because of an unaccepted input dependency.
If recovery excludes that terminal transaction, its approved action remains pending in the recovered view.
Stable output IDs permit deduplication after interruption.

## Raw history, recovery, and shutdown

Stored raw activity can include unfinished and excluded branches.
The accepted view remains a separate, validated projection of that history.
The clean operation writes a recovered ghost without changing its source capture.
The owner selected the clean causal view as the default startup projection.
Raw provisional history remains available for debugging.

A normal shutdown first stops new intake and autonomous timers.
It then drains running work and lets main cognition accept the remaining conversation.
The owner requires saving workflows parked for outside input and draining running work.
Connected simulations need coordinated participant checkpoints, clock state, and routing receipts.
Restoring one participant alone cannot reverse speech already accepted by another participant.

## Reference basis

The curated [cognitive source guide](../alpha-plan/cognitive-sources.md) identifies the introspection ontology paper.
Physical PDF pages 3, 6, and 9 distinguish communication, information flow, state transitions, and derivation.
Those passages support explicit inputs, outputs, and causal links.
They do not establish runtime correctness. Execution receipts provide that evidence.
The extracted passages are in `.runtime/kanban/k002-20261027-202608/introspection-source-passages.txt`.

## Runtime implementation

[`pm.memory.causal.CausalHistory`][] stores the import baseline and immutable commit transactions.
Main GWT commits admit their dependency closure.
Later results need an accepted action authority and accepted input dependencies.
Commit sequences establish storage order; they do not establish causality.

[`pm.memory.read_scope.ReadScope`][] records conservative input manifests across an invocation.
A manifest can include the complete scoped view supplied to a tool.
It therefore records available evidence, rather than claiming token-level influence on a model decision.
Untracked inputs raise instead of becoming accepted implicitly.

[`pm.memory.store.MemoryStore.export_clean`][] exports the accepted view to a different ghost file.
Use the launcher option `--clean DESTINATION` for this operation.
Normal checkpoint loading uses the accepted view; explicit `raw=True` supports debugging.
Legacy checkpoints become explicit import baselines because their lost earlier payloads cannot be reconstructed.

Logical edits append successors; normal queries select chain heads.
Exact-ID lookups retain accepted historical payloads for causal references.
Derived embeddings, embedding-model labels, access times, and media descriptions remain cache data.
Operational delivery and execution updates use transaction receipts without duplicating public speech.
Later workspace participation is recorded on GWT turns instead of changing an occurred mental event.

Parked workflow receipts embed their kernel journals and workspace bytes.
Restore installs those recorded bytes and archives the later workspace branch.
A fork rebinds its workflow paths and cannot read later controls from its source home.

Direct host collection reads capture stable membership and operational overlays.
Cycle and workspace reads also name their creating transactions.
An unrelated entity lookup does not acquire a dependency on a new dialogue message.
Main cognition consumes managed-run receipts through a captured read API.

Current-head guards remain separate from exact historical revision guards.
A superseded source retains its original ID and payload.
A caller that requires the current version uses `expect_head`.
Legacy commit receipts and logical commit aliases retain replay protection after recovery and clean export.

Speech preparation authorizes its derived ontology records through the selected action.
Those records retain their input dependencies and cannot admit an unrelated new stimulus.
The live crash gate exposed this requirement before the repair.
