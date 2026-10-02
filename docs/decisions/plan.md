# Earlier plan and current implementation

## Summary

The historical plan divided delivery into W0 through W8.
Current code implements the main boundaries from that plan.
Several deliberate limits remain.
Historical acceptance results do not prove quality for a later revision.

Parent: [manual](../index.md).
Current structure: [system map](../concepts/architecture.md).

## Recover the original documents

The original plan is available at commit `6415dd0`.
The following command shows its architecture document.

```powershell
git show 6415dd0:docs/plan/architecture.md
```

Use the same command with `docs/plan/work-items.md`, `docs/plan/tests.md`, or `docs/plan/models-memory.md`.
Those documents describe planning state at that revision.
They do not describe current delivery status.

## Delivery map

| Item | Planned result | Current implementation |
| --- | --- | --- |
| W0 | Establish a known baseline | Historical delivery evidence and foundation suite |
| W1 | Model connections and embedding cache | `runtime/models.py`, activation, `pm_core/embeddings.py` |
| W2 | Ghost, bus, durable accepted state | Ghost batches, revisions, checkpoints, detached envelopes |
| W3 | Workflow-to-agent boundary | Shared runner, scoped targets, per-run services and controls |
| W4 | Standalone Story dialogue | Story workflow, direct CLI, ontology contracts |
| W5 | Real cognitive children and conversation | Seven child workflows and real application wiring |
| W6 | Assistant and managed work | Assistant tools, supervisor, steward, consolidation |
| W7 | First-delivery acceptance | Conversation harness and evidence-based evaluator |
| W8 | Minecraft and richer consolidation | One-action workflow, isolated live gate, spatial and image evidence |

History includes W5 wiring at `97af9ae`, W6 at `d5d4114`, W7 at `e6f160a`, and W8 at `47bfaea`.
Commit `3ea45a1` reconciles later real-path and evidence requirements.
Later commits change backend retries and Minecraft startup.

## Recorded decisions

The historical architecture explicitly separates module lifetime from workflow lifetime.
It keeps child cognition synchronous instead of creating a thread per cognitive stage.
It keeps character policy inside workflow folders.
It excludes a broker, a plugin framework, and automatic repair from the first delivery.

The plan requires Ghost to accept durable state before publication.
It requires evidence-backed task claims instead of treating scheduling intent as success.
It keeps real internal thought deferred.
These decisions explain the current state and control boundaries.

## Reconstructed rationale

**Inferred rationale:** Ordered stages make intermediate decisions available for inspection and direct tests.
This inference agrees with the stage reports and typed contracts.
It does not prove that seven model calls are the fastest useful design.

**Inferred rationale:** Separate Story and Assistant paths isolate character speech from tool execution.
Their distinct prompts, schemas, and tool grants support this interpretation.

## Open limits

Main cognition does not yet retrieve candidates through embeddings.
Minecraft has no sustained survival policy.
Real internal thought is disabled.
The console adds local operation and observation, not a new cognitive policy.
The repository contains no implemented JEPA or LAYA integration.
The intended model and integration contract remain undefined.

## Go deeper

Read [cognition](../subsystems/cognition.md), [evidence](../concepts/evidence.md), and [next experiments](experiments.md).

