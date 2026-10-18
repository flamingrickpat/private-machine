# Codelet pathways

## Summary

A pathway is a chain of codelets with a step strength between each pair.
When a codelet runs, it boosts the next registered codelet downstream in each chain that contains it.
The boost raises the stored activation of that codelet, so it tends to run in the next wave or the next cycle.
The boosted codelet's view of the CSM favors the percepts of the codelet that boosted it.

A19F step 5 replaced the A16 pathway signals.
The chains are the old-generation data, kept without change of content.

Parent: [codelets and attention competition](codelets.md).
Plan: [A19F](../work_items/alpha-plan/work-items/a19f.md), section "Pathways".
Earlier evidence: [A16 verification](../work_items/verification/a16-codelets.md). It describes the removed pathway signals.

## Data

`workflows/main_cognition/codelet_pathways.py` holds the chains of `private-machine-head/pm/codelets/pathways_detailed.py`.
[`get_codelet_pathways`][workflows.main_cognition.codelet_pathways.get_codelet_pathways] returns them by concept, for example `love`, `anxiety`, `curiosity`, or `wrath`.
A chain alternates codelet names and [`CodeletActivation`][workflows.main_cognition.codelet_pathways.CodeletActivation] steps.
The old file referenced codelet classes. The copy names each codelet as a string, because the registry holds codelets as data.

Most named codelets are not registered.
Of the twelve registered codelets, these chains link two directly:

- `BlockedGoalFrustration` boosts `DisrespectAttribution` (wrath).
- `AnalogousEpisodeRecall` boosts `ToMVignetteSimulator` (empathy).

Other links skip unregistered codelets and are therefore weak, for example `SelfWorthSensitivity` to `IdentitySpecialNarrative` across seven skipped codelets.

## Rules

[`pathway_boosts`][workflows.main_cognition.codelet_registry.pathway_boosts] applies these rules for each chain that contains the codelet that ran:

1. The strength is the step strength right after that codelet.
2. Walk downstream. Each unregistered codelet on the way multiplies the strength by `PATHWAY_SKIP_DECAY` (0.5).
3. The first registered codelet gets the boost. A chain without a registered successor gives no boost.

[`apply_pathway_boost`][workflows.main_cognition.codelet_registry.apply_pathway_boost] adds `PATHWAY_BOOST_SCALE * strength` to the stored activation of the target, up to `CODELET_MAX_ACTIVATION`.
It also adds the percepts of the run to the target's `precursor_ids` (port of `precursor_dict`).
`SituationalModel.snapshot_for` gives those records the bonus `CSM_PRECURSOR_BONUS`.

The old code multiplied the activation by the strength and boosted every codelet after the one that ran.
A multiplication has no effect on an activation of 0, and that was the usual value.
The additive boost of the next registered codelet keeps the chain order and has an effect from the first run.

## State

The boosts change `CycleState.codelets`, so they survive to the next cycle.
The stored activation decays by `CODELET_ACTIVATION_DECAY` in every cycle.
A run of the boosted codelet clears its precursor ids.
The frame metadata of each tick records the boosts under `pathway_boosts`.

## Limits

- A pathway raises activation only. It does not force a codelet to run or guarantee a reply style.
- The value of `PATHWAY_BOOST_SCALE` is an engineering choice. Tune it with replays.
