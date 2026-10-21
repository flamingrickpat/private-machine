# Companion cognition plan revision

## Summary

This revision expands A16 through A20 from selected historical mechanisms and the owner's current priorities.
It makes companion quality the primary acceptance concern.
It separates deferred large gameplay from cognitive dependencies.
It changes documentation only.
It does not establish new runtime behavior.

Parent: [alpha plan](../alpha-plan/index.md).
Policy: [companion development](../alpha-plan/companion-development.md).
Sources: [curated source guide](../alpha-plan/cognitive-sources.md).

## Source review

The starting pm revision is `3ff0ea051686f5837b64ebfdae6d9a3167e12434`.
The historical `private-machine` checkout is `7f9bd048dd211ffe0c027cf9865e0250101628c7`.
The historical `pm_next` checkout is `d5ddda71881db812ceb2de4ee59a3f871aa1c1cc`.
Those revisions establish source identities, not old runtime health.

Evidence stays under `.runtime/alpha/20261014-cognition-plan-revision/`.
Eight local PDFs were extracted to numbered pages with pypdf.
The source manifest records absolute paths, SHA-256 digests, page counts, and extracted text sizes.
Relevant passages were reviewed for workspace history, intentions, learning, introspection, and metacognitive episodes.
The guide links six papers with a direct implementation use.
It records why the other reviewed themes do not require new subsystems.

Selected old code was inspected for proposal families, thought operations, anticipation, and outcome consolidation.
Useful brainstorming distinctions were extracted into current requirements.
The summaries are not required reading for implementation.
No preserved source brief, historical receipt, persona seed, or dependency changed.

## Requirement changes

| Item | Added implementation and acceptance requirements |
| --- | --- |
| A16 | Family coverage, precursor pathways, proposal consumers, subjective belief revision, and causal behavior cases |
| A17 | Real private operations, shard isolation, commitments, prior expectations, metacognitive noticing, disclosure, and felt-state continuity |
| A18 | Cognitive episode questions, causal event links, capability status, evidence refresh, and revisable self-interpretation |
| A19 | Recorded hypotheses, alternatives, effective configuration versions, observed intervention results, failed diagnoses, and reversal |
| A20 | Integrated companion matrix and sustained use with modest independently verified tasks |

The policy retains existing reliability controls and diagnostic performance evidence.
It does not introduce a conversation latency threshold.
It retains A09R's advisory ontology and sole hard speech block, `false_task_result`.
Large historical gameplay remains failed, unverified, or deferred according to its actual receipts.

The owner also clarified two meanings of a companion society.
The society of mind inside one companion is the A15 and A16 codelet repertoire.
Many simple mechanisms compete. No separate engine is required.
A multi-companion society is a later deployment of the same runtime.
[Multi-persona deployment](../beta-ideas/multi-persona-deployment.md) repeats one companion container with a shared world and chat.
The policy records the deployment boundary so the alpha does not hardcode one user or one chat partner.

## Validation

This change edits documentation only, so it does not re-run the unit suite.
That suite covers code these pages do not affect.
The root contract suite passed 63 tests in 1.770 seconds.
Its receipt is `contract.log` and `contract.exit` in the evidence directory.

The initial strict documentation build passed in 73.88 seconds.
Its receipt is `docs-before.log` and `docs-before.exit`.
The final strict build passed after the completed revision.
Its receipt is `docs-final.log` and `docs-final.exit`.
The local HTML inspection read all 342 generated pages and resolved every local link and anchor.
`site-inspection.json` records zero missing files or fragments.
`inspect_site.py` reproduces the inspection.

No new live model or world gate is claimed.
The changed requirements remain planned until their owning items implement and verify them.
The next implementation item is A16.
