# Curated cognitive sources

## Summary

These sources informed the revised A16 through A19 requirements.
Each entry names the useful mechanism and the limits of its adoption.
The work items contain the required behavior.
Developers do not need to search the historical document folders before implementation.

Parent: [alpha plan](index.md).
Policy: [companion development](companion-development.md).
Review evidence: [plan revision](../verification/cognition-plan-revision.md).
Requirements: [A16](work-items/a16.md), [A17](work-items/a17.md), [A18](work-items/a18.md), and [A19](work-items/a19.md).

## How to use a source

Read a source only to resolve a specific mechanism or implementation question.
Treat the extracted requirements as the current plan.
Record any justified departure in the selected item's verification record.

Links use absolute local paths at the owner's request.
PDF page numbers count from the first physical page, including a publisher or repository cover.
They can differ from printed page numbers.

## Papers with an implementation use

| Source and passage | Selected mechanism | Application and boundary |
| --- | --- | --- |
| [LIDA brief account](file:///C:/source/pm/pm_next/documents/pdfs/LidaBrief-Account.pdf#page=7), PDF page 7 | Recent workspace broadcasts remain available for later structure building. Action schemes connect context, action, and expected result | A17 reads recent committed turns. Reuse current records. Do not add another cognitive engine |
| [Long-term intentions and narratives](file:///C:/source/pm/pm_next/documents/pdfs/KronstedEtAl-2021-ModelingLongTermIntentionsAndNarratives.pdf#page=8), PDF pages 8, 11, 20–22, 27–29 | Autobiographical narratives influence commitments. Long-term goals guide immediate choices. Deliberation admits supporters and objectors | A16 supplies motive and narrative proposals. A17 links commitments to chosen actions. Narratives remain interpretations with source references |
| [Learning in LIDA](file:///C:/source/pm/pm_next/documents/pdfs/LearninginLIDA-CSR2019-InPress-Final.pdf#page=13), PDF pages 13–14 and 19 | Affect can influence attention, action, and memory retention. Learning can add a pattern or revise an existing pattern's strength | A16 demonstrates all three effects. A19 changes a supported strategy. This plan does not require neural training or LIDA's complete memory system |
| [Practical artificial qualia](file:///C:/source/pm/pm_next/documents/pdfs/practical_arrabales_2010.pdf#page=5), PDF page 5, Section 4.1 | Content representation, introspective representation, and reportability or self-modulation have distinct roles | A17 derives felt-state records from accepted drivers. Reports do not establish consciousness or direct access to model internals |
| [Ontology for introspection and metacognition](file:///C:/source/pm/pm_next/documents/pdfs/Towards_an_Ontology_for_Robot_Introspection_and_Me.pdf#page=3), PDF pages 3, 6, 9, and 11–13 | Questions define the trace contract. Events identify inputs, outputs, participants, derivation, and causal links. Software design differs from a running instance | A18 answers concrete questions about execution and capability. Reuse records and graph links. OWL, SOMA, and SPARQL are not required dependencies |
| [Remembering metacognitive experiences](file:///C:/source/pm/pm_next/documents/pdfs/How_Metacognitive_Architectures_Remember_Their_Own.pdf#page=29), PDF pages 29–30, 40, and 47–48 | Episodic traces support diagnosis, recall, and changed strategy. Parallel events need explicit links. Diagnosis includes success and failure patterns | A18 retains inspectable cognitive episodes. A19 compares interventions. The review does not establish which architecture produces the best companion |

These are engineering adaptations of the sources.
The papers do not verify pm's behavior.
The project's acceptance cases establish that behavior separately.

## Historical code with a narrow reuse scope

| Source | Reuse | Do not inherit |
| --- | --- | --- |
| [Codelet definitions](file:///C:/source/pm/pm_next/pm_core/subsystems/codelet/codelet_definitions.py) and [pathways](file:///C:/source/pm/pm_next/pm_core/subsystems/codelet/pathways_detailed.py) | Family coverage, `data_aquisation` requirements, and precursor activation | All 342 definitions, narrator prompts, invented embodiment, or clinical labels as diagnoses |
| [Thought operations](file:///C:/source/pm/pm_next/pm_core/subsystems/thought/thought_graph.py) | Temporal focus, generative and discriminatory modes, and 37 named operations | Random transition weights as verified psychology, unrestricted branching, or copied dialogue prompts |
| [Original pm_lida.py](file:///C:/source/pm/private-machine/pm_lida.py) | `_deliberate_and_select_reply`, `_generate_expectations_for_action`, and `_perform_learning_and_consolidation` connect anticipation to later outcomes. `_render_meta_insights` exposes recorded processing | The monolithic runtime, hardcoded text-only capabilities, fixed story framing, or placeholder learning as completed adaptation |

Inspect named functions before borrowing code.
The older implementations have no current operational acceptance claim.

## Extracted brainstorming decisions

The revised work items incorporate these useful distinctions:

- Codelets produce typed internal proposals. Public dialogue owns final wording.
- Subjective beliefs retain supporting and opposing evidence. They do not rewrite original observations.
- Private thought can revisit a memory, compare possibilities, or change a goal without producing a public reply.
- Inward attention differs from pipeline execution stage.
- Recent workspace turns support noticing attention drift and repeated interpretation.
- A persistent narrative develops from attributed episodes. A single generated thought does not redefine identity.
- Felt-state vocabulary links recurring internal patterns to stable terms.
- Reflection records a prediction before an intervention and observes its later outcome.

These requirements replace broad historical reading instructions.
The brainstorming summaries remain historical records rather than required implementation manuals.

## Ideas excluded from this phase

The following ideas are not implementation requirements:

- Claims that state derivatives, thermodynamics, or a thought graph prove consciousness.
- A second state owner, an alternative runtime, or separate authoritative qualia storage.
- Full OWL infrastructure or neuronal event taxonomies without a demonstrated pm use.
- Reconstruction of meaning from embeddings as a substitute for original records.
- Blanket suppression of metaphor, sarcasm, chosen deception, or emotionally expressive speech.
- Automatic transformation of generated dialogue into external facts.
- Automatic source rewriting, a third personality, or a compulsory meta-meta loop.
- Automatic discovery of hundreds of codelets before a useful repertoire works.
- Fixed low inference counts or a faster response as the main quality criterion.

Review of smooth coping and neuronal ontology sources did not justify new mandatory subsystems.
Their relevant themes are already covered by commitments, event provenance, and selective attention.
Workspace momentum and automatic codelet discovery retain their existing beta scope.
