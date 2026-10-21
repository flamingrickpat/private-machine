# The private ontology

## Summary

The private ontology is durable state. It gives the companion stable words for
its inner life. A term records one word. A binding links that word to the
accepted state that supports its current use. The ontology is advisory: it
never rejects a reply sentence by sentence. The review reports an unsupported
label, and the label stays visible. It does not create a new feeling.

Parent: [public dialogue](public-dialogue.md).
Design: [the private ontology](../work_items/alpha-plan/ontology.md).
Record schema: [`pm.records.cognitive_records.OntologyTermRecord`][] and
[`pm.records.cognitive_records.OntologyBindingRecord`][].
Observer: [`workflows.public_speech.ontology.observe_ontology`][][].

## Purpose

The companion can describe its inner state in consistent words without
inventing a sense or an emotion that the runtime does not have.
A word such as `anger` is usable only because `StateEmotions` implements
`anger`. A word such as `smell` is not a term, because the runtime has no smell
mechanism. When a reply uses a word outside the term list, the
[agentic review](story-speech.md#agentic-review) reports an
`unsupported_label` finding. The finding is advisory.

## Records

Both record classes already exist in `pm/records/cognitive_records.py`. The
observer writes them. It adds no second ontology store.

| Record | Owns | Key fields |
| --- | --- | --- |
| `OntologyTermRecord` | One vocabulary word. | `term`, `term_kind`, `definition`, `usage_rule`, `stability`, `preferred` |
| `OntologyBindingRecord` | The current fit between a term and accepted state. | `term_id`, `bound_knoxel_ids`, `binding_kind`, `fit_score`, `explanation`, `active` |

A term kind is one of `quale`, `self_descriptor`, `relationship_label`,
`mental_move`, `mood_word`, or `goal_word`.

## Seed terms

[`seed_terms`][workflows.public_speech.ontology.seed_terms] derives the first
terms from mechanisms that exist:

- One `mood_word` per field of `StateEmotions`. The field description becomes
  the definition. The emotion word is usable only while the accepted state
  shows that emotion.
- One `relationship_label` per field of `StateRelationship`, except
  `entity_id`.
- A fixed set of `mental_move` terms for actions the architecture performs:
  `decide`, `withhold`, `deceive`, `reflect`, `plan`, and `speculate`.

A term seed has `stability = 1.0` because its mechanism already exists.
A later item (A17) may grow the term set from accepted experience.

## Observe one accepted state

[`observe_ontology`][workflows.public_speech.ontology.observe_ontology] is called
by [`PublicDialogue.public_context`][workflows.public_speech.public_dialogue.PublicDialogue.public_context]
before it freezes the public context. Story and Assistant share that method.
The observer:

1. Reads the latest accepted `internal_state` record and the newest committed
   cycle.
2. Creates any missing seed terms in one batch.
3. Reads the accepted mental state from `mental_state_appraisal_vec`.
4. Creates one active binding for each emotion at or above `BIND_THRESHOLD`
   (0.35), up to `MAX_BINDINGS` (12).
5. Marks the previous active binding for the same term inactive in the same
   commit, unless the support ids and the fit score are unchanged.

The binding names the internal-state record and its causal evidence ids. The
commit happens before the public snapshot, so the revision guards cover the new
records.

## State owner and commit boundary

The Ghost owns the records. One batch commits the terms and the binding
changes together. A reader sees either the previous ontology or the new one.
The observer commits no tick, so it does not disturb the accepted cycle or the
`state_tick` guard.

## Review input

[`workflows.public_speech.speech_review._ontology_lines`][] renders the terms
and the active bindings into the review prompt. A term line carries the word,
its kind, and its definition and usage rule. A binding line says that the term
fits right now and shows its fit score. The review may report an
`unsupported_label` finding for a word that is not in this list.

## Inspect a real result

Run a console exchange, then read the Story or Assistant report and the public
reply Feature.

- The report's `speech_review.findings` records any advisory finding.
- The report's `review_reason` records the reason of the accepted review.
- The public reply Feature's `review_reason` and `review_judgment` carry the
  same values beside the durable utterance.
- The ontology records are `OntologyTermKnoxel` and `OntologyBindingKnoxel`
  in the same MemoryStore. A binding has `active = true` only for the current
  state.

The unit check `tests/unit/workflows/public_speech/test_ontology.py` runs the
observer against a real MemoryStore. It verifies the seed terms, the notable
and weak feelings, the superseded binding, and that no term invents a sense.

## Failure modes and limits

- No accepted internal state: the observer writes the seed terms and creates no
  binding. The review receives the term list and reports an unsupported state
  word as a finding.
- A changed state: the previous active binding becomes inactive and a new one
  points at the new state. The public context filters inactive bindings.
- The term set is small and fixed at this stage. It grows only from an
  implemented mechanism.
- The ontology does not promote a reply into a feeling. Public speech stays an
  attributed utterance with `factual_promotion = false`.
- An unsupported label does not block speech. Only a false task result does.

## Related pages

- [Public dialogue](public-dialogue.md)
- [Story speech and the agentic review](story-speech.md)
- [Cognition state](cognition-state.md)
