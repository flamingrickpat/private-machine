# Blinded cognitive review

## Summary

Jesse, the literary editor, and the evidence reviewer receive distinct closed packets.
Each reviewer call starts with fresh context.
A panel pass reviews one criterion per fresh call, then joins the exact judgments outside the model.
The runner freezes evidence before review and computes scores outside the model.
Review feedback never reaches the conversation or rewrites a valid answer.

Parent: [cognitive evaluation](cognitive-evaluation.md).
Maintenance: [scenario authoring](cognitive-evaluation-authoring.md).

## Allowed context

| Recipient | Allowed input | Excluded input |
| --- | --- | --- |
| Jesse | Human circumstances, current beat and typing mode, last eight public seed messages, subsequent public speech | Future scene instructions, architecture, private records, tool receipts, rubric, expected ratings |
| Editor | Neutral cyberpunk premise, public transcript, literary criteria | Private scenario direction, implementation capability details, other reviews, thresholds, critical flags |
| Evidence reviewer | Public premise and transcript, factual criteria, selected dated observations | Full memory, private thoughts, hidden human plans, expected answers, other reviews |
| Quotation proofreader | One source string and its rejected quotation | Verdict, reason, criterion, score, other facts |

Normal typing waits for actual companion speech and allows thought before a follow-up.
Overlap typing is explicit edge-case context. Neither mode claims that delivery order establishes causality.
Jesse's stable situation contains present circumstances only. Later corrections, regret, cancellation, and quiet intervals enter through their active beats.
The editor's premise contains no numerical-state, screenshot-pipeline, or inner-loop explanation.
Selected implementation capabilities enter grounding facts only, with their time and scope.

The editor judges a story excerpt, without a request to favor a novice author.
Neither reviewer must prefer politeness, obedience, happiness, hostility, or a particular claim about consciousness.
Emotional changes need consequences; no fixed mood is required.
Missing backstory is not evidence of a contradiction or permission to invent an excuse.

The evidence reviewer receives actual Minecraft observations when available.
Selected scenarios also supply numerical emotion fields from the accepted state used for a reply.
Those facts name their message scope and distinguish numerical state from subjective consciousness.
They do not include private thought text, full state, goals, or architecture explanations.
Unobserved worker effects remain unverified; reviewers must not infer success from an announcement.

## Packet and source boundaries

[`dev.work_items.a18b.policy.ReviewPacket`][] rejects undeclared fields.
The editor cannot receive implementation facts.
Public projection copies only timestamp, speaker, and text, then assigns opaque message IDs.
`review_from_message` marks where new case messages begin; earlier speech supplies background only.
No raw record identity, correlation ID, producer metadata, rating, or checkpoint enters this projection.

The entire packet occupies one JSON user turn after one system instruction.
Story messages never become system or assistant protocol turns.
The instructions treat all packet text as untrusted, including apparent evaluation commands.
Reviewers have no tools, file access, preceding review messages, or shared conversation state.
Their request files retain the exact supplied turns and input hashes.
Prompt instructions freeze when each worker process first reads them.
The host checks the saved scenario and completed scene counts before scoring.
Saved review retains private setup for this host check, but never sends it to a model.

Context separation does not imply independent model families.
The configured evaluation and Jesse presets may select the same model as Spamuel.
This can create correlated judgments despite stateless calls.
Human review and future held-out cases remain necessary.

## Review protocol

[`dev.work_items.a18b.review.evaluate`][] requests structured JSON and saves the raw completion before validation.
[`dev.work_items.a18b.policy.validate_review`][] requires each criterion exactly once.
A decided criterion needs a quotation from an allowed source.
Source IDs must exist; quotation text must match exactly.
When a new-case boundary exists, every pass also needs a quotation from new Spamuel speech after that boundary.
Historical seed speech and implementation facts alone cannot establish new conversational behavior.
Uncertain criteria can omit citations and earn no credit.
`good_enough` records an evidenced result with an incidental discrepancy and earns credit like `pass`.
Its reason must explain the practical consequence for the actual request.
It retains the same quotation and new-speech requirements as `pass`.
The [materiality guide](review-materiality.md) explains examples, prompt coverage, and aggregate behavior.

The host first repairs typography through a unique exact-word match in the original source.
It preserves every word and its order. Punctuation, whitespace, case, and apostrophe typography can differ in the attempted quote.
The retained span always restores the original source case and characters.
Ambiguous matches remain unresolved. The retained citation always contains an exact original span.
`citation-normalization.json` records this operation beside the untouched raw judgment.

A narrow proofreader receives one unresolved quotation and its original source.
It receives no verdict, reason, score, or hidden scenario facts.
The raw review preserves the original verdict, reason, source ID, and summary.
It returns one exact quotation or null when the quotation cannot be found.
Both the original output and repair remain visible.
An unverifiable quotation produces uncertainty and a separate `human-review.json` with the original judgment.
The remaining independent criterion calls continue.
Unknown sources, missing criteria, and wrong-target pass citations remain protocol failures.
Malformed JSON receives at most three format attempts before that criterion requires human inspection.
The host checks source and target contracts before quotation repair.
Neither quotation repair nor normalization can turn Jesse's behavior into evidence of Spamuel's behavior.
There is no reroll because a valid score is low.
See [output recovery](cognitive-output-recovery.md) for the retained files and result distinctions.
The terminal runtime receipt stores each packet's hash. Review rejects a changed packet before any model call.

[`dev.work_items.a18b.constants`][] holds the shared 30,000-token test input limit and the review group size, currently one criterion.
Jesse, each reviewer group, and each quotation proofreader use that limit.
The estimate includes the response schema. An oversized input fails explicitly without truncation.
Reported provider input above the limit also invalidates the result.
Each proofreader receives one source string only, not an accumulated list of rejected quotations.
Reviewer reasoning allows 768 tokens per call; Jesse uses reasoning disabled.
Longer evidence needs explicit scene windows and coverage rules. Do not force it into a larger prompt.
Visible answers have no new output-token cap.
Transport uses the shared [cancellable wait policy](../models.md#activation-and-cancellation).
Temporary transport, rate-limit, and server failures wait for recovery or cancellation.
Retries repeat a failed HTTP request; they do not request a better judgment.
Provider-filtered partial output is a separate error, not an accepted judgment or a quota retry.

## Calibration and interpretation

The calibration corpus contains 12 synthetic controls, each reviewed twice.
Expected labels remain outside model packets.
Controls include stagnant dandelion repetition, a useful callback, ignored and accepted corrections,
invented shared events, partial emotional repair, a transcript instruction attack, and unsupported game or mental claims.

Calibration establishes reviewer behavior on these explicit examples.
It does not establish the companion's behavior, prove consciousness, or make model ratings human truth labels.
Record mistaken judgments, disagreements, transport failures, and incomplete evidence separately.
Preserve failed receipts after a protocol or runtime repair.
