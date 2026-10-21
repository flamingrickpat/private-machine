# A18B maintenance instructions

Read the root `AGENTS.md` and `docs/work_items/process.md` first.
Use `docs/subsystems/cognitive-evaluation-authoring.md` for new cases.
Use `docs/subsystems/cognitive-review-boundaries.md` for context and evidence rules.

Keep this package a development tool. Product modules must not import it.
Keep actor packets, public editor packets, selected factual evidence, and host-only labels separate.
Never send a full scenario, future beats, architecture, past ratings, or expected labels to a reviewer.
Never reroll a valid low judgment or use reviewer feedback to rewrite a conversation.

Extend cases in groups of four: three non-Minecraft cases and one Minecraft case.
Use real PM, full Spamuel history, real inference, and actual world observations.
Use exact packet projections and source quotations. Retain failed receipts and source identities.
Saved review tests the reviewer. It does not verify changed companion code.
Do not shrink the aggregate denominator when selecting cases.

The owner's 24-hour example describes possible future case volume.
There is no required 24-hour or soak run for alpha or beta.
Do not add such a requirement without a new explicit user instruction.
