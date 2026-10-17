# Development review desk

Run this from the repository root:

```powershell
uv run --locked python dev/review/server.py serve
```

The script opens `http://127.0.0.1:8765` in your browser.
Stop it with Ctrl+C in the launch terminal.
Use `serve --port 8766` if the default port is occupied.
Use `serve --no-browser` to print the address without opening a browser.

This service has no runtime imports, workflow controls, model calls, or documentation-site integration.
It reads completed review bundles and writes separate feedback files.
Code and documentation references open as read-only text previews.
It never reads a companion checkpoint.

## Review a run

1. Choose a run from the result register.
2. Read the current prompt and captured reply on the left.
3. Expand the earlier conversation or open a source reference when necessary.
4. Enter ratings and comments on the right.
5. Press **Save draft** or **Save review & next**.

Ratings range from 1, poor, to 5, excellent.
Leave a rating empty when it does not apply.
Naturalness, self-explanation, immersion, usefulness, and groundedness are independent fields.
Regeneration preference, unsupported claims, and intended deception are separate fields.
A missing reply stays visible but cannot receive a rating.

The page warns before leaving unsaved feedback.
Drafts remain separate from submitted reviews.
Refresh reloads completed runs and saved feedback.
The register includes both reviewed and unreviewed cases.
Your ratings do not change test outcomes or original evidence.

## Store and read feedback

The tracked `dev/review/registry.json` records bundle ids, paths, and digests.
Captured content stays under `.runtime/dev-review/bundles/`.
Human feedback stays under `.runtime/dev-review/feedback/`.
Keep those ignored files when moving or backing up this workspace.
A clone with only the register shows unavailable captures instead of inventing them.

After review, tell the agent **check my feedback**.
The agent reads saved feedback with this command:

```powershell
uv run --locked python dev/review/server.py feedback
```

Each save records its revision, UTC timestamp, optional reviewer, and prior versions.
Atomic replacement keeps the previous feedback file intact until the new file is complete.
The server retries Windows replacement locks.
One HTTP server serializes saves.
A stale browser tab receives a conflict instead of overwriting newer feedback.
Do not run two review servers against the same feedback directory.

## Register a completed run

After a work item, run the real tests and demonstrations first.
Keep their receipts, requests, and actual outputs under an ignored evidence directory.
Then import a completed A17, A18, or console receipt:

```powershell
uv run --locked python dev/review/server.py import-receipt `
    .runtime/alpha/YOUR-RUN/receipt.json `
    --work-item A18 --title "A18 - self-model review"
```

The A18 matched-persona receipt is also supported.
The importer copies review content into a bundle and records the original receipt digest.
It preserves failures and keeps unavailable replies distinct.
It does not infer human ratings from model judgments or gate results.
Gate receipts group replies by arrival order; that does not establish exchange correlation.
The importer uses captured gate source for scripted prompts when available.
It labels the current-source fallback when a historical run has no gate source snapshot.

Commit the register with the work-item changes and tell the user which run needs review.
The browser reads captured bundles, so a review never needs to rerun inference.
The runtime does not register results automatically.
Future development scripts can call this command after their own evidence is complete.

## Register another result format

The runtime can export `pm.record-package.v1` collections through `pm.introspection.record_packages`.
Each package contains existing serialized records and a `records`, `chat`, or `gwt` display hint.
Register package files directly:

```powershell
uv run --locked python dev/review/server.py register-package .runtime/debug/chat.json .runtime/debug/gwt.json --id YOUR-REVIEW-ID
```

The desk formats content and shows useful metadata on hover.
Expand **Record metadata** for every captured field. Embeddings stay hidden.
Each imported file receives its own review item, so separate snapshots retain their own versions.
The engineering manual explains [record packages](../../docs/subsystems/record-packages.md).

A19 also supplies an offline cognitive capture:

```powershell
uv run --locked python dev/review/cognitive_trace.py .runtime/alpha/YOUR-A19-REVIEW --id YOUR-REVIEW-ID
uv run --locked python dev/review/server.py register .runtime/alpha/YOUR-A19-REVIEW/review-bundle.json
```

The source directory must contain a completed `receipt.json` and its `ghost.json`.
The exporter rejects runtime errors, missing speech, and unfinished user exchanges.
It copies actual chat messages, accepted private thoughts, codelet candidates, attention, and routing records.
The browser provides **Chatlog**, **Thoughts & routing**, and **Knoxel contents** views.
Open a knoxel to read its exact text, recorded sources, and consuming records.
These links describe recorded dependencies. They do not establish that an interpretation is correct.
The capture includes the supplied scenario belief and identifies it as setup.
Human ratings remain empty until the reviewer saves feedback.

Write a bundle JSON file under `.runtime`, using this schema:

```json
{
  "schema": "pm.dev-review.bundle.v1",
  "id": "unique-run-id",
  "work_item": "A19",
  "title": "A short review title",
  "summary": "What ran, what is observed, and what needs a human judgment.",
  "created_at": "2026-10-15T12:00:00+00:00",
  "items": [
    {
      "id": "case-id",
      "title": "One situation",
      "persona": "Persona name",
      "scenario": "scenario-id",
      "context": [{"role": "User", "text": "The actual input"}],
      "content": "The actual generated output",
      "reviewable": true,
      "notes": ["Concise evidence limits or mechanical results"],
      "references": [{"label": "Captured receipt", "path": ".runtime/YOUR-RUN/receipt.json"}]
    }
  ]
}
```

The text in this schema illustrates fields. It is not a registered result or test evidence.

```powershell
uv run --locked python dev/review/server.py register .runtime/YOUR-RUN/review-bundle.json
```

Use a new id for each changed result.
The receipt importer accepts `--id NEW-ID` for a new context selection of an older run.
Re-registering identical content retains feedback.
Changing a registered bundle's content is rejected.
If a new capture replaces an older review package, set the older register row's `superseded_by` to the new bundle id.
The desk opens the replacement and keeps the older capture and feedback on disk.
References must point inside the repository to evidence, code, tests, or documentation.
The server exposes only references attached to a registered item.
It refuses live `ghost.json` checkpoints and environment files.
Reference text cannot execute in the browser.

For isolated verification, put global options before the command:

```powershell
uv run --locked python dev/review/server.py `
    --registry .runtime/dev-review/check/registry.json `
    --data .runtime/dev-review/check/data serve --port 8766 --no-browser
```

Automated save probes use that separate store.
They must not create human ratings in the real review queue.
