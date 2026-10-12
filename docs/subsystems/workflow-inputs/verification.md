# A03 input verification

## Summary

A03 checks scoped capture on real host files, the memory filesystem, and PM's Ghost projection.
The live gate uses preserved application logs and an original Spamuel memory.
Actual pm-coder inference copies bound evidence into a verified target file.
These checks establish input binding, rather than classification quality or repaired companion behavior.

Parent: [scoped inputs](index.md).
Contracts: [bindings](bindings.md) and [versions](versions.md).
Earlier evidence: [A01 receipts](../../decisions/alpha-plan/baseline/receipts.md).

## Reproduce the checks

Run these commands in `C:/source/pm/pm-workflows`:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_inputs.py -q
.venv/Scripts/python.exe -m pytest -q
```

Run these commands in `C:/source/pm/pm_next_v2`:

```powershell
uv sync --locked
uv run --locked python -m unittest tests.test_workflow_inputs tests.test_workflow_runner tests.test_agent_target -v -b
$testModules = Get-ChildItem tests/test_*.py | ForEach-Object { 'tests.' + $_.BaseName }
uv run --locked python -m unittest $testModules -b
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
uv run --locked python dev/work_items/alpha_input_gate.py --config config.alpha.example.yaml --out ".runtime/alpha/$stamp-inputs"
uv run --locked --with-requirements requirements-docs.txt mkdocs build --strict
```

The live gate requires chat on `http://127.0.0.1:8080/v1`, alias `qwen`.
It discovers the active model and serving context before inference.
Embeddings, Laya, and Minecraft are unnecessary for this input gate.
The run directory must be fresh and remain under `.runtime`.
[`dev.work_items.alpha_input_gate.main`][] retains manifests, source identities, listings, captures, agent journals, and actual result files.

## Observed checks

The final package suite passed 169 tests in 67.03 seconds.
It includes the existing real agent target and cancellation checks.
The application adapter and runner subset passed 17 tests in 0.693 seconds.
The root foundation suite passed 261 tests in 493.985 seconds against the final pinned package.
This adds four application input tests to the 257-test baseline.
The live gate passed three-candidate listings on both targets, two stale-change rejections, and the actual agent report comparison.
Strict MkDocs passed. Chromium verified all four input guides, the A03 page, and their generated API links.
Evidence remains under `C:/source/pm/pm_next_v2/.runtime/alpha/a03/`.

The package checks cover these boundaries:

- Actual host and memory listing, metadata, and file reads.
- Missing data, empty folders, invalid metadata, malformed values, and incorrect types.
- A file change after listing and a change during capture.
- Replacement with identical content under host stat identity.
- Exact-folder and role grants, including a real Windows directory junction.
- JSON pointers, request fields, literal data, and dynamic report paths.
- Exact named producer attempts, failed latest attempts, and digest integrity.
- Loop restart rejection and distinct foreach child requests.
- Cancellation before dispatch and separate consumer/evidence copies.
- Legacy manifests, recursive host globs, and virtual references that cannot read a host decoy.

## Application evidence

The live gate prepares three copies of actual earlier logs inside one scoped folder.
They come from A01 foundation, model-policy foundation, and model-policy documentation runs.
Host copies retain actual source modification times.
Ghost copies retain their contents but leave times unknown.
The listing contains stable candidate IDs, byte sizes, revisions, digests, and null producer metadata.

Original Windows logs use UTF-16 or the local legacy code page, `cp1252`.
The first gate exposed that encoding and failed before inference.
The repaired gate decodes the original logs and writes only UTF-8 test copies.
`sources.json` records the original encoding, source hash, converted hash, and original modification time.
The source logs remain unchanged.
`live-inputs/` and `live-inputs-verified/` preserve both encoding failures and their logs.
`live-inputs-final/` and its log preserve the subsequent gate.
`live-inputs-delivery/` repeats the passing gate against the final pinned package revision.

Each environment captures a listed foundation log, changes only that test copy, and rejects the stale listing.
The chosen filename is deterministic gate setup. It does not establish inference-based selection.
A04 still owns that decision and its invalid-evidence behavior.

The agent environment contains the complete real Spamuel Ghost projection.
The delivered gate projected 11,202 original records and used memory ID `3` as its exact input.
The gate selects one original memory by its recorded ID for explicit input binding.
It also captures A01's real control-worker report and checkpoint run state.
The independent facts are a completed run receipt and an empty report answer.
The agent copies those facts and the exact memory text into `outbox/report.yaml`.
The gate reads that actual file, compares the mapping with the original sources, and checks the unchanged memory and seed.
The final gate passed both stale-change assertions and the actual agent report comparison.
This result does not establish that the original worker wrote the requested note.

## Inspect and diagnose

1. Open `sources.json` and `identity.json` in the final run directory.
2. Open `host-listing.json` and `ghost-listing.json`.
3. Compare candidate identity, time availability, and listing revision.
4. Open both capture receipts and both stale-rejection receipts.
5. Open the generated `bound-input.workflow.md`.
6. Open `runtime/kernel/a03/journal.jsonl` and inspect `input_evidence` and `named_outputs`.
7. Open `agent-result.json` and compare the actual report with its independent expected mapping.
8. Open the pm-coder session messages and verify the actual file-tool call.

A03 delivered `pm-workflows` 0.6.0 at `95f2842d0e14a2a2c70c41514e0e36055cac73e4`.
The [A05 delivery](pm-verification.md) records the current application pin after A04 classification.
The root constraint and lock select that exact commit.
pm-coder remains pinned to `ac8327baf669dad54ea9f91580f2bed1023825b4`.
No sibling coder implementation or model deployment changes occurred.

## Remaining acceptance

[A04 verification](classification-verification.md) establishes selection, classification, INVALID routing, and scoped coding-agent recovery.
A03 preserves pm-coder compaction but does not itself force overflow.
Direct chat still sends one supplied message list.
Model consumers must construct suitable context and reject incomplete results.

C01 remains open: a completed control worker produced no useful public reply and did not establish exact note contents.
C02 remains open: Sunny claimed codeword loss despite retained records.
A08 owns original evidence retrieval. A09 owns grounded public claims and a real agent-written note with its public reply.
The historical routing holdout and review status remain unchanged.
No Laya classifier became active, and no Minecraft reset or gameplay test occurred.
