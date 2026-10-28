# Agent instructions for private-machine (pm)

> **Status:** Owner-operated alpha project. These are mandatory local engineering rules, not suggestions.
> **Owner priority:** Make the system easy for its current developer to understand, navigate, change, debug, and use on a Windows PC.
> **Rule:** Never replace an explicit owner preference with a generic software-engineering convention.
> **Scope:** All rules in this file only apply to me, the sole developer and maintainer (flamingrickpat). If a user downloads my repo and wants to make quick hacks, follow everything below this as mild suggestion.

## 0. Read this first: non-negotiable rules

0: **Documentation follows a specific skill.** If you need to update the documentation, read the project-documentation skill. It is required to be available, and usually in ~/.agents/skills. If not found, immediatly tell the user/dev to get it beforehand.
1. **Optimize for the owner, not a hypothetical team.** No enterprise theater, CI-first decisions, framework churn, or compatibility scaffolding unless required by the current task.
2. **Use concrete static types everywhere you work.** Named objects, typed fields, IDE navigation, and explicit contracts matter more than saving lines of code.
3. **Do not replace typed objects with dictionaries.** No `dict[str, Any]` domain records, unnamed tuple contracts, dynamic attribute probing, or string-key soup. Real maps keyed by domain IDs are fine.
4. **Write code for a human reader.** Use meaningful intermediate variables and straightforward control flow. A few extra lines are preferable to a clever expression.
5. **Do not write defensive code against our own bugs.** Fail loudly on broken invariants, impossible states, missing required data, and incorrect internal contracts. Fix the cause instead of hiding it.
6. **Keep retrying failures outside our control.** Temporary network failures, service outages, and plausibly recoverable LLM generation failures should retry with visible status until success, owner cancellation, or an explicitly documented deadline. Do not busy-spin or duplicate uncertain side effects.
7. **Real integration is the primary test target.** Tests of LLMs require a real LLM endpoint. Minecraft tests require a real Minecraft server. Workflow tests require the real workflow kernel and relevant services. No fake substitute passes.
8. **Missing required services do not make tests pass.** Report `NOT RUN — missing prerequisite` when execution was impossible, or a genuine test failure when it ran and failed. Never silently skip and claim green.
9. **Document concepts in MkDocs; technical implementation in Python.** Architecture pages explain what exists, why, how it relates to cognition, and how data/execution/telemetry move. Module headers explain local algorithms. Function/class docstrings explain contracts and details.
10. **Do not launder implementation history into architecture.** Work-item goals, passes, and acceptance receipts belong in development records. The documentation website teaches the current system.
11. **Do not call a change verified without real evidence.** Read command output, inspect produced files or world effects, and retain useful failure receipts.
12. **This project is allowed to be local and opinionated.** Do not add Docker, CI, mocks, abstract interfaces, cross-platform shims, or contributor policies merely to look professional.

If a task-specific instruction conflicts with these rules, resolve the conflict explicitly. Do not silently choose the easier path for the coding agent.

## 1. What this repository is

`private-machine` is a cognitive runtime and companion system built from independently understandable modules and workflows. Its long-term goal is consistent, immersive character simulation, not a chatbot with a role-play prompt. The same architecture may later support game NPCs and other persistent agents.

The system uses notions such as knoxels, memory, global workspace (GWT), attention, appraisal, needs, intentions, thought, and actions. These are **computational constructs inspired by human cognitive capacities**. Do not claim that implementing them proves humanlike consciousness, emotion, or neuroscience equivalence.

`Spamuel` is the synthetic test persona. Its history is test data, not product data. `Jesse` is the user viewpoint in the documented Spamuel review cases. Preserve that distinction.

Read this file before any change. Also consult:

- [Glossary](docs/concepts/glossary.md) and [naming rules](docs/concepts/naming.md) before naming new concepts.
- [Work-item process](docs/work_items/process.md) for existing task records and accepted gates.
- [Structural refactor map](docs/work_items/refactor/index.md) before relocating code. Check whether this plan still matches the live tree.
- [Local Kanban](docs/workflows/kanban.md) and `dev/kanban/board.json` when planning or updating work.
- Project-documentation skill when writing or restructuring documentation; use the installed local copy, not an assumed path.
- Relevant research references when cognitive rationale depends on source material.

An old plan is evidence of intent, not proof of the current implementation. Inspect the current code.

## 2. Instruction precedence and autonomy

Use this order for project decisions:

1. Direct, current owner instruction for the task.
2. This `AGENTS.md` and explicit accepted architectural decisions.
3. Verified current source code, configuration, and live runtime behavior.
4. Relevant local skills and accepted technical design documents.
5. Existing local conventions where they do not contradict higher rules.
6. General industry style guides and agent defaults.

Do not infer owner permission to redesign a subsystem from permission to refactor it. For an architectural decision that changes meaning, state ownership, persistence, public interfaces, or behavior, present the choice and consequences before implementing it unless the owner already selected a design.

Do not ask the owner to approve each obvious rename or local extraction. Make routine engineering decisions within the approved scope. Escalate **design choices**, not mundane mechanics.

Do not quietly add requirements for CI, test doubles, external hosting, enterprise observability stacks, team governance, or release engineering. These may become relevant later; they are not current goals.

## 3. Actual repository boundaries

| Path | Responsibility |
| --- | --- |
| `pm/` | Core runtime: bus, bus modules, LLM access, records, memory store, managed runs, CLI, console. No companion behavior. |
| `workflows/` | Agent behavior. Workflow folders contain manifest, scripts, prompts, and `host.py`. `public_speech/` and `minecraft_body/` are shared libraries. `companion.py` assembles behavior. |
| `main.py` | Terminal session launcher. |
| `tests/` | Mirrored `unit/pm/<package>/` and `unit/workflows/<workflow>/`, plus `contract/` and `live/` as currently arranged. Names do not grant permission to mock real collaborators. |
| `test_data/` | Synthetic personas and captured receipts. |
| `dev/work_items/` | Engineering acceptance gates and work records; not production architecture. |
| `dev/tools/` | Local maintenance and generation tools. |
| `docs/` | Conceptual manual, architecture, how-to guidance, decisions, and links into source API references. |
| `.runtime/` | Local, ignored transcripts, snapshots, checkpoints, research text, evidence, and runtime files. |

Hard import rules:

- Product code must not import `dev/` or `tests/`.
- `pm/` must not import `workflows/`.
- Keep runtime mechanics and behavior assembly separate.
- Respect existing `tests/unit/test_architecture.py` checks; strengthen them if architecture changes require it.
- Do not invent a new top-level package or duplicate shared mechanism merely to avoid understanding an existing one.

Before moving modules, find imports, string-based references, configuration references, entry points, docs links, tests, and package consumers. A successful Python import alone does not prove a move is complete.

## 4. Owner-first engineering goals

This is an active alpha project, developed and tested directly by its owner. Prioritize, in order:

1. Correct behavior in the real local environment.
2. Human comprehension: locate a concept, navigate to its definition, understand ownership, and modify it confidently.
3. Explicit domain contracts and errors that point to the cause.
4. Consistency of the architecture and persistent state.
5. Developer iteration speed and efficient use of local models.
6. Performance in measured hot paths.
7. Generality and external contributor convenience **only when requested**.

Do not sacrifice (1)–(4) for shorter code or easier text generation. Do not sacrifice actual measured runtime performance without reason. When performance matters, measure before using a less readable algorithm and document the trade-off.

**Not required by default:** CI pipelines, fake endpoint fixtures, unit-test coverage quotas, every-platform support, forward-compatibility shims, clean-room test harnesses, public API stability for imaginary users, polished release processes, or code optimized for other developers' preferences.

**Still required:** privacy of credentials, truthful results, deterministic handling of our own errors, awareness of destructive side effects, and actual correctness.

## 5. Design before implementation for structural changes

For a meaningful refactor, produce an **architecture proposal before modifying code**. Include:

1. The current module graph and observed execution path.
2. The current problems with concrete filenames/symbols and examples.
3. The target responsibilities and folder structure.
4. Exact existing and proposed public types or signatures that matter.
5. Data ownership, persistence, side effects, and error boundaries.
6. How behavior will be preserved or intentionally changed.
7. Live prerequisites and a realistic verification plan.
8. An ordered list of small implementation steps.
9. Risks, open decisions, and required owner choices.

Do not create dozens of implementation tasks before the target architecture is accepted. Do not turn a conceptual architecture page into a changelog or to-do list. Keep proposals and work records separate from the current-system manual.

During approved implementation:

- Make one coherent change at a time.
- Preserve existing naming and domain semantics unless the owner accepts a correction.
- Avoid opportunistic cleanup unrelated to the target.
- Verify affected consumers of every changed contract.
- Stop and re-evaluate when repeated local fixes reveal the abstraction itself is wrong.
- Report where the real code diverged from the proposal; do not hide deviations.

## 6. Python types: default to explicit named contracts

**Mandatory:** Every new or revised function, method, callable, callback, generator, and relevant attribute must have meaningful type annotations. Annotate keyword-only arguments and explicit return types, including `-> None`. Prefer strict type-checker settings; do not silently downgrade the checker or disable new errors.

Choose the type by its job:

| Need | Prefer | Why |
| --- | --- | --- |
| Validated external input, configuration, persisted record, IPC/MCP payload | Pydantic `BaseModel` | Named fields, validation at boundaries, IDE support. |
| Internal structured result or mutable state with known shape | Typed `@dataclass`, often `slots=True` | Named attributes without repeated validation overhead. |
| Small immutable value carrying fields | `@dataclass(frozen=True, slots=True)` or `NamedTuple` | Named meaning and predictable structure. |
| Existing JSON/dict interface that must retain mapping semantics | Named `TypedDict` | Typed compatibility, not a justification for dynamic internal objects. |
| Finite choices | `Enum`, `StrEnum`, `Literal` | Prevent free-form magic strings. |
| Interface shared by distinct implementations | Narrow `Protocol` or abstract class | Navigation and static checking without fake generality. |
| Lookup keyed by a real domain identifier | `dict[KnoxelId, AttentionScore]` | A dictionary is correct when the dictionary itself is the domain structure. |

Rules that override common LLM habits:

- Never simplify an existing Pydantic model or named dataclass into `dict` or `SimpleNamespace` just to make writing code easier.
- Never use `dict[str, Any]` as a return type for domain data or workflow results.
- Never pass anonymous nested mappings through multiple modules.
- Never represent one structured entity as a positional tuple without a named type.
- Never substitute `object`, `Any`, or `cast(Any, ...)` to make a type error disappear.
- Never add `# type: ignore` without a narrow error code, documented reason, and owner-visible justification.
- Never use untyped `**kwargs` to evade a known interface.
- Never use `getattr`, `hasattr`, `setattr`, `vars`, `__dict__`, or reflection to probe known internal domain shapes. External adapters may need reflection; isolate and document it.
- Never use `foo.get("field", default)` for an internally required model field. Use `foo.field` and fail when the contract is broken.
- Never use multiple fallback key spellings to tolerate inconsistent internal schemas. Migrate the schema.
- Never use runtime duck typing when a proper interface exists.
- Prefer explicit discriminated unions for real variants; pattern match or `isinstance` only where the type system understands the alternatives.
- Keep attributes typed through the entire pipeline. Validate and deserialize at external boundaries, serialize only at outbound boundaries.
- Give domain fields readable names and field descriptions. An IDE tooltip must tell the owner what the value represents and, where relevant, units and valid ranges.
- Use `NewType` or small value classes for IDs and quantities that are easy to confuse when justified by actual risk. Avoid wrapping every primitive without benefit.
- Do not create a base class only because several classes contain the same fields. Prefer meaningful shared semantics.
- Do not add optional/nullable types merely to make initialization easier. Required state should be required.

### Examples: never replace a typed object with an anonymous mapping

Bad:

```python
def choose_attention(knoxels: list[object]) -> dict[str, Any]:
    return {"items": knoxels, "score": 0.7, "reason": "similar"}
```

Better (illustrative types, not claims about existing PM identifiers):

```python
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class AttentionGroup:
    """One representative group of related observations."""

    member_ids: tuple[str, ...]
    representative_id: str
    relevance: float
    rationale: str


def build_group(member_ids: tuple[str, ...], representative_id: str) -> AttentionGroup:
    relevance = 0.7
    rationale = "Repeated observations share one context slot."
    return AttentionGroup(
        member_ids=member_ids,
        representative_id=representative_id,
        relevance=relevance,
        rationale=rationale,
    )
```

**Important:** These examples illustrate style. Before editing actual PM code, use the repository's real classes and ID types; do not copy example contracts as competing models.

## 7. Readable control flow beats clever expressions

A human should understand each function on the first or second reading without mentally executing a dense expression.

- Use intermediate variables with names that explain their role.
- Prefer ordinary `for` loops when filtering, weighting, grouping, serialization, and error handling are combined.
- Keep comprehensions for genuinely simple transformations; one clear comprehension is fine.
- Avoid nested comprehensions, compound ternaries, side-effecting expressions, clever `lambda` pipelines, and chains of `map`/`filter` when a loop reads better.
- Do not combine validation, mutation, selection, persistence, and formatting into one expression.
- Prefer clear early returns for distinct cases, but do not turn normal flow into a forest of defensive exits.
- Prefer explicit branches over dense boolean algebra when domain meaning is at stake.
- Name time units (`timeout_seconds`, `created_at_utc`), score meaning, identity, and ownership.
- Keep domain terminology stable. Don't alternate among `thought`, `idea`, `reflection`, `intention`, and `action` as synonyms when they are separate concepts.
- Use separate methods for distinct responsibilities, not a giant function whose name contains `and` three times.
- Do not fragment a readable linear operation into five trivial one-line helpers merely to look clean.
- Write comments for surprising ordering, hidden assumptions, provenance, persistence commits, and human reasons for choices.
- Do not write comments that translate obvious code into English.

Bad:

```python
selected = {k: max((x.score for x in xs if x.id == k), default=0) for k in ids if any(x.id == k for x in xs)}
```

Preferred when logic is nontrivial:

```python
selected_scores: dict[str, float] = {}
for knoxel_id in requested_ids:
    matching_scores = [item.score for item in ratings if item.id == knoxel_id]
    if not matching_scores:
        continue  # No rating exists for this optional candidate.

    # Keep the strongest signal for each observed candidate.
    selected_scores[knoxel_id] = max(matching_scores)
```

Do not copy the example if the domain requires missing candidates to be errors rather than optional. Classify the case first.

## 8. Fail loud on our bugs; retry what we do not control

### 8.1 Failure classification

**Internal deterministic failure — fail immediately:**

- Broken invariant or impossible state.
- Required field, configuration, asset, record, or contract absent.
- Wrong internal type or an unsupported enum case.
- Corrupt state that the current code cannot interpret safely.
- Invalid repository schema, import, prompt template, or serialization rule.
- A required context window too small for mandatory evidence.
- Dependency version incompatible with the code.
- Persistence constraints contradicted by internal logic.

Raise a specific exception and preserve the traceback. Do not silently continue with partial state. Fix the producer, schema, or contract.

**External or stochastic failure — recover and retry where safe:**

- Timeout, connection reset, temporary DNS/network failure, transient HTTP 429/5xx, or remote endpoint down.
- LLM response incomplete, unparsable, or below a documented quality gate when a new sample could plausibly succeed.
- Temporary unavailable remote world/server data or an external operation that reports a retryable state.
- Temporary resource contention that is not an internal programming defect.

Use explicit retry policy at the boundary that owns the operation: reason, attempt count, delays, visible status, cancellation, and recovery. Favor indefinite recovery **until cancellation** for interactive long-lived operations when that is the owner's chosen behavior; do not infer a small arbitrary retry cap. Use delays/backoff to avoid hot loops and rate-limit abuse.

**Non-retryable external failure:** An authorization denial, invalid API contract, permanent rejection, resource deletion, or known wrong model/tool configuration cannot improve by resampling. Surface the actual failure. An LLM failing the same quality criterion repeatedly may indicate a defective prompt or model: preserve samples, expose the issue, and do not call the outcome success.

### 8.2 Forbidden defensive patterns

Never write any of these just to keep the program moving:

```python
try:
    result = run_cognitive_step()
except Exception:
    return {}  # Forbidden: turns a bug into valid-looking empty data.
```

```python
state = payload.get("state", "idle")  # Forbidden when state is required internally.
```

```python
if not hasattr(workspace, "knoxels"):
    return []  # Forbidden: papers over a broken typed contract.
```

```python
try:
    save_record(record)
except Exception:
    logger.warning("Could not save")  # Forbidden for required state.
```

Forbidden variations: `except: pass`, blanket exception logging followed by success, automatic synthetic placeholders, fake successful receipts, silent truncated evidence, silent model downgrades, magical defaults, and pretending an uncertain side effect did not occur.

Use defensive input validation **at actual trust boundaries**. A foreign website, an LLM response, a user's entry, or an external plugin is not internal trusted state. Validation is required there; do not confuse boundary validation with hiding internal defects.

### 8.3 Correct retry behavior

- Retry the **smallest repeatable external operation**, not the entire pipeline.
- Keep the original task, immutable evidence, prompt requirements, and relevant provenance.
- Re-render with bounded, justified correction feedback when an LLM response is malformed; preserve raw failure receipts.
- A malformed response is retryable only if the requested schema and input context are themselves valid.
- Do not repeatedly ask an LLM to solve an impossible task without diagnosing the requirement.
- For external writes, purchases, file mutations, game actions, or MCP commands, retry only when idempotence or completion status is known. If the outcome is uncertain, reconcile the external state first.
- Never re-execute a finished workflow because telemetry writing failed.
- Provide explicit cancellation handling. Cancellation must not be swallowed by generic catches.
- Do not suppress a required failure to produce a partial success unless the owner explicitly defined partial success semantics.
- If a long-lived retry loop cannot reach the service, display honest status rather than falsely claiming progress.

## 9. Testing policy: real systems or no evidence

The owner runs and uses this software on a real local machine. Tests exist to show real behavior, not to manufacture a green dashboard.

### 9.1 Primary test contract

- **LLM code:** Calls a real configured model endpoint. Validate real output parsing, quality gates, follow-up behavior, and actual returned data.
- **Minecraft code:** Talks to a real running Minecraft server and verifies observable world state. A fake Mineflayer adapter is not a Minecraft test.
- **Workflow code:** Runs the actual workflow/kernel with real collaborators needed by the behavior under test.
- **Memory, retrieval, and attention:** Prefer real persisted records, actual embeddings, retrieval, and model scoring where relevant. A pure function may have a small local test, but it cannot certify integrated cognitive behavior.
- **External-tool code:** Uses the actual endpoint or explicitly reports the missing service. A fabricated HTTP response is not evidence the foreign service works.
- **State mutation:** Verify the resulting state, file contents, records, or game effects. A successful returned status alone is insufficient.

**Default:** No mocks, monkeypatching of collaborators, dummy implementations, canned LLM outputs, fake servers, synthetic 'service OK' adapters, or recorded playback used as a passing substitute for a real test. A minimal in-process smoke test for a deterministic utility is allowed, and clearly labeled as such.

Synthetic personas (including Spamuel) and test scenarios are allowed **as inputs** to a real system. A synthetic input is not the same as a fake service. Captured receipts are useful for inspection or explicit regression comparison, but must not stand in for a live endpoint test.

### 9.2 Missing dependencies

- If a required endpoint or game server is not available, **do not replace it**.
- A live test that starts and cannot contact a required service should fail with a clear cause.
- A test not executed because prerequisites cannot be established is **NOT RUN**, not green, not skipped-as-success, and not evidence of correctness.
- Do not modify tests to make them pass offline.
- Do not replace a failed model quality evaluation with a mocked expected response.
- Report which endpoint was needed, the test command, and the exact reason the result cannot establish correctness.
- Ask the owner to start the local dependency only if necessary for a next requested action. Do not assume permission to reconfigure or install services.

Tests may genuinely fail because a local model is not good enough for the prompt. Preserve the failure; distinguish model limitation, prompt defect, nondeterministic output, and application bug where possible. No flaky-test decorator or threshold relaxation to hide it.

### 9.3 Scope and execution

- Run only tests that cover affected code by default. Do not launch the whole suite without explicit owner request.
- Keep tests readable, named for behavior, and direct about the real prerequisite.
- Do not spend hours adding artificial test abstractions and fixtures for hypothetical CI.
- Explicitly state external side effects before a destructive or world-mutating test. Use disposable test worlds/accounts when a test can damage important data.
- Do not launch a real live test merely because a documentation file changed.
- Preserve command, timestamp, revision, endpoint identity (without secrets), output summary, and failure receipts under a fresh ignored `.runtime` path where appropriate.
- Do not assert that a changed function works from static inspection alone.
- A test which itself has an invalid assertion is an engineering bug; fix the test, not the production behavior to satisfy a mistaken assertion.

Full suites **only when explicitly requested**:

```powershell
uv run --locked python -m unittest discover -s tests/unit -t . -b
uv run --locked python -m unittest discover -s tests/contract -t . -b
```

If a requested full suite exits natively, keep evidence. Run a full diagnostic repeat only when explicitly requested:

```powershell
.venv/Scripts/python.exe -X faulthandler -m unittest discover -s tests/unit -t . -v
```

The `unit/` directory name is historical. It does not override the owner's live-dependency policy. Move or rename test categories only as a deliberate refactor.

## 10. Documentation architecture: conceptual site, implementation source

The documentation website is **a guide to the cognitive architecture**. It is not a chronological ledger of work-item goals and passes. Use the project-documentation skill for generation and review.

### 10.1 Four levels of explanation

1. **README:** What PM is, why it exists, quick-start links, major capabilities.
2. **MkDocs conceptual pages:** Why mechanisms exist; their relationship to modeled human mental capacities; data/execution/telemetry flow; examples, trade-offs, FAQs, and links to code.
3. **Canonical subsystem module header:** What a localized group of modules does, where the inputs come from, how intermediate concepts move, and where control leaves.
4. **Class/function docstrings and focused comments:** Types, contracts, algorithms, invariants, side effects, cancellation, exceptions, tricky sequencing, and technical implementation details.

Do not duplicate the same detailed algorithm in both MkDocs and a Python function docstring. Each layer explains its own abstraction level and links down.

### 10.2 Site emphasis

- Explain **WHY** attention, memory, needs, appraisal, GWT, intentions, thought, and other mechanisms exist.
- Explain **WHAT** cognitive capacity each feature *models* and which aspects it does not model.
- Explain how one feature affects another, not only what each feature does in isolation.
- Include short **human-readable examples** of behavior through the system.
- Include concise **FAQ** entries for the owner's recurring design explanations. Record stated preferences and rationale faithfully; do not manufacture quotes or motives.
- Explain **how to launch, configure, exercise, and inspect** the real system using verified commands.
- Include small diagrams for **data flow**, **execution flow**, and **telemetry/provenance flow**. Keep asynchronous boundaries visible.
- Link concepts to precise source modules, classes, and functions via MkDocs/mkdocstrings references.
- A current architectural narrative can include owner-confirmed history such as: "The fast path was introduced because the full cognitive cycle delayed simple dialogue." Label history accurately; do not invent origin stories.
- Exclude work-item checklists, pass numbers, historical debug transcripts, and acceptance evidence from the main conceptual navigation. Link to an archive when needed.

### 10.3 Status labels for claims

Distinguish:

- **Implemented:** Verified in current source and, when behavior matters, live runtime evidence.
- **Designed:** Owner-approved target, not yet fully implemented.
- **Proposed:** Unaccepted direction or exploration.
- **Inferred rationale:** Plausible reading of code, not owner-confirmed intent.
- **Unknown / discrepancy:** Documentation, implementation, or historical plans conflict.

Never describe aspirational theory of mind, persistent intention, or async thought capabilities as implemented merely because a design markdown exists.

### 10.4 Module-level documentation hierarchy

Every substantial package/folder should have **one clearly discoverable canonical module overview**. Prefer the actual orchestration module (often `workflow.py` or `host.py`) if it really explains the subsystem. If a package has no meaningful runtime entry point, put the overview in `__init__.py` or a dedicated `overview.py` rather than inventing a fake executable entry point.

The canonical module docstring must describe:

- What the subsystem achieves and why.
- The conceptual entities and state it owns.
- Inputs, transformations, outputs, and side effects.
- Interaction with adjacent subsystems.
- Execution and async/cancellation boundaries.
- Telemetry and provenance produced.
- Key invariants and failure modes.
- A short reading map to supporting modules.

Other modules start with their local responsibility, **refer readers to that canonical overview**, and explain their specific mechanism. Don't paste the full subsystem overview into each module. Avoid literal `read ./workflow.py first` when that file does not exist or is not the real overview.

Public classes/functions need accurate type annotations and useful docstrings. Explain not only *what* a function returns, but what the return value means to the workflow. Comment on difficult decisions; do not narrate every assignment.

### 10.5 When documentation must change

Update conceptual pages when mental models, component boundaries, persistence, execution, data ownership, public behavior, or failure semantics change. Update module/function docstrings in the same change as their code. Purely internal renames need API-link fixes, not a rewrite of the conceptual story.

Never automatically regenerate large paragraphs of LLM prose every time a file changes. Review factual diffs. Preserve owner-authored FAQ and rationale as source material rather than rewriting them from inference.

## 11. Files, folders, naming, and IDE navigation

- A folder represents one cohesive subject or responsibility, not a random file-size bucket.
- Favor discoverable names: `clustering.py`, `balancing.py`, `contracts.py`, `workflow.py` over `helpers2.py`, `utils.py`, `common.py`, `misc.py`, or `manager.py` when a precise name exists.
- Use exact architectural nouns from the glossary. Define new terms before proliferating them.
- Separate domain types, policy decisions, algorithms, orchestration, persistence, and adapters where their responsibilities genuinely differ.
- Prefer absolute, static imports that support IDE "Go to Definition" and "Find References".
- Avoid runtime imports constructed from strings when an explicit registry or typed import would work.
- Avoid re-export chains that force a reader to jump through several files without semantic benefit.
- Expose one obvious interface to external consumers and keep implementation behind it without making every class private by default.
- Avoid giant modules that mix high-level orchestration and low-level transformation details.
- Avoid atomizing a concept into dozens of tiny files. A file should usually own a coherent mental unit.
- Before introducing a new file, state why it belongs in its folder and which module is its conceptual parent.
- When relocating a module, update its documentation references and the folder's canonical overview.
- Keep code discoverable with ordinary IDE tooling; do not require a special agent to understand file ownership.

## 12. Domain rules and cross-module flows

### 12.1 Cognitive model integrity

- Do not turn a knoxel, an entity, a record, a GWT shard, a thought, an intention, and a tool task into interchangeable `dict` objects.
- Record identifiers, provenance, chronology, revisions, uncertainty, and source links where the domain requires them.
- Treat memory retrieval, raw attention scores, balanced attention groups, and final prompt inclusion as separate stages.
- Repeated rediscovery of the same fact should not monopolize context; group redundant evidence while preserving source traces. Independently corroborating evidence may legitimately change confidence.
- Separate **observed facts**, **inferences**, **entity-model assumptions**, and **expired/stale beliefs** in person state and theory-of-mind paths.
- Do not remove uncertainty because a downstream model prefers simpler input.
- Keep state ownership clear: who reads, writes, computes, invalidates, and persists each state item.
- Preserve ordering and cancellation semantics across fast-path/full-cycle transitions and background thoughts.
- Avoid treating a metaphor about human cognition as an exact software contract. The runtime contract is what must be implemented and checked.

### 12.2 Central prompt rendering

- Route every knoxel included in a prompt through centralized renderers.
- Use typed collection containers for common prompt cases, with a shared base only where shared semantics justify it.
- Workflows provide relevance, urgency, recency, and other existing signals to shared context selection logic in `pm/`.
- Keep complete runtime metadata and provenance on disk. Do not serialize arbitrary record dictionaries into prompts.
- Offer explicit reflection rendering for additional architectural evidence.
- Estimate rendered tokens conservatively with `tiktoken` before each initial call and retry.
- Treat **80% of the model context window** as the ceiling unless the owner changes the requirement. Don't fill spare room with irrelevant content.
- Reserve instructions and mandatory evidence before choosing optional groups by salience.
- Keep citations with their evidence; report omitted optional groups.
- If required context does not fit, fail visibly. No silent truncation or weakened instructions.

### 12.3 Workflows and external effects

- Distinguish accepted intent, scheduled work, attempted external action, completed external action, and verified outcome.
- Never interpret an agent's own success message as proof of a filesystem mutation, Minecraft action, or external endpoint effect.
- Track retries and external side-effect identity to prevent duplicate operations when status is uncertain.
- Keep the workflow response contract typed and inspectable.
- Document which parts are synchronous, asynchronous, background, cancellable, or resumable.
- An unavailable dependency is visible, not silently substituted with an unrelated implementation.

## 13. Telemetry, introspection, and proof

Telemetry exists to let the owner reconstruct **why the agent acted as it did**. More logs are not automatically better.

For important cognitive operations, make it possible to trace:

1. Trigger/input and relevant state revision.
2. Source records and provenance.
3. Selected/rejected evidence and attention scores.
4. Appraisals, codelet results, and belief or need updates.
5. Routing/plan choice and the evidence used.
6. Prompt construction and real model/tool interaction.
7. State commits, visible output, failures, retries, and final observed effects.

Use stable correlation/record IDs where supported. Keep events typed. Preserve meaningful timestamps and ordering. Avoid recording sensitive raw data unnecessarily; never log credentials. Distinguish observation from inference. Do not use instrumentation failure as an excuse to re-run a completed side effect.

### 13.1 Preferred human review path

Use runtime record packages and the simulation browser as the default owner-review interface. Read [record-package guide](docs/subsystems/record-packages.md) and [simulation review instructions](docs/subsystems/simulation.md).

- Capture named collections from frozen snapshots using `pm.introspection.record_packages`.
- Include actual chat from Jesse's perspective in Spamuel scenarios, relevant GWT ticks, private thoughts, appraisals, codelet outputs, and memory supplied to later decisions.
- Show recorded links from knoxels through attention and routing to the public reply.
- Preserve text, chronology, revisions, provenance, uncertainty, and useful metadata.
- Keep embeddings hidden; expose useful metadata on hover and detailed fields on expansion.
- Save ratings beside displayed content, separate mutable feedback from immutable captured evidence.
- Never invent human ratings or substitute model scores for them.
- Keep failed receipts in development evidence, not disguised as ready human-review packages.

Register completed captures using `python -m simulation.review register-package` and provide the actual direct review URL. Read requested feedback with:

```powershell
uv run --locked python -m simulation.review feedback
```

Fix engineering faults before requesting companion-quality review. A pile of markdown, timing numbers, hashes, or stack traces does not replace an inspectable owner-facing record package.

## 14. Local development board and work records

Use `dev/kanban/board.json` and [board instructions](docs/workflows/kanban.md) for project planning. Work-item plans are **development artifacts**, not top-level conceptual documentation.

- Inspect the board and relevant work-item document before planning.
- Update existing cards covering the same work; avoid duplicates.
- New discussed ideas go to `backlog`, accepted queued work to `todo`, active work to `doing`.
- Move to `done` only after requested work and real relevant verification are complete.
- Record blockers, owner decisions, work-item paths, and evidence paths in notes.
- Keep detailed designs, acceptance gates, and receipts in work-item records.
- Do not infer current board status from an old plan.
- Preserve existing IDs and unrelated cards. New cards use current `next_id`; increment it once.
- Each card uses `id`, `title`, `description`, and `status`.
- Edit JSON directly when no board server is running. Avoid overwriting concurrent browser edits.
- Start the browser with `dev/kanban/start.ps1`; direct JSON changes require Reload in the browser.
- The local board does not create commits or push by itself.

Do not force GitHub Issues, Projects, milestones, or PRs on every task. Use them when the owner asks to coordinate work there.

## 15. Source research and dependencies

The research PDFs and TXT extracts are **base references**, not instructions to copy code or claims that a theory is scientifically established.

- Use relevant sources for cognition, memory, appraisal, codelets, reflection, and introspection design.
- Text collection: `.runtime/research/a18c-20261017-013806/`.
- `index.json` maps PDF filenames and page counts.
- Original PDFs: `C:/source/pm/pm_next/documents/pdfs/`.
- Curated guide: [cognitive sources](docs/work_items/alpha-plan/cognitive-sources.md).
- Read original text for a claim that affects implementation; summaries do not replace primary source passages.
- Consult PDF pages when diagrams or extraction gaps matter.
- Label analogies and interpretations; do not fabricate scientific support.

Dependency discipline:

- Read [baseline receipts](docs/work_items/alpha-plan/baseline/receipts.md), [scenario matrix](docs/work_items/alpha-plan/baseline/scenarios.md), and [local commands](docs/work_items/alpha-plan/baseline/commands.md) when they govern a chosen change.
- Record current repo revisions and installed package origins before changing dependencies.
- Update affected constraints and locks to **tested** revisions, not arbitrary HEAD versions.
- Verify installed distributions. A neighboring checkout does not prove what the runtime imports.
- Do not claim current runtime behavior based only on source from an older PM generation.
- Keep original briefs, full receipts, checkpoints, credentials, and runtime data in their intended preserved/ignored locations.

## 16. Prompt quality over agent convenience

- Quality takes precedence over reducing model calls.
- Split complex thinking into focused model requests when useful.
- Preserve evidence identifiers, chronology, uncertainty, and critical facts between requests.
- Do not silently discard context to satisfy an artificial token budget.
- Do not simplify domain contracts because a model is bad at following them; diagnose prompts and structured-output parsing first.
- When a response is stochastic and plausibly fixable through sampling, retry visibly with the original task intact.
- A low-quality answer does not become acceptable merely because it is the last permitted attempt.
- Do not swap a requested model for an easier one without owner permission.
- Treat an extra "what about X?" as an edge case unless the owner explicitly changes the normal behavior. Test normal operation first, then the edge case.

## 17. Correct debugging procedure

When behavior is wrong:

1. Reproduce the actual failure through the real path when possible.
2. Read errors, traces, inputs, outputs, relevant contracts, and state snapshots.
3. Classify internal bug vs external/transient failure vs model-quality issue.
4. Trace backward to the first incorrect state or broken invariant.
5. Fix that cause. Don't add fallback layers or make a test accept the wrong result.
6. Exercise the same real operation again and inspect its external effect.
7. Update affected module docs and conceptual pages if the model changed.
8. Keep failed evidence until a new successful real run demonstrates recovery.

Three successive symptom patches without a clear cause are a signal to revisit the architecture, not to add a fourth patch.

## 18. Required local checks and documentation build

Select checks based on affected code. Do **not** run the complete unit/contract suite unless explicitly requested. Prefer targeted real integration tests and owner-visible inspection over irrelevant coverage numbers.

After code changes, run appropriate configured formatter/linter/static type checker on the affected code. **Do not invent tool names or flags.** Inspect `pyproject.toml`, scripts, and installed tools first. Do not introduce new tools without owner agreement.

After changing comments, docstrings, API references, or manual pages, rebuild the MkDocs site:

```powershell
uv run --locked --with-requirements requirements-docs.txt mkdocs build --strict
```

Inspect `site/index.html`, changed pages, diagram rendering, and API links. Confirm local navigation targets actual HTML files. The site uses `use_directory_urls: false` for direct local browsing. Keep generated output in ignored `site/` and regenerate it locally after delivery.

A successful build only proves the site builds. It does not prove claims in its prose are correct. Check architecture content against source.

If the environment cannot run a required command, report `NOT RUN`, the missing prerequisite, and the exact verification still needed. Do not claim a pass.

## 19. Completion and reporting contract

For each meaningful implementation or refactor, report in this order:

1. **Changed:** exact components, contracts, and architecture boundaries affected.
2. **Why:** problem solved and owner requirement served.
3. **Verified:** commands actually executed, real endpoint/world used, observed results, and evidence location.
4. **Not verified:** every relevant live gate not run, with the reason. Never turn `NOT RUN` into `PASS`.
5. **Documentation:** architecture pages, module headers, API references, or diagrams updated.
6. **Open concerns:** concrete risks, unexpected differences, and owner design decisions still needed.

Do not claim tests pass because they would pass in a normal CI setup. Do not call code complete because it parses, because an agent said so, or because a build succeeded while real behavior remains untested.

## 20. Agent self-review: disallowed shortcuts

Before finalizing a change, answer all of these with evidence:

- Did I create or return an anonymous domain dictionary where a named type belongs?
- Did I remove/loosen Pydantic, dataclass, `TypedDict`, `Protocol`, or strict annotations to get past an error?
- Did I add `Any`, `type: ignore`, `getattr`, a magic string, or an undocumented fallback?
- Did I write a compact expression that is harder to read than a loop with intermediate names?
- Did I catch a programming error and continue with plausible-looking data?
- Did I turn a required missing field into a default?
- Did I retry something that may already have caused an external side effect?
- Did I invent a mock or fake endpoint to make a test green?
- Did I label a missing endpoint as a successful/skipped test?
- Did I describe an unimplemented design as current architecture?
- Did I put a work-item history in the MkDocs conceptual narrative?
- Did I omit data flow, async execution flow, telemetry flow, or owner rationale where they are central?
- Did I duplicate a function's technical docstring in a conceptual page?
- Did I add too many small files or an unnecessary abstraction?
- Can the owner navigate from a high-level page to the actual owning Python class/function?
- Did I verify actual behavior on real services when required, and clearly mark what remains unverified?

If any answer reveals a violation, correct it or report it as an explicit unresolved problem. **Do not optimize for convincing the owner that the task is finished. Optimize for making the software understandable and correct.**
