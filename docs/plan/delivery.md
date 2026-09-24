# Commits, packages, and GitHub delivery

## Summary

Implementation agents can change the required repositories and push tested changes directly to their working branches.
Publish dependencies first. Then update this application from GitHub with uv, test it, and push its lockfile.
One application commit identifies the dependency commits that the home computer installs.

Parent: [implementation handoff](../../IMPLEMENTATION_PLAN.md).
Related: [work items](work-items.md) and [tests](tests.md).

## Repositories and authority

| Repository | Local directory | Working branch |
| --- | --- | --- |
| [pm_next_v2](https://github.com/flamingrickpat/pm_next_v2) | `C:\source\pm\pm_next_v2` | `master` |
| [pm-workflows](https://github.com/flamingrickpat/pm-workflows) | `C:\source\pm\pm-workflows` | `main` |
| [pm-coder](https://github.com/flamingrickpat/pm-coder) | `C:\source\pm\pm-coder` | `main` |

The application already uses `master`. Do not rename it as part of an implementation item.
The dependency projects use `main`. The earlier `steward` branch requirement is superseded.
The user authorizes required code changes, version updates, commits, and ordinary pushes in these projects.
The same permission covers a directly required change in another user-owned PM dependency.
Before that change, inspect its instructions, remote, branch, and working tree.
This permission does not cover unrelated projects or the running Hadal environment.

No pull request, PyPI upload, or version tag is required.
Do not force-push, rewrite published history, or include unrelated user changes.
Do not change repository visibility, branch protection, or GitHub credentials.

## One application environment

The application uses Python 3.12 and the repository-local `.venv`.
`.python-version` selects Python 3.12. `pyproject.toml` restricts the supported interpreter range.
Use `uv sync --locked` to reproduce the environment.
Use `uv run --locked` for commands instead of a hardcoded old interpreter path.
Do not create `.venv312` or set `UV_PROJECT_ENVIRONMENT` to a second project environment.

The old `.venv` contained Python 3.13 and no installed packages.
The old `.venv312` contained the working Python 3.12 environment with editable sibling packages.
The delivery cleanup replaces both with a fresh `.venv` built from the GitHub lockfile.
The retired environments remain recoverable outside the repository:

```text
C:\Temp\pm-next-v2-retired-environments-20260924\.venv312
C:\Temp\pm-next-v2-retired-environments-20260924\.venv-313-retired
```

The second directory contains the old Python 3.13 environment.
These are recovery copies, not supported execution environments at their new paths.
The tool blocked permanent deletion, so the cleanup moved both directories instead.
Virtual environments are disposable. Source, fixtures, secrets, and memory do not belong inside them.
An IDE interpreter must point to `.venv\Scripts\python.exe` on Windows.

## Dependency source and lockfile

`pyproject.toml` names GitHub sources at `main` for both PM dependencies.
The current entries use `rev = "main"` to match the transitive dependency reference in `pm-workflows`.
Mixing `branch = "main"` with a transitive `@main` reference caused a uv URL conflict during setup.
Keep the source spelling consistent across these requirements.

`uv.lock` records the exact resolved Git commits and package versions.
Commit it with every dependency update.
An ordinary sync preserves those commits instead of silently advancing to a newer `main`.
An explicit package upgrade resolves a newer commit. [uv locking documentation](https://docs.astral.sh/uv/concepts/projects/sync/)

Final application validation must use installed GitHub packages, not editable sibling checkouts.
Do not commit local paths or temporary source overrides in `pyproject.toml`.
Do not put sibling repositories on `PYTHONPATH` to make a failing integration pass.
The imported dependency files must come from `.venv\Lib\site-packages` on Windows.

Dependency development can use each dependency's own environment and source checkout.
Only the final application integration must use the pushed Git commits.
Do not install packages into system Python or refresh an environment used by an unrelated active workflow.

## Version rules

Each project owns its version in `pyproject.toml`.
Update that version in the same commit as a runtime or public API change.
Use a patch increase for compatible fixes and dependency-only application updates.
Use a minor increase for a new capability.
Before version 1.0, a breaking change also requires a minor increase and a clear migration note.
Documentation-only commits do not require a version increase.

Run `uv lock` after a metadata change in any repository that tracks its lockfile.
Do not edit generated lockfile entries by hand.
Do not change an unrelated third-party pin without a reason and relevant tests.

## Release sequence for one work item

1. Inspect Git status and remote URLs in each affected repository.
2. Fetch the remote branches.
3. Fast-forward a clean checkout before starting new changes.
4. Preserve existing user changes and identify any overlap.
5. Add the specified tests and implement the change in its owning repository.
6. Update that package's version and lockfile.
7. Run its affected tests and inspect the diff for credentials and unrelated files.
8. Commit and push the dependency to `main`.
9. Update and test the next dependent repository against that GitHub version.
10. Commit and push that repository before updating the application.
11. Update application metadata, then refresh its dependency lock from GitHub.
12. Synchronize `.venv` and run the required application tests.
13. Record documentation and evidence for this work item.
14. Commit and push the application to `master`.
15. Compare local HEAD with the remote branch and report the commit links.

The usual dependency order is `pm-coder` -> `pm-workflows` -> `pm_next_v2`.
An unchanged repository needs no artificial release commit.
A documentation-only application item needs no dependency upgrade.
If the remote advances, fetch and integrate without a force-push.
If unrelated changes prevent a safe integration, report the exact conflict.

## Commands

From a changed dependency repository, after its version update:

```powershell
uv lock
uv run --locked --extra dev python -m pytest -q
git diff --check
```

Use targeted test paths during development. Use the required affected suite before publication.
For a workflow release that needs a new coder version, refresh that dependency after its push:

```powershell
uv lock --upgrade-package pm-coder
uv run --locked --extra dev python -m pytest -q
```

From `pm_next_v2`, after the required dependency pushes:

```powershell
uv lock --upgrade-package pm-coder --upgrade-package pm-workflows
uv sync --locked
uv run --locked python -m unittest discover -s tests -v -b
git diff --check
```

Stage only the files for the work item.
Include `pyproject.toml` and `uv.lock` when they change.
Use explicit normal pushes: `git push origin main` for dependencies, and `git push origin master` for this application.

## Commit contents and remote review

Prefer one coherent checkpoint per work item in each affected repository.
Smaller tested commits are also valid. Do not accumulate the whole implementation into one unexplained commit.

A commit subject identifies the result, such as `feat(ghost): persist atomic memory batches`.
Its body records the work-item ID, behavior change, test command and result, and remaining limitations.
An application dependency update also names the dependency versions and commit SHAs.
This lets the user inspect progress through GitHub without access to the working computer.

Keep large transcripts and runtime artifacts outside Git unless the task explicitly requires a fixture.
Record a concise result summary and the artifact location in the handoff.
Never commit `.env`, API keys, virtual environments, runtime folders, or unreviewed private logs.
Required live tests must pass before a feature is declared complete.
Tests of the current dummy prototype do not establish real cognitive acceptance.

## Planned commit checkpoints

Use these subjects as a guide. Each row requires its work item's acceptance gate.
Split a large row into smaller tested checkpoints when necessary.
Choose version numbers from the current repository state, using the version rules above.

| Item | Repository and suggested commit subject |
| --- | --- |
| W0 | Application: `chore(baseline): record tested GitHub dependency revisions` |
| W1 | Application: `feat(models): add model presets and compatible embedding cache` |
| W2 | Application: `feat(core): add isolated bus delivery and durable Ghost commits` |
| W3a | Coder: `feat(sessions): isolate agent state and enforce working target permissions` |
| W3b | Workflows: `feat(runtime): propagate scoped targets, MCP grants, and cancellation` |
| W3c | Application: `feat(runtime): integrate the published workflow runner` |
| W4 | Application: `feat(dialogue): add standalone Story generation and ontology checks` |
| W5 | Application: `feat(cognition): wire tested cognitive stages to Story dialogue` |
| W6 | Application: `feat(management): add Assistant tools and managed workflow control` |
| W7 | Application: `test(acceptance): verify live cognition and extension contracts` |
| W8 | Application: `feat(minecraft): add a managed body workflow and memory consolidation` |

For W5, commit each tested child workflow before the final wiring checkpoint.
For W6 or W8, publish any required dependency changes before the application commit.
W0 does not need an empty commit when the existing baseline evidence remains accurate.
A failing live test means the feature is incomplete, even when a smaller infrastructure checkpoint passes.

## Home computer

For an existing authenticated checkout with no conflicting local edits:

```powershell
git pull --ff-only
uv sync --locked
uv run --locked python main.py
```

For a new computer, clone the private application repository first.
Authenticate Git to GitHub through its credential mechanism, not a token inside a repository URL.
No sibling checkout is required to run the application.
The computer still needs the configured model endpoints and its own local secret config for live cognition.
Git does not synchronize model files, API keys, or the user's embedding cache.
Transfer the compatible embedding cache separately to avoid recomputing cached text on that computer.

For the current dummy real-folder demonstration, select the working Git Bash before running `demo.py`.
The README contains that command.

## Bootstrap releases

The delivery setup publishes these dependency versions:

- `pm-coder` 0.7.0, commit `810a79fe223419be2d28461fb03c80f1ba8947ed`.
- `pm-workflows` 0.3.0, commit `d68022f809dcb4ab1f1a021b2c47035ab5e85846`.

The application bootstrap version is 0.1.1.
Its lockfile is the source of truth for the installed dependency commits.

Validation completed on 2026-09-24:

- Coder: 30 BashMachine tests passed before publication.
- Workflows: 55 runtime-context, driver, and kernel tests passed before publication.
- Application: 34 tests passed in the fresh `.venv` with GitHub-installed dependencies.
- Demonstration: 21 checks passed with Git Bash first on PATH.
- Dependency check: `uv pip check` found all 100 installed packages compatible.

Both PM dependency imports resolved inside the application's `.venv\Lib\site-packages`.
Their installed Git commit metadata matched the published commits above.
These checks validate the dummy prototype and delivery setup. They make no live model calls.
