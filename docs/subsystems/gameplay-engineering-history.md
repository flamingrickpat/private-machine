# Earlier A13 engineering checks

## Summary

Earlier checks include completed suites and access failures.
Their logs remain unchanged.
Final validation must cover the delivered source and current manual.

Parent: [A13 verification](gameplay-verification.md).

These records describe earlier source revisions. Pending statements apply to those named runs.
The parent page records current acceptance.

## Engineering results and restored access

The earlier foundation run passed 406 tests in 514.072 seconds.
The earlier focused run passed 70 tests in 5.307 seconds.
These results precede the latest captured-escape, service-waiting, and exact-name changes.
The full rerun at `foundation-02.log` has no completed result and must be repeated.

The earlier selected check passed 16 evidence/contract tests in 1.218 seconds and excluded three filesystem-backed A13 tests.
The broader restricted run recorded 79 tests with 43 access errors.
`focused-final.log` preserves that failed result.
Temporary-directory creation succeeded, but child access and cleanup were denied, including under the writable workspace.
Access probes now pass in Temp, the workspace, Git metadata, and the real server directory.
`focused-resumed-01.log` records 80 passing tests in 6.191 seconds.
`foundation-resumed-01.log` records 416 passing tests in 512.145 seconds.
These results precede the latest compact replies and destination geometry guard.
`focused-resumed-02.log` records 82 passing tests in 4.677 seconds after those changes.
`foundation-resumed-02.log` records 418 passing tests in 475.972 seconds after those changes.
`foundation-resumed-03.log` records 421 passing tests in 589.643 seconds after the lava, body-coordinate, and pillar changes.
`focused-resumed-03.log` records 84 passing tests in 4.847 seconds after the lava-edge and body-coordinate changes.
`focused-resumed-05.log` records 86 passing tests in 4.516 seconds after wet-body recovery and specific planning feedback.
`a13-pillar-and-fall.json` retains original placed-block counts, climb counts, and fall observations from the failed native adaptation.
Its regressions retain useful pillar output and exact negative body coordinates.
The final foundation run must cover the final accepted source.

The earlier strict documentation build passed in 63.39 seconds at `mkdocs-01.log`.
The final strict build failed with a temporary-directory access error at `mkdocs-final.log`.
It used an existing cached documentation environment and a separate runtime output directory.
The previous `site/` remains preserved; it does not contain the latest edits.
`mkdocs-resumed-01.log` records a strict build in 42.84 seconds after access restoration.
The regenerated HTML audit found 243 pages, 78,210 local links, and no errors.
That build precedes the latest compact-reply and destination-guard documentation.
`mkdocs-resumed-02.log` records the subsequent strict build.
`.runtime/alpha/a13/browser/receipt.json` retains actual Chromium visits to the index and all three gameplay pages.
The browser also checked the gameplay API anchor and retained screenshots.
Final documentation regeneration remains required.
Git and real-server access now work; behavioral acceptance still governs delivery.

## Latest access interruption

The earlier restored-access statements above describe their named runs.
The source-41 session's workspace profile excluded writes to Git metadata and the real server directory.
It also denies child access and cleanup in newly created test temporary directories.
The source-41 selected run used workspace `TEMP` and `TMP` paths but still recorded 55 errors across 131 tests.
`focused-resumed-41.log` preserves the failed result.
`captured-resumed-41.log` records three passing captured-evidence checks in 2.226 seconds without temporary directories.
These results do not replace the final foundation suite or live M3 acceptance.
`strict-docs-resumed-41.log` records a generated-reference temporary-directory access failure using an existing cached documentation environment.
It used a fresh output path and workspace temporary paths; the earlier `site/` remains preserved.

Foundation run 10b passed 463 tests on source 39.
Run 11 failed with a memory error; run 12 stopped during the access-profile change without a final result.
`foundation-resumed-12.interrupted.json` records that interruption without inventing an exit code or pass.
That session required normal access before final tests, server continuation, and Git delivery.

## Restarted-session checks

The new session permits ordinary writes to Git metadata, the real server directory, and temporary child directories.
Disposable write and cleanup probes passed without ACL changes.
The earlier temporary-child access failures remain real failures of their recorded session.
No Git write or commit was attempted in that earlier restricted profile.

Source 44 passed 473 foundation tests in 481.970 seconds.
Source 48 passed 476 foundation tests in 475.782 seconds with owned game services stopped.
The source-48 strict build passed in 64.22 seconds.
Source 47's HTML audit covered 253 pages and 83,487 links without errors.
Its Chromium review opened eight pages and followed seven API anchors.
Source 49 passed 142 selected checks in 5.081 seconds.
The latest complete foundation suite and final documentation checks must follow the staircase repair.

The Minecraft build and ten isolated contract checks passed after the staircase repair.
Four Node checks passed against captured geometry and descent data, with the positive step explicitly marked as a projection.
They do not replace a repaired live descent or M3 acceptance.
Minecraft's lockfile now matches its existing package name and version; dependency versions remain unchanged.

Source 50 passed 144 selected checks in 5.160 seconds after correcting a test's repository ownership.
The first selected run failed because the root interpreter does not import the separate Minecraft package.
The captured collision check now runs in its owning repository.
Eleven Minecraft contract checks, the TypeScript build, and six Node boundary checks passed.
The full foundation suite and final HTML/browser checks remain pending.
