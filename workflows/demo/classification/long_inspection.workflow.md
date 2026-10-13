---
name: classification_long_inspection
driver: {kind: python}
human_resolver: {mode: forbid}
failure_policy: {max_attempts_per_phase: 2}
roles:
  inspector:
    skill: workflows/demo/classification/inspect_long.md
    readable_paths: ["logs/**"]
  publish:
    skill: workflows/demo/classification/steps.py
    result_contract:
      type: json
      schema:
        status: {enum: [done]}
        summary: string
phases:
  - name: check
    kind: classify
    backend: llm
    connection: Default
    input_file: logs/overflow.log
    classification_limits: {max_input_tokens: 512, timeout_seconds: 900}
    instruction: Classify only the current completed test run's final test summary. Missing, contradictory, or incomplete test summaries require INVALID.
    result_contract:
      type: classes
      enum:
        all_good: The current completed test run ran at least one test and all tests passed.
        some_failed: The current completed test run reports at least one passed test and at least one failed test.
        all_failed: The current completed test run explicitly reports that every test failed.
    on_class: {all_good: publish, some_failed: publish, all_failed: publish, INVALID: inspect}
  - name: inspect
    kind: classify
    fallback_for: check
    role: inspector
    on_class: {all_good: publish, some_failed: publish, all_failed: publish, INVALID: publish}
  - name: publish
    kind: role
    role: publish
    on_status: {done: stop}
    on_invalid: {action: stop}
---

# Long inspection through PM

This acceptance workflow forces one full read of a captured long report.
Its direct budget requires isolated recovery through the unchanged pm-coder loop.
