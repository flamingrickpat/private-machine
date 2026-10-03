---
name: classification_demo
driver: {kind: python}
human_resolver: {mode: forbid}
failure_policy: {max_attempts_per_phase: 2}
roles:
  inspector:
    skill: workflows/classification_demo/inspect.md
    readable_paths: ["logs/**"]
  publish:
    skill: workflows/classification_demo/steps.py
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
    input_folder: "${request.artifact_folder}"
    data_selection: Select the latest foundation test log by its UTC filename and available metadata. Ignore documentation logs.
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

# PM artifact classification

Supply `artifact_folder` with a granted relative path below `logs`.
PM selects the latest foundation report and classifies its captured contents.
INVALID invokes the isolated inspector once, then publishes the observed result.
A completed publication can contain INVALID. It does not establish passing tests.
