# Write one requested note

Read inbox/request.yaml.
Use the supplied note_spec.path and note_spec.text exactly.
Write that text as UTF-8 bytes, without a byte-order mark or an extra newline.
Keep all work inside this task directory.
Do not change inbox files or write a task verification report.
Inspect the resulting file before you finish.
Do not call subagents.
Return only {"status": "done", "summary": "description of the actual write"}.
