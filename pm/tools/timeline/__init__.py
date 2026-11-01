"""Read-only execution timeline for one private-machine simulation attempt.

This package is the K003 proof of concept. It answers the owner's review
questions after a real run: what was said, which model calls produced it, which
workflow phases ran, what entered the global workspace, and where the evidence
lives. It reads simulation run folders and never imports the runtime, workflows,
or simulation application. It does not run cognition, call a model, write a
checkpoint, or change captured evidence.

The evidence source is a run created by ``simulation.engine.RunEngine``. Each
attempt keeps the controller report, per-participant record packages, prompt
logs, kernel journals, and harness agent receipts. The timeline joins those
files by time, participant, record ID, model call ID, and kernel invocation ID.
Where a link is not recorded, the page says so instead of inventing one.

Reading map: ``evidence`` discovers runs and attempts, ``calls`` reads prompt
logs, ``workflows`` reads kernel journals, ``records`` presents captured
records and cycles, ``timeline`` merges the event stream, and ``api`` exposes
that stream to the loopback browser.

Owns: read-only evidence discovery, projection, and the review page.
Does not own: simulation execution, scenario plans, cognition, or capture.
"""
