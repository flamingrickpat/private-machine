"""Simulation harness: real private-machine participants in scripted scenarios.

This package runs whole companions as separate real processes and records what
they do, so the owner can judge companion behavior from evidence. It implements
no cognition, prompts, or memory; every model call happens inside the ordinary
application. It supplies no simulated cognition and no fixed companion replies.

Two runtimes live here, and they must not be confused:

* The controller (``engine``, ``server``) owns runs. It forks persona homes into
  a run folder, keeps the coordinated controller document, schedules scenario
  stages, routes speech between participants, and captures evidence. ``server``
  exposes that controller to the loopback browser; ``plans`` and ``agents``
  produce and check the scenario script with the configured default model.
* The participant (``participant``, ``client``) hosts one persona.
  ``participant`` runs one assembled application behind the shared public and
  control APIs; ``client`` owns that subprocess and speaks to it.

Durable run artifacts (``state.json``, ``plan.json``, ``attempts/<n>/``,
``checkpoints/``) are the source of truth. Browser views are detached copies.
Owner feedback is stored beside captured evidence, never inside it.

Shared invariants: input and routing receipts are durable before delivery; a
restart resumes from the last coordinated checkpoint; a run that ends without
delivery evidence is a failure, not a completed scenario.

Reading map: ``engine.RunEngine`` is the run controller,
``participant`` hosts one companion, ``review.record_format`` presents captured
records, and ``review.catalog`` freezes a capture for human review.
"""
