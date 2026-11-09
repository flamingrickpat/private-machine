"""Simulation harness: real private-machine participants in scripted scenarios.

This package runs whole companions as separate real processes and records what
they do, so the owner can judge companion behavior from evidence. It implements
no companion cognition or companion memory mechanisms. Harness model calls
plan scenes, write grounded user messages, and review delivered evidence.
Companion cognition stays inside the ordinary application, with no fixed replies.

Two runtimes live here, and they must not be confused:

* The controller (``engine``, ``server``) owns runs. It forks persona homes into
  a run folder, keeps the coordinated controller document, schedules scenario
  stages, routes speech between participants, and captures evidence. ``server``
  exposes that controller to the browser; ``plans`` and ``agents``
  produce and check the scenario script with the configured default model.
  ``actors`` freezes complete public history and actual identities from the
  selected source. Actor generation sees current circumstances, not future plot.
* The participant (``participant``, ``client``) hosts one persona.
  ``participant`` runs one assembled application behind the shared public and
  control APIs; ``client`` owns that subprocess and speaks to it.

Durable run artifacts (``state.json``, ``plan.json``, ``attempts/<n>/``,
``checkpoints/``, ``actors/``) are the source of truth. Actor profiles remain
fixed across attempts. Browser views are detached copies.
Owner feedback is stored beside captured evidence, never inside it.

``templates`` owns the versioned browser and CLI input. Daily planning turns
director guidance into an editable exact plan. ``analysis`` rates completed
public conversation through independent evaluator calls and retains citations.
Rating failure does not rerun or invalidate completed companion execution.
The server mounts the reusable ``pm.tools.timeline`` debugger on the same origin.

Shared invariants: input and routing receipts are durable before delivery; a
restart resumes from the last coordinated checkpoint; a run that ends without
delivery evidence is a failure, not a completed scenario.

Reading map: ``engine.RunEngine`` is the run controller,
``participant`` hosts one companion, ``pm.introspection.record_format`` presents
captured records, and ``review.catalog`` freezes a capture for human review.
``templates.SimulationTemplate`` is the portable input contract;
``analysis.AnalysisSettings`` owns editable automatic rating criteria.
"""
