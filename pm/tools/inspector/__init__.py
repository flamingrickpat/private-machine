"""Inspect one character checkpoint in a browser, without running the character.

Start it from the app launcher with ``--inspector``, or on its own with
``python -m pm.tools.inspector --home <folder>``.

The package has four parts:

- :mod:`pm.tools.inspector.model` loads a checkpoint into lookup tables:
  records, kinds, subtypes, relations, cluster hierarchy, and shard history.
- :mod:`pm.tools.inspector.rendering` shows how the cycle renders a record, a
  workspace shard, and a whole cycle, and what each line costs. It calls the
  same renderers that the cycle calls, so what you read is what the model got.
- :mod:`pm.tools.inspector.api` and :mod:`pm.tools.inspector.server` answer
  the page. Both are read-only.
- :mod:`pm.tools.inspector.launch` loads the store and starts the server.

Nothing in this package runs cognition, a bus cycle, a workflow, a model
endpoint, or an embedder. Nothing saves the store.
"""
from pm.tools.inspector.launch import main, open_index, run_inspector, serve

__all__ = ["main", "open_index", "run_inspector", "serve"]
