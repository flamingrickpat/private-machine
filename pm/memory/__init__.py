"""Accepted memory: record ownership, causal history, retrieval, and read grants.

``pm.memory`` owns the accepted knoxels of one persona and every operation that
reads or replaces them. Nothing in this package decides what to think;
workflows make that decision and hand the store a typed ``WriteBatch``.

What the package owns
---------------------
- Live membership and its indexes (``record_index``).
- Atomic acceptance of one batch and its causal transaction (``store``,
  ``write_batch``, ``causal``).
- Durable state: pickle checkpoints (``checkpoint``) and the legacy YAML/JSON
  seed codec (``yaml_codec``).
- Frozen read views that outlive the store lock, and the reads they record
  (``snapshot``, ``read_scope``).
- Ranking and evidence attribution over a frozen view (``retrieval``,
  ``evidence``, ``references``, ``vector_index``).
- Model-facing read tools behind one explicit grant (``agent_tools``,
  ``store_tools``, ``bash_view``).
- Context planning over already-ranked evidence (``context_plan``,
  ``temporal_context``).
- Derived clusters, facts, and narratives, committed through normal batches
  (``reconsolidation``).

Boundaries
----------
No model or network call runs under the commit lock. Embedding work happens in
``embeddings`` before ``MemoryStore.commit`` takes the lock; retrieval ranks
stored vectors outside it. Reads share record instances instead of copying
their values, so any text that reaches a prompt goes through the central
renderers in ``pm.context_render``.

Invariants
----------
- A record ID is never reused, and a replacement increments that record's
  revision.
- A rejected proposal, a revision conflict, or a failed checkpoint write
  leaves accepted memory and its causal history unchanged.
- Cancellation is explicit (``GhostCancelled``) and never becomes a partial
  commit or a hidden fallback.

Reading map
-----------
Start with :mod:`pm.memory.store` for ownership and commit semantics, then
:mod:`pm.memory.retrieval` for how a frozen view becomes ranked evidence, then
:mod:`pm.memory.agent_tools` for the grant that a workflow hands to a model.
"""
