"""Tool factory of the memory reconsolidation workflow.

The tool closes over the live memory store, the application's stage model,
and its embedding service. The supervisor calls the factory once for each
run. The tool itself runs the whole reconsolidation in the run's worker
thread; its commits use the normal store path.
"""
from __future__ import annotations

from collections.abc import Callable
from threading import Event

from pm.memory.reconsolidation import reconsolidate


def build_reconsolidation_tools(store, stage_model, embeddings) -> Callable[[str, object], dict]:
    """Return the ``tools_factory(run_id, environment)`` of the workflow.

    ``stage_model`` and ``embeddings`` are None in dummy cognition. The tool
    then raises ValueError, because reconsolidation needs real models.
    """

    def factory(run_id, environment) -> dict:
        def reconsolidate_memory(*, cancel: Event | None = None, log=None) -> dict:
            if stage_model is None or embeddings is None:
                raise ValueError("Memory reconsolidation needs the real stage model and embeddings.")
            report = reconsolidate(store, complete=stage_model.complete, embeddings=embeddings, force=True,
                                   cancel=cancel, log=log)
            return report.model_dump(mode="json")
        return {"reconsolidate_memory": reconsolidate_memory}

    return factory


__all__ = ["build_reconsolidation_tools"]
