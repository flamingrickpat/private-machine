"""Module-level tool factory for the real Story dialogue module.

The application supplies one lazily-activated model. This factory is called on
the module thread when a run starts, so the model is activated then and the
role receives an async completion tool bound to the already-active client.
"""

from __future__ import annotations

from typing import Callable


def build_story_tools(model) -> Callable[[object, object], dict]:
    """Return the ``tools_factory(environment, batch)`` for Story dialogue."""

    def factory(environment, batch) -> dict:
        complete = model.build_async_tool()
        return {"story_complete": complete, "public_complete": complete}

    return factory


__all__ = ["build_story_tools"]
