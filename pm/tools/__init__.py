"""Runtime inspection tools that a maintainer starts on purpose.

``pm.tools.inspector`` is the memory inspector: a read-only web view over one
character checkpoint. It loads the store, never saves it, and starts no
cognition, no bus, no model call, and no embedder. See the module docstring of
``pm.tools.inspector`` for the views it serves.
"""
