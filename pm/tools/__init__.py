"""Runtime inspection tools that a maintainer starts on purpose.

``pm.tools.inspector`` is the memory inspector: a read-only web view over one
character checkpoint. It loads the store, never saves it, and starts no
cognition, no bus, no model call, and no embedder.

``pm.tools.timeline`` is the K003 execution timeline: a read-only web view over
one simulation attempt's conversation, model calls, workflow phases, GWT cycles,
and captured records. It reads evidence files and starts no runtime process.

Each tool has its own module overview. They share the serialized-record
presentation in ``pm.introspection.record_format``.
"""
