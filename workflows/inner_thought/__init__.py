"""The private thought loop: the companion's second conscious workflow.

A17 replaced the fixed dummy thought with this package. The loop runs on its
own timer, chooses a thought operation from a deterministic trigger policy,
and commits its accepted work into its own GWT shard (slot ``inner``). The
main loop sees accepted thoughts as self-origin events on the shared bus.
"""
