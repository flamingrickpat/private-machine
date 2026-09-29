"""Expose the dialogue_story contract tests to the root unittest discovery.

The tests live beside the workflow (``workflows/dialogue_story/tests``). The
root suite runs ``discover -s tests``, so this module imports those classes
into a discovered module without duplicating their code.
"""
from workflows.dialogue_story.tests.test_story_contract import (  # noqa: F401
    CleanReplyTests,
    ContextSelectionTests,
    KernelContractTests,
    MessageContractTests,
    OntologyTests,
)

__all__ = [
    "CleanReplyTests",
    "ContextSelectionTests",
    "KernelContractTests",
    "MessageContractTests",
    "OntologyTests",
]
