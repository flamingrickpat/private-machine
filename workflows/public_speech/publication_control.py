"""Typed scheduling callbacks supplied to the public-speech library.

See :mod:`workflows.public_speech.public_dialogue` for the publication path.
The assembly supplies these callbacks; speech never imports a behavior host.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import Event


class ReplySuperseded(RuntimeError):
    """Required reconsideration made a draft or selected action obsolete."""


def validate_unmanaged_selection(selection_id: str) -> None:
    """Legacy actions have no ordinary selection owner to validate."""
    if selection_id:
        raise RuntimeError("An ordinary selection ID requires its injected selection owner.")


@dataclass(frozen=True, slots=True)
class PublicationControl:
    """Wait and validate against the state owner's critical-input fence.

    ``validate_publication`` receives the draft epoch and accepted guidance
    tick. Its caller holds the memory-store lock through durable publication.
    """

    wait_for_reconsideration: Callable[[Event], bool]
    wait_for_classification: Callable[[Event], bool]
    publication_epoch: Callable[[], int]
    validate_publication: Callable[[int | None, int | None], None]
    validate_selection: Callable[[str], None] = validate_unmanaged_selection
