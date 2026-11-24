"""Scheduling contract supplied to the public-speech library.

See :mod:`workflows.public_speech.public_dialogue` for the publication path.
The assembly supplies the control owner; speech never imports a behavior host.
"""
from __future__ import annotations

from threading import Event
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from workflows.main_cognition.schemas import ActionRequest


class ReplySuperseded(RuntimeError):
    """Required reconsideration made a draft or selected action obsolete."""


def validate_unmanaged_selection(selection_id: str) -> None:
    """Legacy actions have no ordinary selection owner to validate."""
    if selection_id:
        raise RuntimeError("An ordinary selection ID requires its injected selection owner.")


def validate_unmanaged_action(request: ActionRequest) -> None:
    """A stale selected action requires its injected behavior owner."""
    if request.knoxel_id is not None or request.interaction_selection_id:
        raise RuntimeError("Public action refresh requires its injected behavior owner.")


class PublicationControl(Protocol):
    """The scheduling and commit checks owned by cognitive control."""

    def wait_for_reconsideration(self, cancel: Event, stop_source: int | None = None) -> bool: ...
    def wait_for_classification(self, cancel: Event) -> bool: ...
    def publication_epoch(self, stop_source: int | None = None) -> int: ...
    def validate_publication(self, epoch: int | None, guidance_tick: int | None, stop_source: int | None = None) -> None: ...
    def validate_selection(self, selection_id: str) -> None: ...
    def refresh_selection(self, selection_id: str) -> None: ...
    def refresh_action(self, request: ActionRequest) -> None: ...
