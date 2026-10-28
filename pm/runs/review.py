"""Shared passing judgments; each workflow selects its own review mode.

A workflow owns its required checks and its commit. This module only names what
counts as an acceptable judgment of evidence, so two workflows cannot drift on
the meaning of ``good enough``. Strict mode accepts one judgment; flexible mode
also accepts a sufficient result or an equivalent improvement.
"""
from typing import Literal, TypedDict

ReviewMode = Literal["strict", "flexible"]
PASSING_REVIEWS = frozenset({"accepted", "good_enough", "equivalent_or_better"})


class ReviewPolicy(TypedDict):
    """The judgments a reviewer may use to pass, for one review mode.

    ``mode`` is the configured :data:`ReviewMode`, and
    ``passing_judgments`` holds the accepted judgment names in sorted order.
    """

    mode: ReviewMode
    passing_judgments: list[str]


def review_passes(judgment: str, mode: ReviewMode = "flexible") -> bool:
    """Accept a permitted judgment without changing evidence or task state.

    Strict mode accepts only ``accepted``. Flexible mode also accepts a
    sufficient result or an equivalent improvement. The workflow still owns
    required checks, external effects, cancellation, and its durable commit.
    """
    return judgment == "accepted" or mode == "flexible" and judgment in PASSING_REVIEWS


def review_policy(mode: ReviewMode = "flexible") -> ReviewPolicy:
    """Describe the workflow's permitted passing judgments to its reviewer."""
    return {"mode": mode, "passing_judgments": sorted(PASSING_REVIEWS if mode == "flexible" else {"accepted"})}
