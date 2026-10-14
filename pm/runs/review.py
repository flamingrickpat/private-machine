"""Shared passing judgments; each workflow selects its own review mode."""
from typing import Literal

ReviewMode = Literal["strict", "flexible"]
PASSING_REVIEWS = frozenset({"accepted", "good_enough", "equivalent_or_better"})


def review_passes(judgment: str, mode: ReviewMode = "flexible") -> bool:
    """Accept a permitted judgment without changing evidence or task state.

    Strict mode accepts only ``accepted``. Flexible mode also accepts a
    sufficient result or an equivalent improvement. The workflow still owns
    required checks, external effects, cancellation, and its durable commit.
    """
    return judgment == "accepted" or mode == "flexible" and judgment in PASSING_REVIEWS


def review_policy(mode: ReviewMode = "flexible") -> dict:
    """Describe the workflow's permitted passing judgments to its reviewer."""
    return {"mode": mode, "passing_judgments": sorted(PASSING_REVIEWS if mode == "flexible" else {"accepted"})}
