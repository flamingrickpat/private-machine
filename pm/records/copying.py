"""Explicit pickle copies for state that a caller must change in isolation."""
from __future__ import annotations

import pickle
from typing import TypeVar

T = TypeVar("T")


def pickle_copy(value: T) -> T:
    """Copy one value with pickle. Unsupported values raise the pickle error."""
    return pickle.loads(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
