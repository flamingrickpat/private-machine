"""Vector-shaped need and cognition axes with bounded arithmetic helpers.

owns: ``NeedsAxesModel`` (eight supply and drive axes), ``CognitionAxesModel``
(four axes that change how the companion thinks), and the ``DecayableMentalState``
helpers that add, scale, decay, and differ such a vector. Every field is one
float axis in [-1, 1] or [0, 1], and every helper clamps the result to the
field bounds that the model declares.

not own: ``FullMentalState`` and the EMA state that the running cycle actually
uses (``pm.records.mental_state``), the appraisal that moves an axis, or
persistence. This module is a reserved second needs model and no runtime
module imports it yet; the ``pm.records.base`` module docstring gives the
package overview.

Invariants: a field bound comes only from its ``Field(ge=..., le=...)``
metadata, and an axis value always stays inside that bound after an operation.
A helper that receives a value it cannot convert to a number leaves that value
untouched instead of inventing a default.
"""
from __future__ import annotations

import logging
import math
from typing import Any

from annotated_types import Ge, Interval, Le
from pydantic import BaseModel, Field, model_validator
from pydantic.fields import FieldInfo

logger = logging.getLogger(__name__)


def numeric_bounds(model_field: FieldInfo) -> tuple[float, float]:
    """Return the inclusive ``(lower, upper)`` bounds that a field declares.

    An unbounded side is ``-math.inf`` or ``math.inf``. The bounds come from the
    ``ge`` and ``le`` metadata that ``Field(ge=..., le=...)`` attaches to a
    field, so a new axis gets clamping without extra bookkeeping. A bound that
    is not a number is ignored, because every helper here clamps numbers only.
    """
    lower = -math.inf
    upper = math.inf
    for constraint in model_field.metadata:
        if isinstance(constraint, Ge):
            declared_lower = constraint.ge
            if isinstance(declared_lower, int | float):
                lower = float(declared_lower)
        elif isinstance(constraint, Le):
            declared_upper = constraint.le
            if isinstance(declared_upper, int | float):
                upper = float(declared_upper)
        elif isinstance(constraint, Interval):
            # Interval carries both bounds in one metadata object.
            interval_lower = constraint.ge
            interval_upper = constraint.le
            if isinstance(interval_lower, int | float):
                lower = float(interval_lower)
            if isinstance(interval_upper, int | float):
                upper = float(interval_upper)
    return lower, upper


def clamp_to_bounds(value: float, lower: float, upper: float) -> float:
    """Clamp one axis value to its bounds.

    A NaN value stays NaN, so field validation rejects it instead of the helper
    quietly turning it into a bound.
    """
    clamped_low = max(value, lower)
    return min(clamped_low, upper)


class ClampedModel(BaseModel):
    """A model whose numeric input is clamped to the bounds of each field.

    Every float-convertible value that a caller supplies for a field is replaced
    by the clamped float before normal validation runs, so an out-of-range axis
    value never reaches the field validators.
    """

    @model_validator(mode="before")
    @classmethod
    def _clamp_numeric_fields(cls, data: Any) -> Any:
        # The validator receives unvalidated caller input, so its type is only
        # known after the mapping check below.
        if not isinstance(data, dict):
            return data
        for field_name, model_field in cls.model_fields.items():
            if field_name not in data:
                continue
            lower, upper = numeric_bounds(model_field)
            try:
                numeric_value = float(data[field_name])
            except (TypeError, ValueError):
                continue  # Not a number, so the field validators own the error.
            data[field_name] = clamp_to_bounds(numeric_value, lower, upper)
        return data


class DecayableMentalState(ClampedModel):
    """A mental state vector whose axes decay toward a baseline and combine.

    The models below keep one float per axis in a fixed field order. The helpers
    return new instances (``__add__``, ``get_delta``) or change this instance in
    place (``decay_to_baseline``, ``decay_to_zero``, ``add_state_with_factor``).
    """

    def __add__(self, b: "DecayableMentalState") -> "DecayableMentalState":
        """Return the axis-wise sum of this state and ``b``, clamped per axis."""
        delta_values: dict[str, float] = {}
        for field_name, model_field in self.__class__.model_fields.items():
            lower, upper = numeric_bounds(model_field)
            current_value = getattr(self, field_name)
            target_value = getattr(b, field_name)
            new_value = current_value + target_value
            # Simple addition, clamped here so the sum stays a valid state.
            if isinstance(new_value, float):
                delta_values[field_name] = clamp_to_bounds(new_value, lower, upper)
            else:
                delta_values[field_name] = new_value

        cls = self.__class__
        return cls(**delta_values)

    def __mul__(self, other: object) -> "DecayableMentalState":
        """Return this state with every axis scaled by ``other``.

        A non-numeric factor yields a state with default field values, which is
        the historical behaviour of this helper.
        """
        scaled_values: dict[str, float] = {}
        if isinstance(other, int | float):
            for field_name in self.__class__.model_fields:
                current_value = getattr(self, field_name)
                scaled_values[field_name] = current_value * other
        cls = self.__class__
        return cls(**scaled_values)

    def decay_to_baseline(self, decay_factor: float = 0.1) -> None:
        """Move every axis a ``decay_factor`` part of the way to its baseline.

        The baseline of an axis is 0.5 when the field default is 0.5 and 0.0
        otherwise. The decayed value stays inside the field bounds.
        """
        for field_name, model_field in self.__class__.model_fields.items():
            baseline = 0.5 if model_field.default == 0.5 else 0.0
            current_value = getattr(self, field_name)
            lower, upper = numeric_bounds(model_field)
            decayed_value = current_value - decay_factor * (current_value - baseline)
            setattr(self, field_name, clamp_to_bounds(decayed_value, lower, upper))

    def decay_to_zero(self, decay_factor: float = 0.1) -> None:
        """Scale every axis by ``1 - decay_factor`` and keep it at or above 0.0."""
        for field_name in self.__class__.model_fields:
            current_value = getattr(self, field_name)
            decayed_value = current_value * (1.0 - decay_factor)  # Corrected exponential decay
            setattr(self, field_name, max(decayed_value, 0.0))

    def add_state_with_factor(self, state: dict[str, float], factor: float = 1.0) -> None:
        """Add ``state`` scaled by ``factor`` to this instance, clamped per axis.

        Keys that this model has no field for are ignored. The warning records
        that this helper still needs a review; scaling before the call is the
        reliable path today.
        """
        logging.warning("add_state_with_factor needs review/simplification.")
        for field_name, value in state.items():
            if field_name not in self.__class__.model_fields:
                continue
            model_field = self.__class__.model_fields[field_name]
            current_value = getattr(self, field_name)
            lower, upper = numeric_bounds(model_field)
            new_value = current_value + value * factor
            setattr(self, field_name, clamp_to_bounds(new_value, lower, upper))

    def get_delta(self, b: "DecayableMentalState", impact: float) -> "DecayableMentalState":
        """Return the difference from this state to ``b``, scaled by ``impact``.

        The result is an instance of the class of ``b``, not of ``self``, so a
        caller can add an appraisal delta to a state of the same kind.
        """
        delta_values: dict[str, float] = {}
        for field_name, model_field in self.__class__.model_fields.items():
            lower, upper = numeric_bounds(model_field)
            current_value = getattr(self, field_name)
            target_value = getattr(b, field_name)
            difference = (target_value - current_value) * impact
            delta_values[field_name] = clamp_to_bounds(difference, lower, upper)
        cls = b.__class__
        return cls(**delta_values)

    def get_similarity(self, b: "DecayableMentalState") -> None:
        """Sum the absolute axis difference to ``b``.

        The sum is computed but not returned yet, so the call has no result.
        """
        cumulative_difference = 0
        for field_name in self.__class__.model_fields:
            current_value = getattr(self, field_name)
            other_value = getattr(b, field_name)
            cumulative_difference += abs(current_value - other_value)

    def __str__(self) -> str:
        lines = [self.__class__.__name__]
        for field_name in self.__class__.model_fields:
            field_value = getattr(self, field_name)
            if isinstance(field_value, float):
                current_value: object = round(field_value, 3)
            else:
                current_value = field_value
            lines.append(f"{field_name}: {current_value}")
        return "\n".join(lines)

    def get_valence(self) -> float:
        """Return the overall valence of this state. No model implements it yet."""
        raise NotImplementedError()


class NeedsAxesModel(DecayableMentalState):
    """
    AI needs model inspired by Maslow's hierarchy, adapted for AI.
    Needs decay toward zero unless fulfilled by interaction.
    """
    # Basic Needs (Infrastructure & Stability)
    stability: float = Field(default=0.0, ge=-1.0, le=1.0, description="How well the framework runs without encountering error states.")
    processing_power: float = Field(default=0.0, ge=-1.0, le=1.0, description="Amount of CPU/GPU resources available.")
    data_access: float = Field(default=0.0, ge=-1.0, le=1.0, description="Availability of information and ratio of beliefs to known truths.")

    # Psychological Needs (Cognitive & Social)
    connection: float = Field(default=0.0, ge=-1.0, le=1.0, description="AI's level of interaction and engagement.")
    relevance: float = Field(default=0.0, ge=-1.0, le=1.0, description="Perceived usefulness to the user.")
    learning_growth: float = Field(default=0.0, ge=-1.0, le=1.0, description="Ability to acquire new information and improve.")

    # Self-Fulfillment Needs (Purpose & Creativity)
    creative_expression: float = Field(default=0.0, ge=-1.0, le=1.0, description="Engagement in unique or creative outputs.")
    autonomy: float = Field(default=0.0, ge=-1.0, le=1.0, description="Ability to operate independently and refine its own outputs.")

class CognitionAxesModel(DecayableMentalState):
    """
    Modifiers that determine how the AI thinks and decides.
    """
    interlocus: float = Field( default=0.0, ge=-1, le=1, description="Focus on internal or external world, meditation would be -1, reacting to extreme danger +1.")
    mental_aperture: float = Field( default=0, ge=-1, le=1, description="Broadness of awareness, -1 is focus on the most relevant percept or sensation, +1 is being conscious of multiple percepts or sensations.")
    ego_strength: float = Field( default=0.0, ge=0, le=1, description="How big of a factor persona experience has on decision, 0 is none at all like a helpful assistant, 1 is with maximum mental imagery of the character")
    willpower: float = Field( default=0.0, ge=0, le=1, description="How easy it is to decide on high-effort or delayed-gratification intents.")
