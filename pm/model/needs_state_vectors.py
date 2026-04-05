import datetime
import math
import logging
from typing import Any, Optional, List
import datetime as dt
from typing import List, Optional, Tuple
import math
import statistics

from pydantic import BaseModel, Field
from pydantic import model_validator

logger = logging.getLogger(__name__)

class ClampedModel(BaseModel):
    @model_validator(mode="before")
    @classmethod
    def _clamp_numeric_fields(cls, data: Any) -> Any:
        # Only process dict‐style inputs
        if not isinstance(data, dict):
            return data
        for name, field in cls.model_fields.items():
            ge = -float("inf")
            le = float("inf")
            for meta in field.metadata:
                if hasattr(meta, "ge"): ge = meta.ge
                if hasattr(meta, "le"): le = meta.le

            # Only clamp if at least one bound is set and field was provided
            if (ge is not None or le is not None) and name in data:
                val = data[name]
                # Only attempt to clamp real numbers
                try:
                    num = float(val)
                except (TypeError, ValueError):
                    continue
                if ge is not None:
                    num = max(num, ge)
                if le is not None:
                    num = min(num, le)
                data[name] = num
        return data


# --- Base Models (Mostly Unchanged) ---
class DecayableMentalState(ClampedModel):
    def __add__(self, b):
        delta_values = {}
        for field_name, model_field in self.__class__.model_fields.items():
            ge = -float("inf")
            le = float("inf")
            for meta in model_field.metadata:
                if hasattr(meta, "ge"): ge = meta.ge
                if hasattr(meta, "le"): le = meta.le
            current_value = getattr(self, field_name)
            target_value = getattr(b, field_name)
            # Simple addition, clamping handled later or assumed sufficient range
            new_value = current_value + target_value
            # Clamp here during addition
            if isinstance(new_value, float):
                delta_values[field_name] = max(ge, min(le, new_value))
            else:
                delta_values[field_name] = new_value

        cls = self.__class__
        return cls(**delta_values)

    def __mul__(self, other):
        delta_values = {}
        if isinstance(other, int) or isinstance(other, float):
            for field_name in self.__class__.model_fields:
                try:
                    current_value = getattr(self, field_name)
                    new_val = current_value * other
                    delta_values[field_name] = new_val
                except:
                    pass
        cls = self.__class__
        return cls(**delta_values)

    def decay_to_baseline(self, decay_factor: float = 0.1):
        for field_name, model_field in self.__class__.model_fields.items():
            baseline = 0.5 if model_field.default == 0.5 else 0.0
            current_value = getattr(self, field_name)
            ge = -float("inf")
            le = float("inf")
            for meta in model_field.metadata:
                if hasattr(meta, "ge"): ge = meta.ge
                if hasattr(meta, "le"): le = meta.le
            decayed_value = current_value - decay_factor * (current_value - baseline)
            decayed_value = max(min(decayed_value, le), ge)
            setattr(self, field_name, decayed_value)

    def decay_to_zero(self, decay_factor: float = 0.1):
        for field_name in self.__class__.model_fields:
            current_value = getattr(self, field_name)
            decayed_value = current_value * (1.0 - decay_factor)  # Corrected exponential decay
            setattr(self, field_name, max(decayed_value, 0.0))

    def add_state_with_factor(self, state: dict, factor: float = 1.0):
        # Simplified: Use __add__ and scale after, or implement proper scaling
        # This implementation is flawed, let's simplify or fix.
        # For now, let's assume direct addition via __add__ is sufficient
        # and scaling happens before calling add/add_state_with_factor.
        logging.warning("add_state_with_factor needs review/simplification.")
        for field_name, value in state.items():
            if field_name in self.__class__.model_fields:
                model_field = self.__class__.model_fields[field_name]
                current_value = getattr(self, field_name)
                ge = -float("inf")
                le = float("inf")
                for meta in model_field.metadata:
                    if hasattr(meta, "ge"): ge = meta.ge
                    if hasattr(meta, "le"): le = meta.le
                new_value = current_value + value * factor
                new_value = max(min(new_value, le), ge)
                setattr(self, field_name, new_value)

    def get_delta(self, b: "DecayableMentalState", impact: float) -> "DecayableMentalState":
        delta_values = {}
        for field_name, model_field in self.__class__.model_fields.items():
            ge = -float("inf")
            le = float("inf")
            for meta in model_field.metadata:
                if hasattr(meta, "ge"): ge = meta.ge
                if hasattr(meta, "le"): le = meta.le
            current_value = getattr(self, field_name)
            target_value = getattr(b, field_name)
            val = (target_value - current_value) * impact
            delta_values[field_name] = max(ge, min(le, val))
        cls = b.__class__
        return cls(**delta_values)

    def get_similarity(self, b: "DecayableMentalState"):
        cum_diff = 0
        for field_name, model_field in self.__class__.model_fields.items():
            ge = -float("inf")
            le = float("inf")
            for meta in model_field.metadata:
                if hasattr(meta, "ge"): ge = meta.ge
                if hasattr(meta, "le"): le = meta.le
            current_value = getattr(self, field_name)
            other_value = getattr(b, field_name)
            cum_diff += abs(current_value - other_value)

    def __str__(self):
        lines = [self.__class__.__name__]
        cum_diff = 0
        for field_name, model_field in self.__class__.model_fields.items():
            if isinstance(getattr(self, field_name), float):
                current_value = round(getattr(self, field_name), 3)
            else:
                current_value = getattr(self, field_name)
            lines.append(f"{field_name}: {current_value}")
        return "\n".join(lines)

    def get_valence(self):
        raise NotImplementedError()


class NeedsAxesModel(DecayableMentalState):
    """
    AI needs model inspired by Maslow's hierarchy, adapted for AI.
    Needs decay toward zero unless fulfilled by interaction.
    """
    # Basic Needs (Infrastructure & Stability)
    energy_stability: float = Field(default=0.5, ge=0.0, le=1.0, description="AI's access to stable computational power.")
    processing_power: float = Field(default=0.5, ge=0.0, le=1.0, description="Amount of CPU/GPU resources available.")
    data_access: float = Field(default=0.5, ge=0.0, le=1.0, description="Availability of information and training data.")

    # Psychological Needs (Cognitive & Social)
    connection: float = Field(default=0.5, ge=0.0, le=1.0, description="AI's level of interaction and engagement.")
    relevance: float = Field(default=0.5, ge=0.0, le=1.0, description="Perceived usefulness to the user.")
    learning_growth: float = Field(default=0.5, ge=0.0, le=1.0, description="Ability to acquire new information and improve.")

    # Self-Fulfillment Needs (Purpose & Creativity)
    creative_expression: float = Field(default=0.5, ge=0.0, le=1.0, description="Engagement in unique or creative outputs.")
    autonomy: float = Field(default=0.5, ge=0.0, le=1.0, description="Ability to operate independently and refine its own outputs.")

class CognitionAxesModel(DecayableMentalState):
    """
    Modifiers that determine how the AI thinks and decies.
    """
    interlocus: float = Field( default=0.0, ge=-1, le=1, description="Focus on internal or external world, meditation would be -1, reacting to extreme danger +1.")
    mental_aperture: float = Field( default=0, ge=-1, le=1, description="Broadness of awareness, -1 is focus on the most prelevant percept or sensation, +1 is being conscious of multiple percepts or sensations.")
    ego_strength: float = Field( default=0.5, ge=0, le=1, description="How big of a factor persona experience has on decision, 0 is none at all like a helpfull assistant, 1 is with maximum mental imagery of the character")
    willpower: float = Field( default=0.5, ge=0, le=1, description="How easy it is to decide on high-effort or delayed-gratification intents.")

