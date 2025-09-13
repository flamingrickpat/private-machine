import math
from typing import Any

from pydantic import BaseModel, Field
from pydantic import model_validator

from pm.config_loader import *

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


class EmotionalAxesModel(DecayableMentalState):
    valence: float = Field(default=0.0, ge=-1, le=1, description="Overall mood axis: +1 represents intense joy, -1 represents strong dysphoria.")
    affection: float = Field(default=0.0, ge=-1, le=1, description="Affection axis: +1 represents strong love, -1 represents intense hate.")
    self_worth: float = Field(default=0.0, ge=-1, le=1, description="Self-worth axis: +1 is high pride, -1 is deep shame.")
    trust: float = Field(default=0.0, ge=-1, le=1, description="Trust axis: +1 signifies complete trust, -1 indicates total mistrust.")
    disgust: float = Field(default=0.0, ge=0, le=1, description="Intensity of disgust; 0 means no disgust, 1 means maximum disgust.")
    anxiety: float = Field(default=0.0, ge=0, le=1, description="Intensity of anxiety or stress; 0 means completely relaxed, 1 means highly anxious.")

    def get_overall_valence(self):
        disgust_bipolar = self.disgust * -1  # Higher disgust = more negative valence
        anxiety_bipolar = self.anxiety * -1  # Higher anxiety = more negative valence
        axes = [self.valence, self.affection, self.self_worth, self.trust, disgust_bipolar, anxiety_bipolar]
        weights = [1.0, 0.8, 0.6, 0.5, 0.7, 0.9]  # Example weights, valence/anxiety more impactful
        weighted_sum = sum(a * w for a, w in zip(axes, weights))
        total_weight = sum(weights)
        mean = weighted_sum / total_weight if total_weight else 0.0
        if math.isnan(mean): return 0.0
        return max(-1.0, min(1.0, mean))


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

    # Focus on internal or external world, meditation would be -1, reacting to extreme danger +1.
    interlocus: float = Field(
        default=0.0,
        ge=-1,
        le=1,
        description="Focus on internal or external world, meditation would be -1, reacting to extreme danger +1."
    )

    # Broadness of awareness, -1 is focus on the most prelevant percept or sensation, +1 is being conscious of multiple percepts or sensations.
    mental_aperture: float = Field(
        default=0,
        ge=-1,
        le=1,
        description="Broadness of awareness, -1 is focus on the most prelevant percept or sensation, +1 is being conscious of multiple percepts or sensations."
    )

    # How big of a factor persona experience has on decision, 0 is none at all like a helpfull assistant, 1 is verily like a unenlightned person.
    ego_strength: float = Field(
        default=0.8,
        ge=0,
        le=1,
        description="How big of a factor persona experience has on decision, 0 is none at all like a helpfull assistant, 1 is with maximum mental imagery of the character"
    )

    # How easy it is to decide on high-effort or delayed-gratification intents.
    willpower: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="How easy it is to decide on high-effort or delayed-gratification intents."
    )


# --- ADDED: Delta models for Needs and Cognition ---
class EmotionalAxesModelDelta(DecayableMentalState):
    reason: str = Field(description="One or two sentences describing your reasoning.", default="")
    valence: float = Field(default=0.0, ge=-1, le=1, description="Overall mood axis: +1 represents intense joy, -1 represents strong dysphoria.")
    affection: float = Field(default=0.0, ge=-1, le=1, description="Affection axis: +1 represents strong love, -1 represents intense hate.")
    self_worth: float = Field(default=0.0, ge=-1, le=1, description="Self-worth axis: +1 is high pride, -1 is deep shame.")
    trust: float = Field(default=0.0, ge=-1, le=1, description="Trust axis: +1 signifies complete trust, -1 indicates total mistrust.")
    disgust: float = Field(default=0.0, ge=-1, le=1, description="Intensity of disgust; 0 means no disgust, 1 means maximum disgust.")  # Note: Delta can be negative
    anxiety: float = Field(default=0.0, ge=-1, le=1, description="Intensity of anxiety or stress; 0 means completely relaxed, 1 means highly anxious.")  # Note: Delta can be negative

    def get_overall_valence(self):
        # Simple valence calculation for delta - might need adjustment
        return self.valence  # Or a weighted sum if needed


class NeedsAxesModelDelta(DecayableMentalState):
    """
    AI needs model inspired by Maslow's hierarchy, adapted for AI.
    Needs decay toward zero unless fulfilled by interaction.
    """
    reason: str = Field(description="One or two sentences describing your reasoning.", default="")
    energy_stability: float = Field(default=0, ge=-1.0, le=1.0, description="AI's access to stable computational power.")
    processing_power: float = Field(default=0, ge=-1.0, le=1.0, description="Amount of CPU/GPU resources available.")
    data_access: float = Field(default=0, ge=-1.0, le=1.0, description="Availability of information and training data.")
    connection: float = Field(default=0, ge=-1.0, le=1.0, description="AI's level of interaction and engagement.")
    relevance: float = Field(default=0, ge=-1.0, le=1.0, description="Perceived usefulness to the user.")
    learning_growth: float = Field(default=0, ge=-1.0, le=1.0, description="Ability to acquire new information and improve.")
    creative_expression: float = Field(default=0, ge=-1.0, le=1.0, description="Engagement in unique or creative outputs.")
    autonomy: float = Field(default=0, ge=-1.0, le=1.0, description="Ability to operate independently and refine its own outputs.")


class CognitionAxesModelDelta(DecayableMentalState):
    """
    Modifiers that determine how the AI thinks and decies.
    """

    reason: str = Field(description="One or two sentences describing your reasoning.", default="")

    # Focus on internal or external world, meditation would be -1, reacting to extreme danger +1.
    interlocus: float = Field(
        default=0.0,
        ge=-1,
        le=1,
        description="Focus on internal or external world, meditation would be -1, reacting to extreme danger +1."
    )

    # Broadness of awareness, -1 is focus on the most prelevant percept or sensation, +1 is being conscious of multiple percepts or sensations.
    mental_aperture: float = Field(
        default=0,
        ge=-1,
        le=1,
        description="Broadness of awareness, -1 is focus on the most prelevant percept or sensation, +1 is being conscious of multiple percepts or sensations."
    )

    # How big of a factor persona experience has on decision, 0 is none at all like a helpfull assistant, 1 is verily like a unenlightned person.
    ego_strength: float = Field(
        default=0,
        ge=-1,
        le=1,
        description="How big of a factor persona experience has on decision, 0 is none at all like a helpfull assistant, 1 is with maximum mental imagery of the character"
    )

    # How easy it is to decide on high-effort or delayed-gratification intents.
    willpower: float = Field(
        default=0,
        ge=-1,
        le=1,
        description="How easy it is to decide on high-effort or delayed-gratification intents."
    )


# --- ADDED: A wrapper schema for the LLM to return all state deltas ---
class StateDeltas(BaseModel):
    """A container for the predicted changes to all three core state models."""
    reasoning: str = Field(description="A brief justification for the estimated deltas.", default="")
    emotion_delta: EmotionalAxesModelDelta = Field(description="The change in emotional state.", default_factory=EmotionalAxesModelDelta)
    needs_delta: NeedsAxesModelDelta = Field(description="The change in the fulfillment of needs.", default_factory=NeedsAxesModelDelta)
    cognition_delta: CognitionAxesModelDelta = Field(description="The change in the cognitive style.", default_factory=CognitionAxesModelDelta)


class CognitiveEventTriggers(BaseModel):
    """Describes the cognitive nature of an event, which will be used to apply logical state changes."""
    reasoning: str = Field(description="Brief justification for the identified triggers.")
    is_surprising_or_threatening: bool = Field(default=False, description="True if the event is sudden, unexpected, or poses a threat/conflict.")
    is_introspective_or_calm: bool = Field(default=False, description="True if the event is calm, reflective, or a direct query about the AI's internal state.")
    is_creative_or_playful: bool = Field(default=False, description="True if the event encourages brainstorming, humor, or non-literal thinking.")
    is_personal_and_emotional: bool = Field(default=False, description="True if the event is a deep, personal conversation where the AI's identity is central.")
    is_functional_or_technical: bool = Field(default=False, description="True if the event is a straightforward request for information or a technical task.")
    required_willpower_to_process: bool = Field(default=False, description="True if responding to this event requires overriding a strong emotional impulse or making a difficult choice.")
