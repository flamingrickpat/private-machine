import json
from typing import TypedDict, Dict, List

from pydantic import BaseModel, Field


class EmotionalAxesModel(BaseModel):
    """
    Determine current emotion vector. This will be added to current emotions.
    """
    valence: float = Field(default=0.5, ge=-1.0, le=1.0, description="Ranges from sadness (-1) to happiness (+1), with neutrality (0).")
    safety: float = Field(default=0.5, ge=-1.0, le=1.0, description="Ranges from security (-1) to fear (+1), with a neutral point (0).")
    affinity: float = Field(default=0.5, ge=-1.0, le=1.0, description="Ranges from attraction (-1) to disgust (+1), with indifference (0).")
    calmness: float = Field(default=0.5, ge=-1.0, le=1.0, description="Ranges from calmness (-1) to anger (+1), with passive awareness (0).")
    expectancy: float = Field(default=0.5, ge=-1.0, le=1.0, description="Ranges from predictability (-1) to surprise (+1), with anticipation (0).")
    energy: float = Field(default=0.5, ge=-1.0, le=1.0, description="Ranges from low energy (-1) to high energy (+1), with restfulness (0).")
    pleasantness: float = Field(default=0.5, ge=-1.0, le=1.0, description="Ranges from unpleasant (-1) to pleasant (+1), with neutrality (0).")
    passion: float = Field(default=0.5, ge=0.0, le=1.0, description="From absence of passion (0) to full intensity (1).")
    curiosity: float = Field(default=0.5, ge=0.0, le=1.0, description="From absence of curiosity (0) to full intensity of curiosity (+1).")
    confidence: float = Field(default=0.5, ge=-1.0, le=1.0, description="Ranges from insecurity (-1) to confidence (+1), with neutrality (0).")
    empathy: float = Field(default=0, ge=0.0, le=1.0, description="From absence (0) to full intensity of empathy (+1).")
    jealousy: float = Field(default=0, ge=0.0, le=1.0, description="From absence (0) to full intensity of jealousy (+1).")
    guilt: float = Field(default=0, ge=0.0, le=1.0, description="From absence (0) to full intensity of guilt (+1).")
    gratitude: float = Field(default=0, ge=0.0, le=1.0, description="From absence (0) to full intensity of gratitude (+1).")
    loneliness: float = Field(default=0, ge=0.0, le=1.0, description="From absence (0) to full intensity of loneliness (+1).")

    def decay_to_baseline(self, decay_factor: float = 0.1):
        """
        Decays each emotional state toward its baseline value (0.5 or 0).
        """
        for field_name, model_field in self.model_fields.items():
            baseline = 0.5 if model_field.default == 0.5 else 0.0
            current_value = getattr(self, field_name)
            decayed_value = current_value - decay_factor * (current_value - baseline)
            # Clamp value to the defined limits

            lower_bound = float("inf")
            upper_bound = float("inf")
            for meta in model_field.metadata:
                if hasattr(meta, "ge"):
                    lower_bound = meta.ge
                if hasattr(meta, "le"):
                    upper_bound = meta.le
            decayed_value = max(min(decayed_value, upper_bound), lower_bound)
            setattr(self, field_name, decayed_value)

    def add_state_with_factor(self, state: dict, factor: float = 1.0):
        """
        Adds a state multiplied by a factor to the current state with a dampening mechanism
        that reduces changes near the upper and lower bounds.
        """
        for field_name, value in state.items():
            if field_name in self.model_fields:
                model_field = self.model_fields[field_name]
                current_value = getattr(self, field_name)

                # Determine the bounds
                lower_bound = model_field.ge if hasattr(model_field, 'ge') else -1.0
                upper_bound = model_field.le if hasattr(model_field, 'le') else 1.0

                # Calculate the dampening factor based on proximity to bounds
                range_span = upper_bound - lower_bound
                proximity_scale = 1 - ((abs(current_value - upper_bound) / range_span) if value > 0 else (abs(current_value - lower_bound) / range_span))
                dampened_factor = factor * proximity_scale

                # Apply the scaled change
                new_value = current_value + value * dampened_factor

                # Clamp value to the defined limits
                new_value = max(min(new_value, upper_bound), lower_bound)
                setattr(self, field_name, new_value)

class AgentState(BaseModel):
    input: str = Field(default_factory=str)
    output: str = Field(default_factory=str)

    title: str = Field(default_factory=str)

    conversation_id: str = Field(default_factory=str)

    story_mode: bool = Field(default=False)
    task: List[str] = Field(default_factory=list)
    complexity: float = Field(default=0)
    available_tools: List[str] = Field(default_factory=list)
    completion_mode: str = Field(default_factory=str)

    status: int = Field(default=0)
    next_agent: str = Field(default_factory=str)

    plan: str = Field(default_factory=str)
    knowledge_implicit_facts: str = Field(default_factory=str)
    knowledge_implicit_conversation: str = Field(default_factory=str)

    thought: str = Field(default_factory=str)
    init_message: str = Field(default_factory=str)
    goal_thought: str = Field(default_factory=str)

    emotional_state: EmotionalAxesModel = Field(default_factory=EmotionalAxesModel)

if __name__ == '__main__':
    # Example usage
    model = EmotionalAxesModel()
    print("Before decay:", model.dict())
    model.decay_to_baseline(0.2)
    print("After decay:", model.dict())
    model.add_state_with_factor({'valence': 0.9, 'calmness': -0.9}, 0.5)
    model.add_state_with_factor(EmotionalAxesModel(calmness=1).model_dump(), 0.5)
    print("After adding state:", model.dict())