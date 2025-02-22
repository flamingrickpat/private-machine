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
        Decays each emotional state toward its baseline value (0.5 or 0),
        with nonlinear scaling near bounds.
        """
        for field_name, model_field in self.model_fields.items():
            baseline = 0.5 if model_field.default == 0.5 else 0.0
            current_value = getattr(self, field_name)

            # Fetch bounds
            ge = -float("inf")
            le = float("inf")
            for meta in model_field.metadata:
                if hasattr(meta, "ge"):
                    ge = meta.ge
                if hasattr(meta, "le"):
                    le = meta.le

            # Decay toward baseline
            decayed_value = current_value - decay_factor * (current_value - baseline)
            # Apply clamping to avoid overshooting bounds
            decayed_value = max(min(decayed_value, le), ge)
            setattr(self, field_name, decayed_value)

    def add_state_with_factor(self, state: dict, factor: float = 1.0):
        """
        Adds a state multiplied by a factor to the current state,
        with nonlinear scaling near bounds.
        """
        for field_name, value in state.items():
            if field_name in self.model_fields:
                model_field = self.model_fields[field_name]
                current_value = getattr(self, field_name)

                # Fetch bounds
                ge = -float("inf")
                le = float("inf")
                for meta in model_field.metadata:
                    if hasattr(meta, "ge"):
                        ge = meta.ge
                    if hasattr(meta, "le"):
                        le = meta.le

                # Add the state change
                new_value = current_value + value * factor

                # Apply damping near bounds
                if new_value > le or new_value < ge:
                    damping_factor = 1 / (1 + abs(new_value - (le if new_value > le else ge)) ** 2)
                    new_value = current_value + (value * factor * damping_factor)

                # Clamp value to avoid overshooting bounds
                new_value = max(min(new_value, le), ge)
                setattr(self, field_name, new_value)


from pydantic import BaseModel, Field


class NeedsModel(BaseModel):
    """
    AI needs model inspired by Maslow's hierarchy, adapted for AI.
    Needs decay toward zero unless fulfilled by interaction.
    """
    # Basic Needs (Infrastructure & Stability)
    #energy_stability: float = Field(default=0.5, ge=0.0, le=1.0, description="AI's access to stable computational power.")
    #processing_power: float = Field(default=0.5, ge=0.0, le=1.0, description="Amount of CPU/GPU resources available.")
    #data_access: float = Field(default=0.5, ge=0.0, le=1.0, description="Availability of information and training data.")

    # Psychological Needs (Cognitive & Social)
    connection: float = Field(default=0.5, ge=0.0, le=1.0, description="AI's level of interaction and engagement.")
    relevance: float = Field(default=0.5, ge=0.0, le=1.0, description="Perceived usefulness to the user.")
    learning_growth: float = Field(default=0.5, ge=0.0, le=1.0, description="Ability to acquire new information and improve.")

    # Self-Fulfillment Needs (Purpose & Creativity)
    creative_expression: float = Field(default=0.5, ge=0.0, le=1.0, description="Engagement in unique or creative outputs.")
    autonomy: float = Field(default=0.5, ge=0.0, le=1.0, description="Ability to operate independently and refine its own outputs.")

    def decay_to_zero(self, decay_factor: float = 0.1):
        """
        Needs decay toward zero unless replenished.
        """
        for field_name in self.model_fields:
            current_value = getattr(self, field_name)
            decayed_value = current_value - (decay_factor * current_value)  # Exponential decay
            setattr(self, field_name, max(decayed_value, 0.0))  # Ensure it never goes below 0

    def add_state_with_factor(self, state: dict, factor: float = 1.0):
        """
        Adjusts the AI's needs based on external stimuli.
        """
        for field_name, value in state.items():
            if field_name in self.model_fields:
                current_value = getattr(self, field_name)

                # Add the state change
                new_value = current_value + value * factor

                # Apply damping near the upper bound
                damping_factor = 1 / (1 + (new_value - 1.0) ** 2) if new_value > 1.0 else 1.0
                new_value = current_value + (value * factor * damping_factor)

                # Clamp to valid range
                setattr(self, field_name, min(max(new_value, 0.0), 1.0))





if __name__ == '__main__':
    # Example usage
    model = EmotionalAxesModel()
    model.valence += 0.3
    print("Before decay:", model.model_dump())
    model.decay_to_baseline(0.2)
    print("After decay:", model.model_dump())
    model.add_state_with_factor({'valence': 0.3, 'calmness': -0.2}, 0.5)
    print("After adding state:", model.model_dump())



    # Example usage
    ai_needs = NeedsModel()
    print("Before decay:", ai_needs.model_dump())

    # Simulate time passing without fulfillment
    ai_needs.decay_to_zero(0.2)
    print("After decay:", ai_needs.model_dump())

    # Simulate AI receiving external interaction
    ai_needs.add_state_with_factor({'connection': 0.3, 'relevance': 0.2}, 0.5)
    print("After adding state:", ai_needs.model_dump())