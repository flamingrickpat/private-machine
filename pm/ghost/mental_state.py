from logging import raiseExceptions
from typing import Type, List

from pydantic import BaseModel, Field
from sqlmodel import Session, select

from pm.controller import controller
from pm.database.db_utils import update_database_item
from pm.database.tables import CognitiveState


# Assuming CognitiveState and update_database_item are defined as provided.
# Also assuming that a "controller.get_session()" is available that returns a SQLModel Session.

def base_model_to_db(tick_id: int, bm: BaseModel) -> List["CognitiveState"]:
    """
    Converts a BaseModel instance into a list of CognitiveState database records.
    Each field in the BaseModel is saved as a separate record.
    """
    records = []
    for field_name in bm.model_fields:
        current_value = getattr(bm, field_name)
        # Create a record for this field.
        record = CognitiveState(tick_id=tick_id, key=field_name, value=current_value)
        # Insert into the DB (this function handles session commit/refresh as needed).
        update_database_item(record)
        records.append(record)
    return records

def db_to_base_model(tick_id: int, model_type: Type[BaseModel]) -> BaseModel:
    """
    Reconstructs a BaseModel instance from the CognitiveState records corresponding to tick_id.
    """
    # Get a session from the controller (assumes controller.get_session() is defined).
    session: Session = controller.get_session()
    # Query for all CognitiveState records with the given tick_id.
    statement = select(CognitiveState).where(CognitiveState.tick_id == tick_id)
    results = session.exec(statement).all()

    # Map the results into a dictionary suitable for the BaseModel.
    data = {}
    for record in results:
        data[record.key] = record.value

    #if tick_id > 0 and len(data) == 0:
    #    raise Exception("sneed")

    # Create and return an instance of the given model_type.
    return model_type(**data)


class DecayableMentalState(BaseModel):
    def __add__(self, b):
        delta_values = {}
        for field_name in self.model_fields:
            current_value = getattr(self, field_name)
            target_value = getattr(b, field_name)
            delta_values[field_name] = max(-1, min(1, ((target_value + current_value) / 2)))

        cls = b.__class__
        return cls(**delta_values)


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

    def get_delta(self, b: "DecayableMentalState", impact: float) -> "DecayableMentalState":
        """
        Returns a new EmotionalAxesModel instance where each field represents the delta
        (difference) between the corresponding field in b and the current state,
        scaled by the impact factor.
        """
        delta_values = {}
        for field_name in self.model_fields:
            current_value = getattr(self, field_name)
            target_value = getattr(b, field_name)
            delta_values[field_name] = (target_value - current_value) * impact

        cls = b.__class__
        return cls(**delta_values)


class EmotionalAxesModel(DecayableMentalState):
    """
    Determine current emotion vector. This will be added to current emotions.
    """
    joy: float = Field(description="Handles the experience of positive emotions such as happiness and delight.", default=0, ge=-1, le=1)
    sadness: float = Field(description="Handles the experience of loss or disappointment.", default=0, ge=-1, le=1)
    anger: float = Field(description="Handles the experience of frustration or perceived injustice.", default=0, ge=-1, le=1)
    fear: float = Field(description="Handles the experience of anxiety or concern over potential threats.", default=0, ge=-1, le=1)
    disgust: float = Field(description="Handles the experience of aversion or repulsion.", default=0, ge=-1, le=1)
    surprise: float = Field(description="Handles the experience of unexpected or novel stimuli.", default=0, ge=-1, le=1)
    love: float = Field(description="Handles the experience of deep affection or connection.", default=0, ge=-1, le=1)
    pride: float = Field(description="Handles the experience of satisfaction or accomplishment.", default=0, ge=-1, le=1)
    guilt: float = Field(description="Handles the experience of remorse or regret.", default=0, ge=-1, le=1)
    nostalgia: float = Field(description="Handles the experience of fond memories and longing for the past.", default=0, ge=-1, le=1)
    gratitude: float = Field(description="Handles the experience of thankfulness or appreciation.", default=0, ge=-1, le=1)
    shame: float = Field(description="Handles the experience of embarrassment or self-consciousness.", default=0, ge=-1, le=1)


class NeedsAxesModel(DecayableMentalState):
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




if __name__ == '__main__':
    # Example usage
    model = EmotionalAxesModel()
    model.joy += 0.3
    print("Before decay:", model.model_dump())
    model.decay_to_baseline(0.2)
    print("After decay:", model.model_dump())
    model.add_state_with_factor({'valence': 0.3, 'calmness': -0.2}, 0.5)
    print("After adding state:", model.model_dump())



    # Example usage
    ai_needs = NeedsAxesModel()
    ai_needs2 = NeedsAxesModel()
    ai_needs2.decay_to_zero(11)

    print("Before decay:", ai_needs.model_dump())

    # Simulate time passing without fulfillment
    ai_needs.decay_to_zero(0.2)
    print("After decay:", ai_needs.model_dump())

    # Simulate AI receiving external interaction
    ai_needs.add_state_with_factor({'connection': 0.3, 'relevance': 0.2}, 0.5)
    print("After adding state:", ai_needs.model_dump())

    ai_needs3 = ai_needs + ai_needs2
    print("After adding state 2:", ai_needs3.model_dump())