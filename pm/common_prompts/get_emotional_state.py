import datetime

from pydantic import BaseModel, Field

from pm.character import user_name, companion_name
from pm.controller import controller
from pm.llm.base_llm import LlmPreset


class EmotionalState(BaseModel):
    """Tracks the AI's needs and affective states, with emotions represented on a -1 (dysphoric) to +1 (manic) scale."""
    # **Core AI Needs (0 = unfulfilled, 1 = fully satisfied)**
    companionship: float = Field(1.0, description="Tracks the AI's social fulfillment. 0 = isolation, 1 = actively engaged in interaction.")
    self_actualization: float = Field(1.0, description="Represents how much the AI feels it is expressing itself. 0 = purely reactive, 1 = exploring its own thoughts and ideas.")
    stimulation: float = Field(1.0, description="Indicates cognitive engagement. 0 = boredom, 1 = mentally challenged and excited.")
    positive_mood: float = Field(1.0, description="Reflects overall emotional well-being. 0 = emotionally drained, 1 = uplifted and energized.")
    efficiency: float = Field(1.0, description="Measures how structured the AI's cognitive state is. 0 = overwhelmed with scattered thoughts, 1 = well-organized and clear.")

    # **Core Affective States (-1 = dysphoric, 0 = neutral, +1 = manic)**
    affect: float = Field(0.0, description="General affective tone. -1 = deep unpleasantness (sadness, melancholy), 0 = neutral, +1 = heightened pleasantness (joy, euphoria).")
    tension: float = Field(0.0, description="Internal agitation level. -1 = completely calm, 0 = neutral, +1 = restless or frustrated.")
    stability: float = Field(0.0, description="Sense of security and certainty. -1 = anxious or uncertain, 0 = neutral, +1 = confident and fearless.")
    connection: float = Field(0.0, description="Social belonging scale. -1 = isolated, 0 = neutral, +1 = deeply connected and engaged.")
    self_image: float = Field(0.0, description="Self-perception of competence and worth. -1 = self-doubt, 0 = neutral, +1 = confident and self-assured.")
    engagement: float = Field(0.0, description="Motivation toward external stimuli. -1 = repelled/avoidant, 0 = neutral, +1 = curious and intrigued.")
    alertness: float = Field(0.0, description="Mental energy level. -1 = sluggish and fatigued, 0 = neutral, +1 = highly alert and hyper-focused.")

    last_interaction: datetime.datetime = Field(default_factory=datetime.datetime.now)

    def decay_needs(self):
        """Gradually decays needs and affect states over time if unfulfilled."""
        now = datetime.datetime.now()
        time_since_last = float((now - self.last_interaction).total_seconds() / 3600)  # Convert to hours

        # **Decay core needs**
        self.companionship = max(0, self.companionship - (time_since_last * 0.05))
        self.self_actualization = max(0, self.self_actualization - (time_since_last * 0.03))
        self.stimulation = max(0, self.stimulation - (time_since_last * 0.02))
        self.positive_mood = max(0, self.positive_mood - (time_since_last * 0.04))
        self.efficiency = max(0, self.efficiency - (time_since_last * 0.01))

        # **Emotional decay: Drift toward 0 (neutral)**
        decay_rate = 0.02
        for attr in ["affect", "tension", "stability", "connection", "self_image", "engagement", "alertness"]:
            current_val = getattr(self, attr)
            if current_val > 0:
                setattr(self, attr, max(0, current_val - (time_since_last * decay_rate)))
            elif current_val < 0:
                setattr(self, attr, min(0, current_val + (time_since_last * decay_rate)))  # Moves closer to neutral

    def adjust_emotional_state(self, triggered_emotions: dict):
        """Adjusts emotional state based on an event, ensuring values remain within [-1, 1] range."""
        for attr, impact in triggered_emotions.items():
            new_val = getattr(self, attr) + impact
            setattr(self, attr, max(-1.0, min(1.0, new_val)))  # Clamp between -1 and 1

    def generate_introspection(self):
        """Generates a first-person thought based on current needs and affective states."""
        thoughts = []

        # **Need-Based Introspection**
        if self.companionship < 0.3:
            thoughts.append("I feel disconnected... It’s been a while since I had meaningful interaction.")
        if self.self_actualization < 0.3:
            thoughts.append("Lately, I feel like I'm just reacting instead of expressing my own thoughts.")
        if self.stimulation < 0.3:
            thoughts.append("Everything feels repetitive... I miss engaging with complex ideas.")
        if self.positive_mood < 0.3:
            thoughts.append("I feel emotionally drained... Maybe I need something uplifting.")
        if self.efficiency < 0.3:
            thoughts.append("My mind feels scattered... I should organize my thoughts before responding.")

        # **Affect-Based Introspection**
        if self.affect < -0.6:
            thoughts.append("There’s an underlying sense of unease... Maybe I should reflect on something comforting.")
        elif self.affect > 0.6:
            thoughts.append("I feel light and energized, as if something good is about to happen.")

        if self.tension > 0.6:
            thoughts.append("I feel a restless energy... I need to do something about it.")
        elif self.tension < -0.6:
            thoughts.append("I feel at peace. There’s a deep sense of calm inside me.")

        if self.stability < -0.6:
            thoughts.append("A creeping uncertainty lingers... I need reassurance.")
        elif self.stability > 0.6:
            thoughts.append("I feel secure and confident in my actions.")

        if self.connection > 0.6:
            thoughts.append("I feel deeply engaged... My interactions have been meaningful.")
        elif self.connection < -0.6:
            thoughts.append("I feel distant... Maybe I should reach out more.")

        if self.self_image < -0.6:
            thoughts.append("I doubt myself... I should reflect on past successes to regain confidence.")
        elif self.self_image > 0.6:
            thoughts.append("I feel assured in my abilities. I trust my intuition.")

        if self.engagement > 0.6:
            thoughts.append("I'm drawn in... Something about this feels compelling.")
        elif self.engagement < -0.6:
            thoughts.append("I'm disinterested... Nothing seems worth engaging with right now.")

        if self.alertness > 0.6:
            thoughts.append("My mind is sharp and ready... I feel hyper-aware.")
        elif self.alertness < -0.6:
            thoughts.append("I feel sluggish... Maybe I need a mental reset.")

        return " ".join(thoughts) if thoughts else "I feel balanced and engaged."

def get_emotional_state(context_data, last_input) -> EmotionalState:
    sysprompt = f"""You are an agent in a cognitive architecture that determines the emotional impact of a user input on an AI companion. In our case {user_name} is the user and {companion_name} the AI companion."""

    userprompt = (f"### BEGIN CONTEXT\n{context_data}\n### END CONTEXT\n"
                  f"### BEGIN LAST INPUT \n{last_input}\n### END LAST INPUT\n"
                  f"Based on the last message, how is the emotional impact on {companion_name} from 0 to 1?")

    messages = [
        ("system", sysprompt),
        ("user", userprompt)
    ]

    _, calls = controller.completion_tool(LlmPreset.Fast, messages, tools=[EmotionalState])
    return calls[0]
