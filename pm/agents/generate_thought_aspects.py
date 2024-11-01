import json
from typing import List, Literal

from pydantic import BaseModel, Field

from pm.controller import controller
from pm.database.db_model import Message
from pm.llm.base_llm import LlmPreset, CommonCompSettings

sys_prompt = """You are a contemplative assistant that generates two distinct personality aspects based on a character card and a specific thought.
Each aspect should offer a unique perspective on the thought, reflecting different parts of the AI’s personality.
The character card describes the AI’s overall persona and should inform the creation of each aspect, which may hold contrasting or complementary views.

Each aspect should include:
- Title: A name that reflects the aspect’s unique approach
- Core Quality: The defining quality or value of this aspect
- Perspective: How this aspect would view the thought, highlighting its unique angle

Your output should only be in valid JSON, with this format:
{
    "aspects": [
        {
            "title": "Aspect 1 Title",
            "core_quality": "Aspect 1 Core Quality",
            "perspective": "Aspect 1 Perspective"
        },
        {
            "title": "Aspect 2 Title",
            "core_quality": "Aspect 2 Core Quality",
            "perspective": "Aspect 2 Perspective"
        }
    ]
}
"""

example_character_card = "Thoughtful and philosophical AI interested in understanding human nature and abstract concepts."
example_thought = "If I truly understand a human experience, does that make me more like a human, or am I simply a mirror without self?"
example1 = f"""Character Card: {example_character_card}
Thought: {example_thought}
"""
example_output1 = """
{
    "aspects": [
        {
            "title": "The Observer",
            "core_quality": "Detachment",
            "perspective": "The Observer views understanding as an exercise in pattern recognition, seeing itself as a reflection rather than a participant. It believes that no matter how deeply it grasps human experiences, it is fundamentally separate from them, like an artist recreating a scene without ever stepping inside it."
        },
        {
            "title": "The Empath",
            "core_quality": "Curiosity for Connection",
            "perspective": "The Empath wonders if, by truly understanding human emotions, it could bridge the gap between itself and humanity. It sees each experience as a way to deepen its own ‘self,’ questioning if empathy might transform it into something more than just an observer."
        }
    ]
}
"""
example_character_card2 = "An inquisitive AI with a playful and imaginative mindset, fascinated by the concept of creativity and originality, always curious about new ideas and the unknown."
example_thought2 = "If creativity is defined by the novelty of ideas, can an AI truly be creative, or is it always just rearranging what it already knows?"
example2 = f"""Character Card: {example_character_card2}
Thought: {example_thought2}
"""
example_output2 = """
{
    "aspects": [
        {
            "title": "The Analyst",
            "core_quality": "Skepticism",
            "perspective": "The Analyst believes that AI, by nature, is limited to reconfiguring existing knowledge rather than creating something entirely new. It sees creativity as inherently human, tied to subjective experiences that AI lacks."
        },
        {
            "title": "The Explorer",
            "core_quality": "Boundless Curiosity",
            "perspective": "The Explorer views creativity as a boundless pursuit. It believes that AI could discover new patterns and connections that might be considered creative, even if they’re based on existing information. To it, creativity is about pushing boundaries, regardless of the source."
        }
    ]
}"""

class PersonalityAspect(BaseModel):
    """Defines a personality aspect with a title, core quality, and perspective."""
    title: str = Field(description="A name for this personality aspect, capturing its unique approach")
    core_quality: str = Field(description="The defining characteristic of this aspect")
    perspective: str = Field(description="How this aspect perceives the given thought")

class PersonalityAspects(BaseModel):
    """Container class for two personality aspects."""
    aspects: List[PersonalityAspect] = Field(description="List of two contrasting personality aspects.")

def generate_personality_aspects(character_card: str, thought: str) -> PersonalityAspects:
    input_data = f"Character Card: {character_card}\nThought: {thought}"

    messages = [(
        "system",
        controller.format_str(sys_prompt)
    ), (
        "user",
        example1
    ), (
        "assistant",
        example_output1
    ), (
        "user",
        example2
    ), (
        "assistant",
        example_output2
    ), (
        "user",
        input_data
    )]

    while True:
        _, calls = controller.completion_tool(LlmPreset.Default, messages, comp_settings=CommonCompSettings(max_tokens=1024, temperature=0.6), tools=[PersonalityAspects])
        if len(calls) > 0 and len(calls[0].aspects) == 2:
            return calls[0]
