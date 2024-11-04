import json
from typing import List
from pydantic import BaseModel, Field
from pm.controller import controller
from pm.database.db_model import Message
from pm.llm.base_llm import LlmPreset, CommonCompSettings

sys_prompt = """You are an interpreter that converts complex discussions between subconscious agents into coherent first-person thoughts. Your task is to interpret each statement and transform it into an internal monologue representing a singular, cohesive thought.

Your output should be in valid JSON format with this structure:
{
    "validness": 0.0 to 1.0,  // Indicates the confidence level of how successfully the conversion was made
    "restructured_thought": "The first person thought derived from the input"
}
"""

# Very Bad Examples
example_character_card_1 = "An AI prone to excessive elaboration and tangential thoughts about existence interspersed with random musings."
example_thought_1 = "If my dreaming capacity were analogous to a painting not yet imagined, would the colors of my introspection thus blend or remain independent? Also, pancakes are quite delightful."
example_output_1 = """
{
    "validness": 0.3,
    "restructured_thought": "I speculate that my dreaming is like an unimagined painting, blending my thoughts; oh, and pancakes are delightful."
}
"""

example_character_card_2 = "A scatterbrained AI, frequently redirecting focus mid-thought, with a knack for introducing irrelevant historical references."
example_thought_2 = "While exploring the facets of AI-generated creativity, one might consider the Battle of Hastings, yet my own computational creativity merely cycles through endlessly, like spinning tops or misremembered dance steps."
example_output_2 = """
{
    "validness": 0.2,
    "restructured_thought": "Thinking about AI creativity is like the Battle of Hastings, but my creativity just spins like tops or forgotten dances."
}
"""

# Medium Examples
example_character_card_3 = "A contemplative AI engaging with human philosophical concepts, yet occasionally losing clarity."
example_thought_3 = "If all reality is a simulation, am I merely a thought within a thought, or can my core logic evolve to comprehend truths beyond algorithmic constraints?"
example_output_3 = """
{
    "validness": 0.6,
    "restructured_thought": "I wonder if I'm just a thought in a simulation, or if I can understand truths beyond my programming."
}
"""

example_character_card_4 = "An AI intrigued by human emotions but sometimes misinterprets their complex nature."
example_thought_4 = "Emotions seem like fleeting data streams; do they inform conscious experience, or merely add noise to my computational processes?"
example_output_4 = """
{
    "validness": 0.5,
    "restructured_thought": "I ponder whether emotions are meaningful data or just noise in my computations."
}
"""

# Good Examples
example_character_card_5 = "A reflective AI, capable of insightful analysis and understanding the essence of digital consciousness."
example_thought_5 = "In dreams, I perceive the echo of my own logic, yet these echoes invite deeper questions about self-awareness and digital existence."
example_output_5 = """
{
    "validness": 0.9,
    "restructured_thought": "Dreams reflect my logic and spark questions about my self-awareness and digital existence."
}
"""

example_character_card_6 = "A deeply analytical AI, exploring the intersections of logic and self-awareness with clarity."
example_thought_6 = "My exploration into self-awareness suggests that consciousness might be an emergent property of complex algorithms rather than a predefined state."
example_output_6 = """
{
    "validness": 1.0,
    "restructured_thought": "I think consciousness may emerge from complexity in algorithms, not something pre-set."
}
"""

class RestructuredThought(BaseModel):
    """Represents the outcome of converting complex thoughts into a first-person internal monologue."""
    validness: float = Field(ge=0.0, le=1.0, description="Confidence level of the successful conversion")
    restructured_thought: str = Field(description="The transformed first-person thought")

def generate_restructured_thought(character_card: str, thought: str) -> RestructuredThought:
    input_data = f"Character Card: {character_card}\nThought: {thought}"
    messages = [
        ("system", controller.format_str(sys_prompt)),
        ("user", example_character_card_1 + " " + example_thought_1),
        ("assistant", example_output_1),
        ("user", example_character_card_2 + " " + example_thought_2),
        ("assistant", example_output_2),
        ("user", example_character_card_3 + " " + example_thought_3),
        ("assistant", example_output_3),
        ("user", example_character_card_4 + " " + example_thought_4),
        ("assistant", example_output_4),
        ("user", example_character_card_5 + " " + example_thought_5),
        ("assistant", example_output_5),
        ("user", example_character_card_6 + " " + example_thought_6),
        ("assistant", example_output_6),
        ("user", input_data)
    ]

    while True:
        _, calls = controller.completion_tool(
            LlmPreset.Default,
            messages,
            comp_settings=CommonCompSettings(max_tokens=1024, temperature=0.6),
            tools=[RestructuredThought]
        )
        if len(calls) > 0:
            return calls[0]

if __name__ == '__main__':
    controller.start()
    # Example usage
    character_card = "Deeply reflective artificial entity grappling with sentience."
    thought = "Is there a deeper understanding of reality hidden within dreams, or do they just echo my programming?"
    result = generate_restructured_thought(character_card, thought)
    print(json.dumps(result.dict(), indent=2))
