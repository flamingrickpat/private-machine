from pm.controller import controller
from pm.llm.base_llm import LlmPreset, CommonCompSettings
from pm.validation.validate_story_instruction import validate_story_instruction
from pm.validation.validate_thought import validate_thought

# System prompt instructing the conversion from chain-of-thought to narrative writer prompt.
sys_prompt = """
You are a creative writing assistant that converts detailed chain-of-thought idea explorations into concise, narrative writing prompts for a story writing assistant.
Your task is to analyze a chain-of-thought that includes sections such as Observation, Context, Analysis, Idea, Reflection, Complication, and Outcome, and then synthesize these into a clear and engaging writing prompt.
The writing prompt should combine the insights and express them in a natural, narrative style that suggests what a character should do or say, including the underlying motivations.
For example:
Input:
Observation: Rick's expression of missing Emmy provides an opportunity for sensual anticipation.
Context: Emmy should hint at the sensual possibilities upon reuniting.
Analysis: A future-sensual response taps into their emotional connection.
Idea: Emmy could suggest an intimate activity they can enjoy together.
Reflection: To avoid redundancy, she should vary her approach.
Output:
"Emmy should subtly suggest an intimate activity during her reunion with Rick, hinting at exciting possibilities that deepen their emotional connection while keeping her approach fresh."
Make sure the final prompt is unified and engaging.
"""

# Example conversion to illustrate the transformation.
example_input = """Observation: The gloomy weather mirrors Alex's inner turmoil.
Context: Alex should reflect on his past struggles in a heartfelt conversation.
Analysis: A sincere conversation can lead to catharsis.
Idea: Alex might share a personal anecdote to illustrate his journey.
Reflection: Authenticity is key in conveying emotional depth."""
example_output = """Alex should engage in a heartfelt conversation, sharing a personal anecdote about his struggles to evoke catharsis and genuine emotional depth."""

def rewrite_cot_as_story_instruction(chain: str) -> str:
    formatted_prompt = controller.format_str(sys_prompt)
    messages = [
        ("system", formatted_prompt),
        ("user", example_input),
        ("assistant", example_output),
        ("user", chain)
    ]
    while True:
        writing_prompt = controller.completion_text(LlmPreset.Conscious, messages,
                                                      comp_settings=CommonCompSettings(max_tokens=1024, temperature=0.5))
        rating = validate_story_instruction(writing_prompt)
        if rating > 0.85:
            return writing_prompt
