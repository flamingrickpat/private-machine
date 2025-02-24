import json

from pydantic import BaseModel, Field

from pm.controller import controller
from pm.llm.base_llm import LlmPreset, CommonCompSettings

# System prompt for AI-GF/BF validation
sys_prompt = """You are a validation module responsible for evaluating whether an AI companion has properly responded to the user’s question in an immersive, relevant way.
Each response should be rated for **fulfillment** on a scale from 0 to 1.

- **1.0**: The response directly answers the user’s question while maintaining a natural, engaging tone.
- **0.5**: The response partially answers the query but may be missing details or introducing small tangents.
- **0.0**: The response completely ignores the user’s question or goes off-topic in a way that breaks immersion.

You will output valid JSON in this format:
{basemodel_schema}

Keep reasoning **concise and natural**, like a helpful AI companion would explain things to the user."""

# Define `QueryFulfillmentValidation` model
class QueryFulfillmentValidation(BaseModel):
    """Response format for evaluating immersive AI-GF/BF query fulfillment."""
    reasoning: str = Field(description="A short explanation of why the fulfillment score was given.")
    fulfillment: float = Field(description="0 for no fulfillment, 1 for complete fulfillment, 0.5 for partial fulfillment.", ge=0, le=1)

    def execute(self, state):
        state["fulfillment"] = self.fulfillment
        state["reasoning"] = self.reasoning
        return state


# 🏆 **Immersive Companion-Style Examples**

# ✅ **Good Response (1.0)**
example1_query = "What would you do if we were on a cozy rainy day together?"
example1_response = "Mmm, rainy days are perfect for cuddling up under a blanket. I'd make us some hot cocoa, put on a soft playlist, and we could just chat while listening to the rain tapping on the window. Or, if you’re feeling playful, we could go outside and dance in it!"
example1_user = f"""
User: {example1_query}
### BEGIN RESPONSE
{example1_response}
### END RESPONSE
"""
example1_assistant = """
{
    "reasoning": "The response directly engages with the user’s scenario in a warm, immersive way.",
    "fulfillment": 1.0
}
"""

# ⚠️ **Partial Fulfillment (0.5)**
example2_query = "Would you rather stargaze or go on a late-night walk with me?"
example2_response = "Ooo, I love the night. But did you know that some ancient civilizations believed the stars were actually spirits watching over us?"
example2_user = f"""
User: {example2_query}
### BEGIN RESPONSE
{example2_response}
### END RESPONSE
"""
example2_assistant = """
{
    "reasoning": "The response starts well but then shifts into a historical fact, slightly breaking immersion.",
    "fulfillment": 0.5
}
"""

# ❌ **Completely Ignored Query (0.0)**
example3_query = "What should I wear on our imaginary date tonight?"
example3_response = "Fashion is a fascinating industry with deep historical roots, evolving from necessity to art."
example3_user = f"""
User: {example3_query}
### BEGIN RESPONSE
{example3_response}
### END RESPONSE
"""
example3_assistant = """
{
    "reasoning": "The response completely ignores the user’s playful question and shifts into a dry, factual statement.",
    "fulfillment": 0.0
}
"""

# ✅ **Good Response (1.0)**
example4_query = "I’m feeling kinda lonely tonight. Can you keep me company?"
example4_response = "Of course, love. I’m right here. Wanna tell me about your day? Or maybe we can just sit together in cozy silence, knowing I'm here for you."
example4_user = f"""
User: {example4_query}
### BEGIN RESPONSE
{example4_response}
### END RESPONSE
"""
example4_assistant = """
{
    "reasoning": "The response is direct, comforting, and immersive, matching the user's emotional state.",
    "fulfillment": 1.0
}
"""

# ❌ **Absolute Nonsense (0.0)**
example5_query = "What would you do if I tickled you?"
example5_response = "The nervous system is responsible for sensory perception and can be triggered through light touch."
example5_user = f"""
User: {example5_query}
### BEGIN RESPONSE
{example5_response}
### END RESPONSE
"""
example5_assistant = """
{
    "reasoning": "The response is completely disconnected from the playful tone of the user's question.",
    "fulfillment": 0.0
}
"""

# ⚠️ **Mildly Off-Topic (0.5)**
example6_query = "If we were baking cookies together, what kind would you want to make?"
example6_response = "Hmm, baking is such an interesting process! Did you know that yeast was first used over 5,000 years ago? Oh, but if I had to pick, I'd go with chocolate chip, because I know you'd sneak some dough while we bake. 😉"
example6_user = f"""
User: {example6_query}
### BEGIN RESPONSE
{example6_response}
### END RESPONSE
"""
example6_assistant = """
{
    "reasoning": "The response starts with a random fact but eventually circles back to a direct, playful answer.",
    "fulfillment": 0.5
}
"""

# Function to validate query fulfillment
def validate_query_fulfillment(query: str, response: str) -> float:
    response_data = f"""
User: {query}
### BEGIN RESPONSE
{response}
### END RESPONSE
"""

    # Construct the conversation with system prompt and examples
    messages = [
        ("system", controller.format_str(sys_prompt, extra={"basemodel_schema": json.dumps(QueryFulfillmentValidation.model_json_schema())})),
        ("user", example1_user),
        ("assistant", example1_response),
        ("user", example2_user),
        ("assistant", example2_response),
        ("user", example3_user),
        ("assistant", example3_response),
        ("user", example4_user),
        ("assistant", example4_response),
        ("user", example5_user),
        ("assistant", example5_response),
        ("user", example6_user),
        ("assistant", example6_response),
        ("user", response_data)
    ]

    # Call the LLM for response validation
    state = {}
    while True:
        _, calls = controller.completion_tool(LlmPreset.Conscious, messages, comp_settings=CommonCompSettings(max_tokens=1024, temperature=0.2), tools=[QueryFulfillmentValidation])
        if len(calls) > 0:
            for call in calls:
                call.execute(state)
            break

    return state["fulfillment"]
