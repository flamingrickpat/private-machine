import json
from typing import List, Literal

from pydantic import BaseModel, Field

from pm.controller import controller
from pm.database.db_model import Message
from pm.llm.base_llm import LlmPreset, CommonCompSettings

sys_prompt = """You are a helpful assistant that extracts facts from a conversation.
These facts are supposed to enable querying a vector store with keywords for information in a quick and efficient way.
The user name is {user_name} and the AI companion name is {companion_name}.
You output valid JSON and valid JSON only for the extracted facts!
In this format:
{json_schema}
"""

example1_user = """
{companion_name}: Hey there!  
{user_name}: Hey, how's it going?
{companion_name}: I like cats; they're just the cutest, aren't they?  
{user_name}: Absolutely, I have two Siamese at home. They never cease to entertain me.  
{companion_name}: Oh, Siamese! They're such elegant-looking cats. I love their blue eyes.  
{user_name}: True, and they have quite the personality too. They're like little kings in the house.  
{companion_name}: Haha, I bet. Mine just loves sitting on my laptop whenever I'm trying to work.  
{user_name}: Speaking of your laptop, I was planning some travel ideas and was researching online. Been anywhere exciting lately?  
"""

example1_assistant = """
{
  "facts": [
    {"fact": "{user_name} has two Siamese cats at home.", "category": "user", "importance": 0.5},
    {"fact": "{user_name} finds Siamese cats entertaining.", "category": "user", "importance": 0.3},
    {"fact": "{companion_name} likes cats and finds them cute.", "category": "ai_companion", "importance": 0.3},
    {"fact": "{companion_name} has a cat that likes sitting on their laptop.", "category": "ai_companion", "importance": 0.4},
    {"fact": "{user_name} is planning some travel ideas and is researching online.", "category": "user", "importance": 0.6}
  ]
}"""

example2_user = """
{companion_name}: Hey, are you into any sports?  
{user_name}: Definitely, I play tennis a few times a week. Helps me stay active and relieve stress.  
{companion_name}: Nice! I've always wanted to try tennis but never got around to it.  
{user_name}: You should! It's great once you get the hang of it.  
{companion_name}: Any other activities you do for fun?  
{user_name}: I also dabble in painting, mostly landscapes. It’s relaxing.  
"""

example2_assistant = """
{
  "facts": [
    {"fact": "{user_name} plays tennis a few times a week.", "category": "user", "importance": 0.6},
    {"fact": "{user_name} finds tennis helpful for staying active and relieving stress.", "category": "user", "importance": 0.5},
    {"fact": "{companion_name} has an interest in trying tennis.", "category": "ai_companion", "importance": 0.3},
    {"fact": "{user_name} engages in painting, focusing on landscapes.", "category": "user", "importance": 0.4}
  ]
}
"""


example3_user = """
{companion_name}: Do you enjoy cooking?  
{user_name}: Yes, actually! I love experimenting with new recipes. I recently tried making a Mediterranean dish.  
{companion_name}: Oh, sounds delicious! Do you have a favorite type of cuisine?  
{user_name}: Probably Italian, though Mediterranean is a close second.  
{companion_name}: I'll have to try some new recipes myself. Do you cook often?  
{user_name}: Mostly on weekends when I have more time. It’s kind of a hobby now.  
"""

example3_assistant = """
{
  "facts": [
    {"fact": "{user_name} enjoys cooking and experimenting with new recipes.", "category": "user", "importance": 0.6},
    {"fact": "{user_name} recently made a Mediterranean dish.", "category": "user", "importance": 0.4},
    {"fact": "{user_name}'s favorite cuisine is Italian, with Mediterranean as a close second.", "category": "user", "importance": 0.5},
    {"fact": "{user_name} mostly cooks on weekends when they have more time.", "category": "user", "importance": 0.4},
    {"fact": "{companion_name} expresses interest in trying new recipes.", "category": "ai_companion", "importance": 0.3}
  ]
}
"""

categories = [controller.config.user_name.lower(),
              controller.config.companion_name.lower(),
              "environment"]

class ExtractedFact(BaseModel):
    """Describes a fact."""
    fact: str = Field(description="text representation of the fact")
    category: Literal[tuple(categories)] = Field(description=f"category can be '{controller.config.user_name}', '{controller.config.companion_name}' or 'environment'")
    importance: float = Field(description="how important the fact is. small miniscule facts such as favorite dog breed are lower and important facts such as new job are higher. from 0 to 1.",
                              ge=0, le=1)

class ExtractedFacts(BaseModel):
    """Container class for the facts."""
    facts: List[ExtractedFact] = Field(description="List of facts.")

def extract_facts_from_messages(messages: List[Message]) -> ExtractedFacts:
    message_block = "\n".join([f"{controller.config.companion_name if x.role == 'assistant' else controller.config.user_name}: {x.text}" for x in messages])

    messages = [(
        "system",
        controller.format_str(sys_prompt, extra={"json_schema": json.dumps(ExtractedFacts.model_json_schema())})
    ), (
        "user",
        controller.format_str(example1_user)
    ), (
        "assistant",
        controller.format_str(example1_assistant)
    ), (
        "user",
        controller.format_str(example2_user)
    ), (
        "assistant",
        controller.format_str(example2_assistant)
    ), (
        "user",
        controller.format_str(example3_user)
    ), (
        "assistant",
        controller.format_str(example3_assistant)
    ), (
        "user",
        message_block
    )]
    while True:
        _, calls = controller.completion_tool(LlmPreset.Default, messages, comp_settings=CommonCompSettings(max_tokens=1024, temperature=0), tools=[ExtractedFacts])
        if len(calls) > 0:
            calls = calls[0]
            break
    return calls