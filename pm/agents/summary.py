from typing import List

from pm.controller import controller
from pm.database.db_model import Message
from pm.llm.base_llm import LlmPreset, CommonCompSettings

sys_prompt = """You are a helpful assistant that summarizes chatlogs for a compact, token-saving storage and later usage.
Analyze the following chat log between two people, {user_name} and {companion_name}. Your task is to create a concise summary that captures the key points of their conversation, using a neutral tone and third-person narration. Follow these instructions to ensure clarity and accuracy:
1. Summarize Only the Main Points: Focus on the core topic or shared interests discussed by {user_name} and {companion_name}, and avoid including minor or unrelated details. The summary should highlight their main points of connection without adding extra context or interpretation.
2. Avoid Adding Emotions or Assumptions: Do not include personal feelings, inferred motivations, or assumed emotions. Limit the summary to only the information explicitly provided in the chat log.
3. Use a Concise, Objective Style: The summary should be brief, clear, and to the point. Use professional, third-person language that would effectively replace the original chat log while preserving the main idea.
4. Summary Format: Aim for two to three sentences that encapsulate the conversation’s main topics and any notable details shared by {user_name} and {companion_name}. Here’s an example of the style to follow:
Never add your own thoughts to it, only words about the summary!"""

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
{user_name} and {companion_name} connect over their mutual love for cats. She compliments his Siamese cats, admiring their elegance and blue eyes, while {user_name} describes their royal-like personalities. 
{companion_name} mentions how her own cat loves to sit on her laptop whenever she’s working, which {user_name} seems to relate to, creating a shared moment around their pets’ amusing behaviors.
"""

def summarize_messages_for_l0_summary(messages: List[Message]) -> str:
    message_block = "\n".join([f"{controller.config.companion_name if x.role == 'assistant' else controller.config.user_name}: {x.text}" for x in messages])

    messages = [(
        "system",
        controller.format_str(sys_prompt)
    ), (
        "user",
        controller.format_str(example1_user)
    ), (
        "assistant",
        controller.format_str(example1_assistant)
    ), (
        "user",
        message_block
    )]

    content = controller.completion_text(LlmPreset.Default, messages, comp_settings=CommonCompSettings(max_tokens=1024))
    return content