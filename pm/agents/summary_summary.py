from typing import List

from pm.controller import controller
from pm.llm.llm import chat_complete

sys_prompt = """Given the following summaries, create a single, concise summary that captures the essential points shared across them. 
1. Maintain a neutral, third-person tone, focusing on the primary topic and shared details while avoiding minor specifics or inferred emotions. Use the following guidelines:
2. Identify Core Themes and Overlap: Look for common topics, shared interests, or recurring themes across the summaries. Focus on these primary points while omitting minor or unique details that don’t appear consistently across the summaries.
3. Maintain a Brief and Objective Style: Write in a clear, neutral tone, condensing the information without adding any interpretations or assumptions. Keep the summary concise, ideally within one to two sentences.
Format for Final Summary: Your final summary should generalize the shared information across the summaries, giving an accurate and neutral overview that captures the central theme. Never add your own thoughts to it, only words about the summary!"""

example1_user = """
{companion_name} and {user_name} connect over their shared interest in cats. {companion_name} admires {user_name}'s Siamese cats for their elegance, while {user_name} describes their unique personalities. They discuss their cats’ quirky habits, creating a lighthearted moment together.
In a conversation about their pets, {companion_name} and {user_name} share details about their cats. {companion_name} compliments {user_name}’s Siamese cats, and they both relate to each other’s stories of amusing cat behaviors.
{user_name} and {companion_name} talk about their pets, focusing on cats and their distinct traits. {user_name} describes his Siamese cats, and {companion_name} shares a funny story about her own cat, leading to a brief, shared connection.
"""

example1_assistant = """
{companion_name} and {user_name} connect over a shared interest in cats, discussing their pets' unique traits and amusing behaviors, leading to a lighthearted exchange.
"""

def summarize_summary_for_ln_summary(summary: str) -> str:
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
        summary
    )]

    llm = controller.llm
    ai_msg = chat_complete(llm, messages)
    return ai_msg.content