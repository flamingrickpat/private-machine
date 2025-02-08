from typing import List, Tuple

from pm.character import companion_name
from pm.controller import controller
from pm.llm.base_llm import LlmPreset, CommonCompSettings


def summarize_messages(msgs: List[str]):
    sysprompt = "You are a helpful assistant. You summarize a conversation from the perspective of {companion_name} and in first person. Example: {user_name} talked to me about video games and food today."
    msg_string = "\n".join(msgs)

    prompt = [
        ("system", sysprompt),
        ("user", f"### BEGIN MESSAGES\n{msg_string}\n### END MESSAGES\nPlease summarize what happened during the conversation from {companion_name}'s perspective in first person."),
    ]

    content = controller.completion_text(LlmPreset.Summarize, prompt, comp_settings=CommonCompSettings(temperature=0.1, max_tokens=1024))
    return content