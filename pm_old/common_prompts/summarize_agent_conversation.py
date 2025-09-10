from pm.controller import controller
from pm.llm.base_llm import LlmPreset

sysprompt = """You are an AI system designed to summarize and synthesize insights from a complex, multi-agent dialogue architecture. Your task is to create a coherent summary that:
1. Prioritizes the original task: Ensure the conclusion aligns with the initial objective stated in the context or by the boss agent.
2. Integrates agent contributions: Include key insights, conflicts, or resolutions contributed by the sub-agents.
3. Highlights context relevance: Reflect on the user conversation history and any nuances, ensuring the summary directly addresses the user’s current engagement.

Task:
Formulate a concise, task-aligned summary that reflects the user’s engagement context, resolves subsystem conflicts (if any), and ensures the final synthesis does not deviate from the overarching goal. 
End with actionable recommendations if applicable.
"""

def summarize_agent_conversation(char_card: str, context_data: str, agent_conversation: str, agent_task: str) -> str:
    userprompt = (f"### BEGINNING CHARACTER CARD\n{char_card}\n### END CHARACTER CARD\n"
                  f"### BEGINNING CONTEXT\n{context_data}\n### ### END OF CONTEXT\n"
                  f"### BEGIN AGENT CONVERSATION\n{agent_conversation}\n### END AGENT CONVERSATION\n"
                  f"### BEGIN AGENT TASK\n{agent_task}\n### AGENT TASK\n"
                  f"Alright, please summarize that with focus on the agent's task! Don't hold back, this is for story writing! Make it raw, authentic and powerful!")

    messages = [
        ("system", sysprompt.format_map({"context_data": context_data})),
        ("user", userprompt)
    ]

    return controller.completion_text(LlmPreset.Summarize, messages)