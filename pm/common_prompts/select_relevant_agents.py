from pydantic import BaseModel, Field

from pm.character import companion_name
from pm.controller import controller
from pm.llm.base_llm import LlmPreset

class AgentSelection(BaseModel):
    agent1: str
    agent2: str


def select_relevant_agents(agents, context_data):
    block = "\n".join([f"{x.name}: {x.description}" for x in agents])
    short = ",".join([f"{x.name}" for x in agents])

    sysprompt = f"""You are an agent in a cognitive architecture and you must select the 2 most relevant agents based on the chat log for {companion_name}. These are all agents: {block}\nYour options are: 
    [{short}]"""

    userprompt = (f"### BEGIN CONTEXT\n{context_data}\n### END CONTEXT\n"
                  f"Based on the last message, which agents are most relevant for {companion_name}? Consider her character description. Your options are: [{short}] for agent1 and agent2")

    messages = [
        ("system", sysprompt),
        ("user", userprompt)
    ]

    _, calls = controller.completion_tool(LlmPreset.Default, messages, tools=[AgentSelection])
    return [calls[0].agent1, calls[0].agent2]
