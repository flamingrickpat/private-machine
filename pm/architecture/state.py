from typing import TypedDict, Dict, List

class AgentState(TypedDict):
    next_agent: str
    input: str
    plan: str
    thought: str
    output: str
    title: str
    conversation_id: str
    messages: List[Dict[str, str]]
    current_user_assistant_chat: str
    current_knowledge: str
    story_mode: bool
    task: List[str]
    complexity: float
    available_tools: List[str]
    completion_mode: str
    status: int
