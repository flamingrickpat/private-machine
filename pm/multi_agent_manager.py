import random
from datetime import datetime
from typing import List, Type, Tuple, Callable

from langgraph.graph import StateGraph
from pydantic import BaseModel, Field

import json
import logging
import uuid
from typing import Literal, TypedDict, Dict, Any, List

from lancedb import LanceDBConnection
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from jinja2 import Template
from scripts.regsetup import description
from spacy.symbols import agent
from torch.utils.data.backward_compatibility import worker_init_fn

from pm.agents.completion_mode import determine_completion_mode, CompletionModeResponseMode
from pm.agents.determine_complexity import determine_complexity
from pm.agents.tool_selection import determine_tools
from pm.clustering.summarize import cluster_and_summarize, high_level_summarize
from pm.config.config import read_config_file
from pm.consts import COMPLEX_THRESHOLD
from pm.controller import controller
from pm.database.db_helper import fetch_messages, fetch_messages_no_summary, rank_table, fetch_relations, \
    fetch_messages_as_string
from pm.database.db_model import Message, MessageSummary
from pm.database.load_messages import build_prompt
from pm.llm.llm import chat_complete
from pm.llm.tools_parser_local import PydanticToolsParserLocal
from pm.prompts.prompt_boss import prompt_boss
from pm.prompts.prompt_main_agent import build_sys_prompt_conscious_assistant
from pm.prompts.prompt_main_agent_story import build_sys_prompt_conscious_story
from pm.prompts.prompt_subagent import prompt_subagent
from pm.rag.tools import hybrid_search_documentation, RagState, answer_user, add_document
from pm.tools.generate_documentation import create_header_functions, create_documentation_functions, \
    create_documentation_json
from pm.utils.misc_utils import get_key_from_value

logger = logging.getLogger(__name__)


class SubAgentState(TypedDict):
    next_agent: str
    confidence: float
    conclusion: str
    agent_idx: int

class AgentMessage(BaseModel):
    created_at: datetime = Field(default_factory=datetime.now)
    text: str
    public: bool = Field(default=True)
    from_tool: bool = Field(default=False)

class Agent(BaseModel):
    name: str
    system_prompt: str = Field(default_factory=str)
    description: str
    goal: str
    task: str = Field(default_factory=str)
    tools: List[Type[BaseModel]] = Field(default_factory=list)
    functions: List[Callable] = Field(default_factory=list)
    messages: List[AgentMessage] = Field(default_factory=list)


class Conclusion(BaseModel):
    """Final conclusion format."""
    confidence: float
    conclusion: str

    def execute(self, state):
        state["confidence"] = self.confidence
        state["conclusion"] = self.conclusion
        return {}

class WaitForMoreInformation(BaseModel):
    """How far have we come for finding a conclusion?"""
    confidence: float

    def execute(self, state):
        pass


class BossAgentChoice(BaseModel):
    """Container for your possible choices."""
    choice: Conclusion | WaitForMoreInformation

    def execute(self, state):
        self.choice.execute(state)


def compile_message(agents: List[Agent], cur_agent: Agent) -> List[Tuple[str, str]]:
    # Collect all messages with their roles and agent names
    all_messages = []

    for agent in agents:
        role = "assistant" if agent == cur_agent else "user"
        for message in agent.messages:
            # Append each message with role, text, timestamp, and agent name if it's a user
            all_messages.append({
                "role": "user" if message.from_tool else role,
                "text": message.text if role == "assistant" else f"{agent.name}: {message.text}",
                "created_at": message.created_at,
                "is_private": not message.public if role == "user" else False
            })

    # Sort messages by timestamp
    all_messages.sort(key=lambda x: x["created_at"])

    # Compile messages into alternating assistant/user blocks
    compiled_messages = []
    user_message_buffer = []  # Buffer for consecutive user messages

    for msg in all_messages:
        if msg["role"] == "assistant" and not msg["is_private"]:
            # Flush user messages buffer before adding an assistant message
            if user_message_buffer:
                compiled_messages.append(("user", "\n".join(user_message_buffer)))
                user_message_buffer = []
            # Add the assistant message
            compiled_messages.append((msg["role"], msg["text"]))
        elif msg["role"] == "user":
            # Add to user buffer if it's a public user message
            if not msg["is_private"]:
                user_message_buffer.append(msg["text"])

    # Flush any remaining user messages
    if user_message_buffer:
        compiled_messages.append(("user", "\n".join(user_message_buffer)))

    return compiled_messages

def _execute_router(state: SubAgentState, agents: List[Agent]):
    if state["next_agent"] == "initial":
        state["next_agent"] = agents[0].name
        state["agent_idx"] = 1
    else:
        if state["agent_idx"] >= len(agents):
            state["agent_idx"] = 0
        state["next_agent"] = agents[state["agent_idx"]].name
        state["agent_idx"] += 1
    return state


def _execute_agent(state: SubAgentState, agents: List[Agent]):
    cur_agent = state["next_agent"]
    agent = next(filter(lambda x: x.name == cur_agent, agents), None)
    if not agent:
        return state

    if agent.name == cur_agent:
        if cur_agent == "Boss Agent" and len(agent.messages) == 0:
            agent.messages.append(AgentMessage(
                text=f"Alright, our task is '{agent.task}'. Get to work!",
                public=True
            ))
        else:
            while True:
                messages = compile_message(agents, agent)
                messages.insert(0, ("system", agent.system_prompt))

                llm = controller.llm
                llm_lc = llm.get_langchain_model().bind_tools(agent.tools, agent.functions)
                ai_msg = llm_lc.invoke(messages)

                calls = PydanticToolsParserLocal(tools=agent.tools).invoke(ai_msg)
                has_tool_call = False
                tool_res_list = []
                for call in calls:
                    has_tool_call = True
                    tool_state = {}
                    tool_res_list.append(call.execute(tool_state))
                    state.update(tool_state)

                    if state["confidence"] > 0:
                        return state

                # run until something called
                if len(agent.tools) > 0 and not has_tool_call:
                    continue

                agent.messages.append(AgentMessage(
                    text=ai_msg.content,
                    public=not has_tool_call
                ))

                if has_tool_call:
                    call_string = "\n".join([json.dumps(x) for x in tool_res_list])
                    agent.messages.append(AgentMessage(
                        text=call_string,
                        public=True,
                        from_tool=True
                    ))

                break

    return state


def get_next_agent(state: SubAgentState):
    if state["confidence"] > 0:
        return END
    return state["next_agent"]


def execute_boss_worker_chat(context_data: str, task: str, agents: List[Agent]) -> (str, float):
    workflow = StateGraph(SubAgentState)
    workflow.add_node("router", lambda x: _execute_router(x, agents))
    workflow.add_edge(START, "router")
    # add boss
    agents.insert(0,
        Agent(
            name="Boss Agent",
            description="",
            goal="",
            system_prompt=controller.format_str(prompt_boss,
                                                extra={"conclusion_schema": json.dumps(Conclusion.model_json_schema())}),
            tools=[BossAgentChoice],
            task=task
        )
    )

    agent_list = ", ".join([x.name for x in agents])

    # format prompts
    for agent in agents:
        if agent.system_prompt == "":
            agent.system_prompt = prompt_subagent

        extra = {
            "context_data": context_data,
            "task": task,
            "description": agent.description,
            "goal": agent.goal,
            "agent_list": agent_list
        }
        extra.update(create_header_functions(agent.functions))
        extra.update(create_documentation_functions(agent.functions))
        extra.update(create_documentation_json(agent.tools))

        agent.system_prompt = controller.format_str(agent.system_prompt, extra=extra)
        workflow.add_node(agent.name, lambda x: _execute_agent(x, agents))
        workflow.add_edge(agent.name, "router")

    workflow.add_conditional_edges("router", get_next_agent)
    graph = workflow.compile()

    state = SubAgentState(
        confidence=0,
        next_agent="initial"
    )
    state = graph.invoke(state, {"recursion_limit": 100})
    print(state)

