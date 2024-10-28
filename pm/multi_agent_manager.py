import random
from typing import List, Type

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
from pm.utils.misc_utils import get_key_from_value

logger = logging.getLogger(__name__)


class SubAgentState(TypedDict):
    next_agent: str


class Agent(BaseModel):
    name: str
    system_prompt: str = Field(default_factory=str)
    description: str
    goal: str
    tools: List[Type[BaseModel]] = Field(default_factory=list)


class Conclusion(BaseModel):
    """Final conclusion format."""
    confidence: float
    conclusion: str


def _execute_router(state: SubAgentState, agents: List[Agent]):
    state["next_agent"] = random.choice(agents).name
    return state


def _execute_agent(state: SubAgentState, agents: List[Agent]):
    cur_agent = state["next_agent"]
    for agent in agents:
        if agent.name == cur_agent:
            print(state["next_agent"])
    return state


def execute_boss_worker_chat(context_data: str, task: str, agents: List[Agent]) -> (str, float):
    def get_next_agent(state: SubAgentState):
        return state["next_agent"]


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
            tools=[Conclusion]
        )
    )

    # format prompts
    for agent in agents:
        if agent.system_prompt == "":
            tool_schemas = "\n".join([json.dumps(t.model_json_schema()) for t in agent.tools])
            agent.system_prompt = controller.format_str(prompt_subagent, extra={
                "context_data": context_data,
                "task": task,
                "description": agent.description,
                "goal": agent.goal,
                "tool_schemas": tool_schemas
            })



        workflow.add_node(agent.name, lambda x: _execute_agent(x, agents))
        workflow.add_edge(agent.name, "router")

    workflow.add_conditional_edges("router", get_next_agent)
    graph = workflow.compile()

    state = SubAgentState(
        next_agent=agents[0].name
    )
    state = graph.invoke(state, {"recursion_limit": 100})
    print(state["next_agent"])

