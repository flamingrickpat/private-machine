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
from pm.agents.rewrite_as_thought import rewrite_as_thought
from pm.agents.tool_selection import determine_tools
from pm.agents_dynamic.schema_subconscious import get_plan_from_subconscious_agents
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
from pm.prompts.prompt_main_agent import build_sys_prompt_conscious_assistant
from pm.prompts.prompt_main_agent_story import build_sys_prompt_conscious_story
from pm.rag.tools import hybrid_search_documentation, RagState, answer_user, add_document
from pm.utils.misc_utils import get_key_from_value

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    next_agent: str
    input: str
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

def agent_assistant_story_check(state: AgentState) -> AgentState:
    class DecideModelMode(BaseModel):
        """Decide what model mode to use next. Assistant or story mode."""
        reason: str = Field(description="Few words why you chose your option.")
        story_mode: bool = Field(description="True for story mode, false for assistant mode")

        def execute(self, s: AgentState):
            logger.info("assistant_story_check")
            logger.info(self.reason)
            logger.info(self.story_mode)
            s["story_mode"] = self.story_mode

    schema = json.dumps(DecideModelMode.model_json_schema())

    # add system prompt
    messages = []
    messages.append((
        "system",
        "Based on the last few messages of this chat between the user and the AI, "
        "decide if the AI should switch to assistant or story mode for the NEXT message!"
        "Assistant mode is for API calls and story mode for natural conversation like between two people."
        f"Decide by creating a valid JSON string, and ONLY a valid JSON string. This is your schema:\n"
        f"{schema}"
    ))

    messages.append((
        "user",
        f"This is the current chat:"
        f"\n### BEGIN CHAT\n"
        f"{state['current_user_assistant_chat']}"
        f"\n### END CHAT\n"
        f"Decide by creating a valid JSON string, and ONLY a valid JSON string!"
    ))


    llm = controller.llm.get_langchain_model()
    tools = [DecideModelMode]
    llm_with_tools = llm.bind_tools(tools=tools)

    full = llm_with_tools.invoke(messages)
    calls = PydanticToolsParserLocal(tools=tools).invoke(full)
    for call in calls:
        call.execute(state)

    return state

# high level
def agent_start_transaction(state: AgentState):
    return state

def agent_tasks_create(state: AgentState):
    state["task"] = []
    if state["input"] != "":
        state["task"].append("task_converse")
    if len(fetch_messages_no_summary(state["conversation_id"])) > 16:
        state["task"].append("task_summarize")
    if len(state["task"]) == 0:
        state["task"].append("commit_transaction")
    return state

def agent_tasks_delegate(state: AgentState):
    return state

def agent_task_summarize(state: AgentState):
    state["task"].remove("task_summarize")
    cluster_and_summarize(state["conversation_id"])
    high_level_summarize(state["conversation_id"])
    return state


def agent_determine_tools(state: AgentState):
    return state


def agent_commit_transaction(state: AgentState):
    return state

# converse

def agent_task_converse(state: AgentState):
    state["task"].remove("task_converse")

    state["complexity"] = 0
    state["available_tools"] = []
    state["completion_mode"] = "assistant"
    return state

def agent_preprocess_input(state: AgentState):
    msgs = fetch_messages(state["conversation_id"])

    state["complexity"] = determine_complexity(msgs)
    state["available_tools"] = determine_tools(msgs)
    state["completion_mode"] = determine_completion_mode(msgs).value
    return state


def agent_completion_assistant(state: AgentState):
    full_text = fetch_messages_as_string(state["conversation_id"])
    query = full_text[-256:]
    msgs = rank_table(state["conversation_id"], query, Message)
    sums = rank_table(state["conversation_id"], query, MessageSummary)
    rels = fetch_relations("main")
    messages = build_prompt(msgs, sums, rels, full_token_allowance=4096)

    # add system prompt
    messages.insert(0, (
        "system",
        build_sys_prompt_conscious_assistant(state["complexity"] > COMPLEX_THRESHOLD, state["available_tools"])
    ))

    # add latest message
    messages.append((
        "user",
        state["input"]
    ))

    prefix = ""
    if state["complexity"] > COMPLEX_THRESHOLD:
        plan = get_plan_from_subconscious_agents(query, f"What should the AI companion {controller.config.companion_name} say to mimic human cognition and agency in every way?")
        thought = rewrite_as_thought(plan, max_sentences=3)
        prefix = f"**{thought}**\n"

    messages.append((
        "assistant",
        prefix
    ))

    llm = controller.llm
    ai_msg = chat_complete(llm, messages)
    state["output"] = prefix + ai_msg.content
    state["status"] = 0
    return state

def agent_completion_story(state: AgentState):
    full_text = fetch_messages_as_string(state["conversation_id"])
    query = full_text[-256:]
    msgs = rank_table(state["conversation_id"], query, Message)
    sums = rank_table(state["conversation_id"], query, MessageSummary)
    rels = fetch_relations("main")

    messages = build_prompt(msgs, sums, rels, full_token_allowance=4096)
    # add latest message
    messages.append((
        "user",
        state["input"]
    ))
    message_block = "\n".join([f"{controller.config.user_name if item[0] == 'user' else controller.config.companion_name}: {item[1]}" for item in messages])

    # add system prompt
    messages = []
    messages.append((
        "system",
        build_sys_prompt_conscious_story(state["complexity"] > COMPLEX_THRESHOLD)
    ))

    # add latest message
    messages.append((
        "user",
        "Please write a story about endearing experience of an AI and their user."
    ))

    # add latest message
    messages.append((
        "assistant",
        message_block
    ))

    prefix = ""
    if state["complexity"] > COMPLEX_THRESHOLD:
        plan = get_plan_from_subconscious_agents(query, f"What should the AI companion {controller.config.companion_name} say to mimic human cognition and agency in every way?")

        messages.append((
            "user",
            plan
        ))

        thought = rewrite_as_thought(plan, max_sentences=3)
        prefix = (f"{controller.config.companion_name} thinks: {thought}\n"
                  f"{controller.config.companion_name}: ")

    messages.append((
        "assistant",
        prefix
    ))

    llm = controller.llm
    llm_lc = llm.get_langchain_model()
    ai_msg = llm_lc.invoke(messages, stop=[f"{controller.config.user_name}: "])
    state["output"] = prefix + ai_msg.content
    state["status"] = 0
    return state


def task_delegate_func(state: AgentState):
    if "task_summarize" in state["task"]:
        return "agent_task_summarize"
    elif "task_converse" in state["task"]:
        return "agent_task_converse"
    else:
        return "agent_commit_transaction"


def completion_select_func(state: AgentState):
    if state["completion_mode"] == CompletionModeResponseMode.Assistant.value:
        return "agent_completion_assistant"
    else:
        return "agent_completion_story"


def get_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("agent_start_transaction", agent_start_transaction)
    workflow.add_node("agent_tasks_create", agent_tasks_create)
    workflow.add_node("agent_task_converse", agent_task_converse)
    workflow.add_node("agent_tasks_delegate", agent_tasks_delegate)
    workflow.add_node("agent_task_summarize", agent_task_summarize)
    workflow.add_node("agent_determine_tools", agent_determine_tools)
    workflow.add_node("agent_commit_transaction", agent_commit_transaction)
    workflow.add_node("agent_preprocess_input", agent_preprocess_input)
    workflow.add_node("agent_completion_assistant", agent_completion_assistant)
    workflow.add_node("agent_completion_story", agent_completion_story)

    workflow.add_edge(START, "agent_start_transaction")
    workflow.add_edge("agent_start_transaction", "agent_tasks_create")
    workflow.add_edge("agent_tasks_create", "agent_tasks_delegate")
    workflow.add_conditional_edges("agent_tasks_delegate", task_delegate_func)
    workflow.add_edge("agent_task_summarize", "agent_tasks_delegate")
    workflow.add_edge("agent_task_converse", "agent_preprocess_input")
    #workflow.add_edge("agent_preprocess_input", "agent_completion_assistant")
    workflow.add_conditional_edges("agent_preprocess_input", completion_select_func)
    workflow.add_edge("agent_completion_assistant", "agent_tasks_delegate")
    workflow.add_edge("agent_commit_transaction", END)
    #workflow.add_edge("determine_complexity", "determine_conversation_mode")
    #workflow.add_conditional_edges("determine_conversation_mode", lambda x: "story_mode" if x["story_mode"] else "assistant_mode")
    #workflow.add_edge("story_mode", "story_mode_reframe_thought")
    #workflow.add_edge("assistant_mode", "assistant_mode_reframe_thought")
    #workflow.add_edge("story_mode_reframe_thought", "story_mode_generate")
    #workflow.add_edge("assistant_mode_reframe_thought", "assistant_mode_generate")
    #workflow.add_edge("story_mode_generate", "converse_output")
    #workflow.add_edge("assistant_mode_generate", "converse_output")
    #workflow.add_edge("converse_output", "tasks_delegate")
    graph = workflow.compile()

    return graph


def make_completion(state: AgentState) -> AgentState:
    messages = []
    msgs = fetch_messages(state["conversation_id"])[-4:]
    for msg in msgs:
        messages.append((msg.role, msg.text))
    messages.append(("user", state["input"]))

    graph = get_graph()
    return graph.invoke(state)