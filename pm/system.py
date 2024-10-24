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

from pm.config.config import read_config_file
from pm.controller import controller
from pm.database.db_helper import fetch_messages, search_documentation, search_documentation_text, fetch_documentation
from pm.database.db_model import Message
from pm.llm.llm import chat_complete
from pm.llm.tools_parser_local import PydanticToolsParserLocal
from pm.rag.tools import hybrid_search_documentation, RagState, answer_user, add_document
from pm.utils.misc_utils import get_key_from_value

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    next_agent: str
    output: str
    title: str
    conversation_id: str
    messages: List[Dict[str, str]]
    current_user_assistant_chat: str
    current_knowledge: str
    story_mode: bool


def find_documents_agent(state: AgentState):
    # add system prompt
    messages = []
    messages.append((
            "system",
            "find information"
        ))

    messages.append((
            "user",
            f"{state['current_user_assistant_chat']}"
        ))

    llm = controller.llm.get_langchain_model()
    tools = [hybrid_search_documentation, add_document, answer_user]
    llm_with_tools = llm.bind_tools(tools=tools, tool_choice="required")

    rag_state = RagState(
        finished=False,
        guid_id_map={},
        status=False,
        docs_temp=[],
        docs_final=[],
        current_id=1
    )

    while True:
        logger.info(f"messages: " + str(messages))
        full = llm_with_tools.invoke(messages)

        messages.append((
            "assistant",
            "<tool_call_placeholder>" if len(full.content) == 0 else full.content
        ))

        buffer = []
        calls = PydanticToolsParser(tools=tools).invoke(full)
        for call in calls:
            tmp = call.execute(rag_state)
            buffer.append(tmp)

        if len(buffer) == 0:
            messages.append((
                "user",
                "must call function"
            ))
        else:
            messages.append((
                "user",
                "\n".join(buffer)
            ))

        if rag_state["finished"]:
            break

    final_guids = []
    for id in rag_state["docs_final"]:
        guid = get_key_from_value(rag_state["guid_id_map"], id)
        final_guids.append(guid)

    final_docs = fetch_documentation(final_guids)
    tmp = []
    for doc in final_docs:
        tmp.append(doc.text)

    state["current_knowledge"] = "\n".join(tmp)
    logger.info(f"current_knowledge: " + state["current_knowledge"])


def main_agent(state: AgentState = None) -> AgentState:
    messages = []
    msgs = fetch_messages(state["conversation_id"])
    for msg in msgs:
        messages.append((msg.role, msg.text))
    messages.append(("user", state["input"]))

    full_text = "\n".join(f"{tp[0]}: {tp[1]}" for tp in messages)

    # add system prompt
    messages.insert(0, (
            "system",
            "You are a nice and helpful assistant."
        ))

    llm = controller.llm
    ai_msg = chat_complete(llm, messages)
    state["output"] = ai_msg.content
    state["status"] = 0
    return state

def assistant_story_check(state: AgentState) -> AgentState:
    class DecideModelMode(BaseModel):
        """Decide what model mode to use next. Assistant or story mode."""
        reason: str = Field(description="Few words you chose your option.")
        story_mode: bool = Field(description="true for story mode, false for assistant mode")

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
        "Based on the last few messages of this chat between the user and the AI, decide if the AI should switch to assistant or story mode."
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


def get_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("main_agent", main_agent)
    workflow.add_node("assistant_story_check", assistant_story_check)


    workflow.add_edge(START, "assistant_story_check")
    workflow.add_edge("assistant_story_check", "main_agent")
    workflow.add_edge("main_agent", END)

    graph = workflow.compile()
    return graph


def make_completion(state: AgentState) -> AgentState:
    messages = []
    msgs = fetch_messages(state["conversation_id"])[-4:]
    for msg in msgs:
        messages.append((msg.role, msg.text))
    messages.append(("user", state["input"]))
    full_text = "\n".join(f"{tp[0]}: {tp[1]}" for tp in messages)
    state["current_user_assistant_chat"] = full_text

    graph = get_graph()
    graph.invoke(state)
    return state

# level 0
#workflow.add_edge(START, "start_transaction")
#workflow.add_edge("start_transaction", "tasks_create")
#workflow.add_edge("tasks_create", "tasks_delegate")
#workflow.add_conditional_edges("tasks_delegate", lambda state: state["next_agent"])
#workflow.add_edge("summarize", "tasks_delegate")
#workflow.add_edge("converse", "tasks_delegate")
#workflow.add_edge("commit_transaction", END)
