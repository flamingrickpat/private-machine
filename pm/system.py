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
from pydantic import BaseModel

from jinja2 import Template

from pm.config.config import read_config_file
from pm.controller import controller
from pm.database.db_helper import fetch_messages, search_documentation, search_documentation_text, fetch_documentation
from pm.database.db_model import Message
from pm.llm.llm import chat_complete
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


def make_completion(state: AgentState = None) -> AgentState:
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



# level 0
#workflow.add_edge(START, "start_transaction")
#workflow.add_edge("start_transaction", "tasks_create")
#workflow.add_edge("tasks_create", "tasks_delegate")
#workflow.add_conditional_edges("tasks_delegate", lambda state: state["next_agent"])
#workflow.add_edge("summarize", "tasks_delegate")
#workflow.add_edge("converse", "tasks_delegate")
#workflow.add_edge("commit_transaction", END)
