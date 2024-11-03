import logging
import time

from langgraph.constants import START, END
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field

from pm.architecture.state import AgentState
from pm.architecture.system_response import add_response_system_to_graph
from pm.architecture.system_thought import add_thought_system_to_graph
from pm.clustering.summarize import cluster_and_summarize, high_level_summarize
from pm.consts import COMPLEX_THRESHOLD, RECALC_SUMMARIES_MESSAGES, THOUGHT_VALIDNESS_MIN, RESPONSE_VALIDNESS_MIN, THOUGHT_SEP
from pm.controller import controller
from pm.database.db_helper import fetch_messages, fetch_messages_no_summary, insert_object
from pm.database.db_model import Message, MessageInterlocus
from pm.database.transactions import start_transaction, rollback_transaction, commit_transaction
from pm.utils.token_utils import quick_estimate_tokens

logger = logging.getLogger(__name__)

# high level
def agent_start_transaction(state: AgentState):
    rollback_transaction()
    start_transaction()

    input = state["input"]
    msg_user = Message(
        conversation_id=state["conversation_id"],
        role='user',
        public=True,
        text=input,
        embedding=controller.embedder.get_embedding_scalar_float_list(input),
        tokens=quick_estimate_tokens(input),
        interlocus=MessageInterlocus.MessageResponse
    )
    insert_object(msg_user)

    return state

def agent_commit_transaction(state: AgentState):
    if state["thought"] != "":
        thought = state["thought"]
        thoguht_ai = Message(
            conversation_id=state["conversation_id"],
            role='assistant',
            text=thought,
            public=True,
            embedding=controller.embedder.get_embedding_scalar_float_list(thought),
            tokens=quick_estimate_tokens(thought),
            interlocus=MessageInterlocus.MessageThought
        )
        insert_object(thoguht_ai)

    if state['output'] != "":
        output = state['output']
        msg_ai = Message(
            conversation_id=state["conversation_id"],
            role='assistant',
            text=output,
            public=True,
            embedding=controller.embedder.get_embedding_scalar_float_list(output),
            tokens=quick_estimate_tokens(output),
            interlocus=MessageInterlocus.MessageResponse
        )
        insert_object(msg_ai)

    commit_transaction()
    return state

def agent_tasks_create(state: AgentState):
    state["task"] = []
    if state["input"].strip() != "":
        state["task"].append("task_converse")
    if state["input"].strip() == "":
        state["task"].append("task_think")
    if len(fetch_messages_no_summary(state["conversation_id"])) > RECALC_SUMMARIES_MESSAGES:
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

def agent_task_converse_start(state: AgentState):
    state["task"].remove("task_converse")
    return state

def agent_task_converse_end(state: AgentState):
    return state

def agent_task_think_start(state: AgentState):
    return state

def agent_task_think_end(state: AgentState):
    return state

def task_delegate_func(state: AgentState):
    if "task_summarize" in state["task"]:
        return "agent_task_summarize"
    elif "task_converse" in state["task"]:
        return "agent_task_converse_start"
    elif "task_think" in state["task"]:
        return "agent_task_think_start"
    else:
        return "agent_commit_transaction"


def get_graph():
    # nodes
    workflow = StateGraph(AgentState)
    workflow.add_node("agent_start_transaction", agent_start_transaction)
    workflow.add_node("agent_tasks_create", agent_tasks_create)
    workflow.add_node("agent_tasks_delegate", agent_tasks_delegate)

    workflow.add_node("agent_task_summarize", agent_task_summarize)

    workflow.add_node("agent_task_converse_start", agent_task_converse_start)
    workflow.add_node("agent_task_converse_end", agent_task_converse_end)

    workflow.add_node("agent_task_think_start", agent_task_think_start)
    workflow.add_node("agent_task_think_end", agent_task_think_end)

    workflow.add_node("agent_commit_transaction", agent_commit_transaction)

    # edges
    workflow.add_edge(START, "agent_start_transaction")
    workflow.add_edge("agent_start_transaction", "agent_tasks_create")
    workflow.add_edge("agent_tasks_create", "agent_tasks_delegate")
    workflow.add_conditional_edges("agent_tasks_delegate", task_delegate_func)
    workflow.add_edge("agent_task_summarize", "agent_tasks_delegate")
    workflow.add_edge("agent_task_converse_end", "agent_tasks_delegate")
    workflow.add_edge("agent_commit_transaction", END)

    # response system
    add_response_system_to_graph(workflow, "agent_task_converse_start", "agent_task_converse_end")
    add_thought_system_to_graph(workflow, "agent_task_think_start", "agent_task_think_end")
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