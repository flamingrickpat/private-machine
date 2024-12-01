import json
import logging
import uuid
from typing import TypedDict, Dict, List

from langgraph.graph import StateGraph

from pm.agents.generate_thought_aspects import generate_personality_aspects
from pm.agents.generate_thoughts import generate_thoughts
from pm.agents.validate_thought import validate_thought
from pm.agents_dynamic.schema_thought_contemplation import thought_contemplation_dialog
from pm.architecture.state import AgentState
from pm.consts import THOUGHT_VALIDNESS_MIN
from pm.controller import controller
from pm.database.db_helper import fetch_messages, fetch_messages_no_summary, rank_table, fetch_relations, fetch_messages_as_string, get_facts_str, insert_object
from pm.database.db_model import Message, MessageSummary, MessageInterlocus
from pm.utils.string_utils import get_last_n_messages_or_words_from_string
from pm.utils.token_utils import quick_estimate_tokens

logger = logging.getLogger(__name__)

def agent_generate_int_thoughts(state: AgentState):
    sysmsg = "The user hasn't responded yet, entering autonomous thinking mode..."
    msg_init = Message(
        conversation_id=state.conversation_id,
        role='system',
        text=sysmsg,
        public=True,
        embedding=controller.embedder.get_embedding_scalar_float_list(sysmsg),
        tokens=quick_estimate_tokens(sysmsg),
        interlocus=MessageInterlocus.AutoThoughtSystemInst
    )
    insert_object(msg_init)

    user_memory = "No memory yet!"
    full_text = fetch_messages_as_string(state.conversation_id)
    conv_context = get_last_n_messages_or_words_from_string(full_text, n_messages=16)
    ai_memory = "\n".join(get_facts_str(state.conversation_id, conv_context))

    while True:
        thought = generate_thoughts(f"Generate a thought for {controller.config.companion_name}!", controller.config.character_card_story, conv_context, ai_memory, user_memory)
        validness = validate_thought(thought)
        if validness > THOUGHT_VALIDNESS_MIN:
            state.thought = thought
            break

    msg_thought = Message(
        conversation_id=state.conversation_id,
        role='assistant',
        text=thought,
        public=False,
        embedding=controller.embedder.get_embedding_scalar_float_list(thought),
        tokens=quick_estimate_tokens(thought),
        interlocus=MessageInterlocus.MessageThought
    )
    insert_object(msg_thought)

    logger.info(f"Thought: {thought}")
    return state


def agent_generate_aspects(state: AgentState):
    full_text = fetch_messages_as_string(state.conversation_id)
    conv_context = get_last_n_messages_or_words_from_string(full_text)

    aspects = generate_personality_aspects(controller.config.character_card_story, state.thought)
    res = thought_contemplation_dialog(conv_context, f"Reach a conclusion on this thought: {state['thought']}", aspects)

    thought = res.as_internal_thought
    msg_thought = Message(
        conversation_id=state.conversation_id,
        role='assistant',
        text=thought,
        public=False,
        embedding=controller.embedder.get_embedding_scalar_float_list(thought),
        tokens=quick_estimate_tokens(thought),
        interlocus=MessageInterlocus.MessageThought
    )
    insert_object(msg_thought)

    return state


def add_thought_system_to_graph(workflow: StateGraph, start_node: str, end_node: str):
    workflow.add_node("agent_generate_int_thoughts", agent_generate_int_thoughts)
    workflow.add_node("agent_generate_aspects", agent_generate_aspects)

    workflow.add_edge(start_node, "agent_generate_int_thoughts")
    workflow.add_edge("agent_generate_int_thoughts", "agent_generate_aspects")
    workflow.add_edge("agent_generate_aspects", end_node)