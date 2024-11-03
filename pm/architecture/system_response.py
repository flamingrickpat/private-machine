import json
import logging
from typing import TypedDict, Dict, List

from langgraph.graph import StateGraph

from pm.agents.assistant_story_check import assistant_story_check
from pm.agents.completion_mode import determine_completion_mode, CompletionModeResponseMode
from pm.agents.determine_complexity import determine_complexity
from pm.agents.quality_feedback import generate_instructions_from_feedback
from pm.agents.rewrite_as_thought import rewrite_as_thought
from pm.agents.tool_selection import determine_tools
from pm.agents.validate_response import validate_response
from pm.agents.validate_thought import validate_thought
from pm.agents_dynamic.schema_subconscious import get_plan_from_subconscious_agents
from pm.architecture.state import AgentState
from pm.clustering.summarize import cluster_and_summarize, high_level_summarize
from pm.consts import COMPLEX_THRESHOLD, RECALC_SUMMARIES_MESSAGES, THOUGHT_VALIDNESS_MIN, RESPONSE_VALIDNESS_MIN, MAX_REGENERATE_COUNT
from pm.controller import controller
from pm.database.db_helper import fetch_messages, fetch_messages_no_summary, rank_table, fetch_relations, fetch_messages_as_string
from pm.database.db_model import Message, MessageSummary
from pm.database.load_messages import build_prompt
from pm.llm.base_llm import LlmPreset, CommonCompSettings
from pm.prompts.prompt_main_agent import build_sys_prompt_conscious_assistant
from pm.prompts.prompt_main_agent_story import build_sys_prompt_conscious_story
from pm.utils.string_utils import get_last_n_messages_or_words_from_string, get_text_after_keyword

logger = logging.getLogger(__name__)


def agent_task_converse(state: AgentState):
    state["complexity"] = 0
    state["available_tools"] = []
    state["completion_mode"] = "assistant"
    return state


def agent_assistant_story_check(state: AgentState) -> AgentState:
    state["story_mode"] = assistant_story_check(state['current_user_assistant_chat'])
    return state


def agent_preprocess_input(state: AgentState):
    msgs = fetch_messages(state["conversation_id"])

    state["complexity"] = determine_complexity(msgs)
    state["available_tools"] = determine_tools(msgs)
    state["completion_mode"] = determine_completion_mode(msgs).value
    state["thought"] = ""

    return state

def agent_generate_thought(state: AgentState):
    full_text = fetch_messages_as_string(state["conversation_id"])
    query = get_last_n_messages_or_words_from_string(full_text)

    plan = get_plan_from_subconscious_agents(query, f"What should the AI companion {controller.config.companion_name} say to mimic human cognition and agency in every way?")
    state["plan"] = plan
    while True:
        thought = rewrite_as_thought(f"{query}\n{plan}", max_sentences_in=4, max_sentences_out=4)
        validness = validate_thought(thought)
        if validness > THOUGHT_VALIDNESS_MIN:
            state["thought"] = thought
            break

    return state

def agent_completion(state: AgentState):
    return state

def agent_completion_assistant(state: AgentState):
    full_text = fetch_messages_as_string(state["conversation_id"])
    query = get_last_n_messages_or_words_from_string(full_text)
    msgs = rank_table(state["conversation_id"], query, Message)
    sums = rank_table(state["conversation_id"], query, MessageSummary)
    rels = fetch_relations("main")
    messages = build_prompt(False, msgs, sums, rels, full_token_allowance=4096)

    # add system prompt
    messages.insert(0, (
        "system",
        build_sys_prompt_conscious_assistant(state["complexity"] > COMPLEX_THRESHOLD, state["available_tools"])
    ))

    if state["thought"] != "":
        thought = state["thought"]
        prefix = controller.get_thought_string_assistant(thought) + "\n" + controller.get_response_string_assistant("")
    else:
        prefix = controller.get_response_string_assistant("")

    messages.append((
        "assistant",
        prefix
    ))

    sws = [f"Response:",
           f"Thought:"]
    feedback = []
    regen_count = 0
    regens = []
    while True:
        content = controller.completion_text(LlmPreset.Default, messages, comp_settings=CommonCompSettings(max_tokens=1024, stop_words=sws)).strip()
        content = get_text_after_keyword(prefix + content, controller.get_response_string_assistant(""))
        if content == "":
            continue
        validness, reason = validate_response(f"{query}\n{content}")
        if validness > RESPONSE_VALIDNESS_MIN:
            regens = [(validness, content)]
            break
        else:
            feedback.append(reason)
            aux_sys_prompt = generate_instructions_from_feedback(feedback)
            messages.append((
                "system",
                aux_sys_prompt
            ))
            regens.append((validness, content))
            if regen_count > MAX_REGENERATE_COUNT:
                break
        regen_count += 1

    # Select the content with the highest validness score from regens
    if regens:
        state["output"] = max(regens, key=lambda x: x[0])[1]
        state["status"] = 0

    return state

def agent_completion_story(state: AgentState):
    full_text = fetch_messages_as_string(state["conversation_id"])
    query = get_last_n_messages_or_words_from_string(full_text)
    msgs = rank_table(state["conversation_id"], query, Message)
    sums = rank_table(state["conversation_id"], query, MessageSummary)
    rels = fetch_relations("main")

    messages = build_prompt(True, msgs, sums, rels, full_token_allowance=4096)
    message_block = "\n".join([f"{controller.config.user_name if item[0] == 'user' else controller.config.companion_name}: {item[1]}" for item in messages])

    # add system prompt
    messages = [(
        "system",
        build_sys_prompt_conscious_story(state["complexity"] > COMPLEX_THRESHOLD)
    ), (
        "user",
        "Please write a story about endearing experience of an AI and their user."
    ), (
        "assistant",
        message_block
    )]

    # add latest message
    prefix = ""
    thought = ""
    if state["thought"] != "":
        plan = state["plan"]
        messages.append((
            "user",
            plan
        ))
        thought = state["thought"]
        prefix = controller.get_thought_string_story(thought)

    messages.append((
        "assistant",
        prefix
    ))

    sws = [f"{controller.config.user_name}:",
           f"{controller.config.user_name} thinks:",
           f"{controller.config.companion_name}:",
           f"{controller.config.companion_name} thinks:"]
    feedback = []
    regen_count = 0
    regens = []
    while True:
        content = controller.completion_text(LlmPreset.Default, messages, comp_settings=CommonCompSettings(max_tokens=1024, stop_words=sws)).replace("Response:", "").strip()
        if content == "":
            continue

        validness, reason = validate_response(f"{query}\n{content}")
        if validness > RESPONSE_VALIDNESS_MIN:
            regens = [(validness, content)]
            break
        else:
            feedback.append(reason)
            aux_sys_prompt = generate_instructions_from_feedback(feedback)
            messages.append((
                "system",
                aux_sys_prompt
            ))
            regens.append((validness, content))
            if regen_count > MAX_REGENERATE_COUNT:
                break
        regen_count += 1

    # Select the content with the highest validness score from regens
    if regens:
        state["output"] = max(regens, key=lambda x: x[0])[1]
        state["status"] = 0

    return state


def complexity_switch_func(state: AgentState):
    if state["complexity"] > COMPLEX_THRESHOLD:
        return "agent_generate_thought"
    else:
        return "agent_completion"


def completion_select_func(state: AgentState):
    if state["completion_mode"] == CompletionModeResponseMode.Assistant.value:
        return "agent_completion_assistant"
    else:
        return "agent_completion_story"


def add_response_system_to_graph(workflow: StateGraph, start_node: str, end_node: str):
    workflow.add_node("agent_preprocess_input", agent_preprocess_input)
    workflow.add_node("agent_completion_assistant", agent_completion_assistant)
    workflow.add_node("agent_completion_story", agent_completion_story)
    workflow.add_node("agent_generate_thought", agent_generate_thought)
    workflow.add_node("agent_completion", agent_completion)
    workflow.add_edge(start_node, "agent_preprocess_input")
    workflow.add_conditional_edges("agent_preprocess_input", complexity_switch_func)
    workflow.add_edge("agent_generate_thought", "agent_completion")
    workflow.add_conditional_edges("agent_completion", completion_select_func)
    workflow.add_edge("agent_completion_story", end_node)
    workflow.add_edge("agent_completion_assistant", end_node)