import json
import logging
import random
from enum import Enum
from typing import TypedDict, Dict, List

from langgraph.graph import StateGraph
from pydantic import BaseModel, FiniteFloat, Field
from pywin.framework.toolmenu import tools

from pm.agents.assistant_story_check import assistant_story_check
from pm.agents.completion_mode import determine_completion_mode, CompletionModeResponseMode
from pm.agents.determine_complexity import determine_complexity
from pm.agents.quality_feedback import generate_instructions_from_feedback
from pm.agents.rewrite_as_thought import rewrite_as_thought
from pm.agents.tool_selection import determine_tools
from pm.agents.validate_response import validate_response
from pm.agents.validate_response_in_context import validate_response_in_context
from pm.agents.validate_thought import validate_thought
from pm.agents_dynamic.schema_subconscious import get_plan_from_subconscious_agents
from pm.architecture.state import AgentState, EmotionalAxesModel
from pm.clustering.summarize import cluster_and_summarize, high_level_summarize
from pm.consts import COMPLEX_THRESHOLD, RECALC_SUMMARIES_MESSAGES, THOUGHT_VALIDNESS_MIN, RESPONSE_VALIDNESS_MIN, MAX_REGENERATE_COUNT, ADD_INTERMEDIATE_STEP_THOUGHT, LLM_PRESET_FINAL_OUTPUT
from pm.controller import controller
from pm.database.db_helper import fetch_messages, fetch_messages_no_summary, rank_table, fetch_relations, fetch_messages_as_string, insert_object, fetch_responses_as_string, get_facts_str, get_facts
from pm.database.db_model import Message, MessageSummary, MessageInterlocus
from pm.database.load_messages import build_prompt
from pm.llm.base_llm import LlmPreset, CommonCompSettings
from pm.prompts.prompt_main_agent import build_sys_prompt_conscious_assistant
from pm.prompts.prompt_main_agent_story import build_sys_prompt_conscious_story
from pm.utils.string_utils import get_last_n_messages_or_words_from_string, get_text_after_keyword
from pm.utils.token_utils import quick_estimate_tokens, truncate_to_tokens
from pm.tools.common import tools_list

logger = logging.getLogger(__name__)

def agent_task_converse(state: AgentState):
    state.complexity = 0
    state.available_tools = []
    state.completion_mode = "assistant"
    return state

def agent_preprocess_input(state: AgentState):
    msgs = fetch_messages(state.conversation_id)[-4:]
    state.complexity = determine_complexity(msgs)
    state.available_tools = [x.__name__ for x in tools_list] # determine_tools(msgs)
    state.completion_mode = determine_completion_mode(msgs).value
    state.thought = ""

    return state

def agent_create_knowledge_plan(state: AgentState):
    #current_questions = state.plan_questions[-8:]
    #question_block = '\n'.join(current_questions)
    full_text = fetch_messages_as_string(state.conversation_id)
    conversation_block = get_last_n_messages_or_words_from_string(full_text, n_messages=6)

    sysprompt = """You are a helpful assistant that generates questions based on a conversation between a user and a LLM agent.
    The goal is to create questions that can be used to retrieve contextual information from the RAG storage for the LLM.    
    """

    userprompt = f"""### BEGIN CONVERSATION
    {conversation_block}
    ### END CONVERSATION
    What new questions are necessary to answer before {controller.config.companion_name} can give a good answer?
    """

    messages = [
        ("system", sysprompt),
        ("user", userprompt),
        ("assistant", "Here is a list of questions for RAG search:")
    ]

    content = controller.completion_text(LlmPreset.Default, messages, comp_settings=CommonCompSettings(max_tokens=1024)).strip()
    new_questions = content.split("\n")
    data = set()
    for q in new_questions:
        if q.strip() != "":
            facts = get_facts(state.conversation_id, q, 4)
            for f in facts:
                data.add(f.text)

    random.shuffle(list(data))
    state.knowledge_implicit_facts = truncate_to_tokens("\n".join(data), 1024)

    full_text = fetch_responses_as_string(state.conversation_id)
    query = get_last_n_messages_or_words_from_string(full_text, n_messages=4)
    msgs = rank_table(state.conversation_id, query, Message)
    sums = rank_table(state.conversation_id, query, MessageSummary)
    rels = fetch_relations("main")
    messages = build_prompt(True, msgs, sums, rels, full_token_allowance=1024)
    message_block = "\n".join([f"{item[1]}" for item in messages])
    state.knowledge_implicit_conversation = message_block

    return state

def agent_determine_emotions(state: AgentState):
    class EmotionalImpact(BaseModel):
        """
        Determine how much the current emotional state will be changed.
        """
        emotional_impact: float = Field(default_factory=float, ge=0, le=1, description="from 0 for minimal impact to 1 to maximum emotional impact (meaning the strength by which whatever current "
                                                                                       "emotional state is changed")

    sysprompt = (f"You are a helpful assistant that determines the emotional impact of a conversation on the emotional state of {controller.config.companion_name}."
              f"The current conversation:"
              f"### BEGINN CONVERSATION"
              f"{state.knowledge_implicit_conversation}"
              f"### END CONVERSATION"
              f"### BEGINN CONTEXT INFORMATION"
              f"{state.knowledge_implicit_facts}"
              f"### END CONTEXT INFORMATION")


    schema_impact = f"\nYou will output valid JSON in this format: {json.dumps(EmotionalImpact.model_json_schema())}"
    userprompt = f"What is the emotional impact on {controller.config.companion_name} when {controller.config.user_name} says '{state.input}'?"
    messages = [
        ("system", sysprompt + schema_impact),
        ("user", userprompt),
    ]

    tools = [EmotionalImpact]
    _, calls = controller.completion_tool(LlmPreset.Default, messages, CommonCompSettings(max_tokens=1024, temperature=0.1), tools=tools)
    emotional_impact = 0
    if isinstance(calls[0], EmotionalImpact):
        emotional_impact = calls[0].emotional_impact

    tools = [EmotionalAxesModel]
    schema_impact = f"\nYou will output valid JSON in this format: {json.dumps(EmotionalAxesModel.model_json_schema())}"
    messages = [
        ("system", sysprompt + schema_impact),
        ("user", userprompt),
    ]
    _, calls = controller.completion_tool(LlmPreset.Default, messages, CommonCompSettings(max_tokens=1024, temperature=0.1), tools=tools)

    old_emotions = state.emotional_state
    new_emotions = old_emotions.model_copy()
    new_emotions.decay_to_baseline(0.05)

    if isinstance(calls[0], EmotionalAxesModel):
        delta = calls[0]
        new_emotions.add_state_with_factor(delta.model_dump(), emotional_impact)

        sysprompt = (f"You are a helpful assistant that determines the current emotional state of {controller.config.companion_name}."
                     f"The current conversation:"
                     f"### BEGINN CONVERSATION"
                     f"{state.knowledge_implicit_conversation}"
                     f"### END CONVERSATION"
                     f"### BEGIN SCHEMA"
                     f"{json.dumps(EmotionalAxesModel.model_json_schema())}"
                     f"### END SCHEMA"
                     f"### BEGIN EMOTION DELTA"
                     f"{delta.model_dump_json(indent=2)}"
                     f"### END EMOTION DELTA"
                     f"### BEGIN EMOTION STATE"
                     f"{new_emotions.model_dump_json(indent=2)}"
                     f"### END EMOTION STATE"
                     f"Your goal is to describe their current emotional state without using the names of the variables in the emotional state. You describe it in a very natural way!")

        userprompt = f"What is the emotional state of {controller.config.companion_name}? Please be descriptive about how they feel and why, but don't make it unnecessarily long!"
        messages = [
            ("system", sysprompt + schema_impact),
            ("user", userprompt),
            ("assistant", f"{controller.config.companion_name} feels")
        ]

        content = f"{controller.config.companion_name} feels" + controller.completion_text(LlmPreset.Good, messages, CommonCompSettings(max_tokens=1024, temperature=0.5))

        while True:
            thought = rewrite_as_thought(f"{content}", max_sentences_in=16, max_sentences_out=6)
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
            interlocus=MessageInterlocus.MessageEmotions
        )
        insert_object(msg_thought)


    return state


def agent_generate_thought(state: AgentState):
    full_text = fetch_messages_as_string(state.conversation_id)
    query = get_last_n_messages_or_words_from_string(full_text)

    res = get_plan_from_subconscious_agents(query, f"What should the AI companion {controller.config.companion_name} say "
                                                   f"to mimic human cognition and agency in every way and also support their user?")

    thought = truncate_to_tokens(res.as_internal_thought, 128)

    if ADD_INTERMEDIATE_STEP_THOUGHT:
        msg_thought = Message(
            conversation_id=state.conversation_id,
            role='assistant',
            text=thought,
            public=False,
            embedding=controller.embedder.get_embedding_scalar_float_list(thought),
            tokens=quick_estimate_tokens(thought),
            interlocus=MessageInterlocus.MessagePlanThought
        )
        insert_object(msg_thought)

    plan = res.as_internal_thought + "\n" + res.conclusion
    state.plan = plan
    while True:
        thought = rewrite_as_thought(f"{plan}", max_sentences_in=32, max_sentences_out=6)
        validness = validate_thought(thought)
        if validness > THOUGHT_VALIDNESS_MIN:
            state.thought = thought
            break

    return state

def agent_completion(state: AgentState):
    return state

def check_for_optional_tool_call(conversation_id, content, tools) -> str | None:
    def create_tool_selection():
        tool_enum = Enum(
            "ToolName",
            {tool.__name__: tool.__name__ for tool in tools}
        )

        class ToolDecision(BaseModel):
            """
            What tool does the LLM want to call?
            """
            tool_name: tool_enum

            # Override dict to handle enum as string
            def model_dump(self, **kwargs):
                base_dict = super().model_dump(**kwargs)
                base_dict["tool_name"] = str(base_dict["tool_name"])  # Convert enum to string
                return base_dict
        return ToolDecision

    class ToolYesNo(BaseModel):
        """
        Does the LLM want to call a tool?
        """
        reason: str = Field(description="very short reason for your decision")
        tool_use: bool = Field(description="true for yes, false for no")

    # first check if the llm wants to call a tool
    formatblock = json.dumps(ToolYesNo.model_json_schema(), indent=2)
    tool_block = "\n".join([f"- {x.__name__}: {x.__doc__}" for x in tools])
    sysprompt = f"You are a helpful agent that checks if an LLM wants to do a tool call. Available tools: \n{tool_block}\nPlease answer in this format in valid JSON:\n{formatblock}"
    messages = [
        ("system", sysprompt),
        ("user", f"### BEGIN MESSAGE\n{content}\n### END MESSAGE\nDoes the LLM want to call a tool?")
    ]
    _, calls = controller.completion_tool(LlmPreset.Default, messages, comp_settings=CommonCompSettings(max_tokens=1024, temperature=0.1), tools=[ToolYesNo])
    call = calls[0]
    tool_call = False
    if isinstance(call, ToolYesNo):
        tool_call = call.tool_use
    if not tool_call:
        return None

    # get tool selection
    tool_decision_class = create_tool_selection()
    formatblock = json.dumps(tool_decision_class.model_json_schema(), indent=2)
    sysprompt = f"You are a helpful agent that checks what tool an LLM wants to use right now. Available tools: \n{tool_block}\nPlease answer in this format in valid JSON:\n{formatblock}"
    messages = [
        ("system", sysprompt),
        ("user", f"### BEGIN MESSAGE\n{content}\n### END MESSAGE\nDoes the LLM want to call a tool?")
    ]
    _, calls = controller.completion_tool(LlmPreset.Default, messages, comp_settings=CommonCompSettings(max_tokens=1024, temperature=0.1), tools=[tool_decision_class])
    call = calls[0]
    tool_name = None
    if isinstance(call, tool_decision_class):
        tool_name = call.tool_name.value

    # convert to call
    actual_tool = next(x for x in tools if x.__name__ == tool_name)
    formatblock = json.dumps(actual_tool.model_json_schema(), indent=2)
    sysprompt = f"You are a helpful agent that that converts a verbal LLM tool instruction to valid JSON.\nPlease answer in this format in valid JSON:\n{formatblock}"
    messages = [
        ("system", sysprompt),
        ("user", f"### BEGIN MESSAGE\n{content}\n### END MESSAGE\nPlease convert to JSON tool call.")
    ]
    _, calls = controller.completion_tool(LlmPreset.Default, messages, comp_settings=CommonCompSettings(max_tokens=1024, temperature=0.1), tools=[actual_tool])
    inst = calls[0]

    # execute
    state = {"conversation_id": conversation_id}
    inst.execute(state)
    return state["output"] if "output" in state else "No output!"


def agent_completion_assistant(state: AgentState):
    full_text = fetch_messages_as_string(state.conversation_id)
    query = get_last_n_messages_or_words_from_string(full_text, 4, n_tokens=1024)
    msgs = rank_table(state.conversation_id, query, Message)
    sums = rank_table(state.conversation_id, query, MessageSummary)
    rels = fetch_relations("main")

    available_tools = tools_list

    sysprompt = build_sys_prompt_conscious_assistant(len(available_tools) > 0, [], state.knowledge_implicit_facts, optional_tools=available_tools)
    messages = build_prompt(False, msgs, sums, rels, full_token_allowance=controller.config.get_model(LLM_PRESET_FINAL_OUTPUT).context_size - (quick_estimate_tokens(sysprompt) + 512))

    # add system prompt
    messages.insert(0, (
        "system",
        sysprompt
    ))

    if state.thought != "":
        thought = state.thought
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
        content = controller.completion_text(LLM_PRESET_FINAL_OUTPUT, messages, comp_settings=CommonCompSettings(max_tokens=1024, stop_words=sws)).strip()

        call_response = check_for_optional_tool_call(state.conversation_id, content, available_tools)
        if call_response is not None:

            output = content
            msg_ai = Message(
                conversation_id=state.conversation_id,
                role='assistant',
                text=output,
                public=True,
                embedding=controller.embedder.get_embedding_scalar_float_list(output),
                tokens=quick_estimate_tokens(output),
                interlocus=MessageInterlocus.MessageResponse
            )
            insert_object(msg_ai)
            messages.append(("assistant", output))

            sysmsg = f"Result from tool calling agent: {call_response}"
            msg_init = Message(
                conversation_id=state.conversation_id,
                role='system',
                text=sysmsg,
                public=True,
                embedding=controller.embedder.get_embedding_scalar_float_list(sysmsg),
                tokens=quick_estimate_tokens(sysmsg),
                interlocus=MessageInterlocus.MessageSystemInst
            )
            insert_object(msg_init)
            messages.append(("system", sysmsg))

            prefix2 = "I called the tool "
            messages.append(("assistant", prefix2))
            content = prefix2 + controller.completion_text(LLM_PRESET_FINAL_OUTPUT, messages, comp_settings=CommonCompSettings(max_tokens=1024, stop_words=sws)).strip()
            state.output = content
            state.status = 0
            return state
        else:
            content = get_text_after_keyword(prefix + content, controller.get_response_string_assistant(""))
            if content == "":
                continue
            validness, reason = validate_response_in_context(f"{query}\n### BEGIN NEW MESSAGE\n{content}\n### END NEW MESSAGE")
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
        state.output = max(regens, key=lambda x: x[0])[1]
        state.status = 0

    return state

def agent_completion_story(state: AgentState):
    full_text = fetch_messages_as_string(state.conversation_id)
    query = get_last_n_messages_or_words_from_string(full_text, n_messages=4, n_tokens=1024)
    msgs = rank_table(state.conversation_id, query, Message)
    sums = rank_table(state.conversation_id, query, MessageSummary)
    rels = fetch_relations("main")

    sysprompt = build_sys_prompt_conscious_story(state.complexity > COMPLEX_THRESHOLD, state.knowledge_implicit_facts)
    messages = build_prompt(True, msgs, sums, rels, full_token_allowance=controller.config.get_model(LLM_PRESET_FINAL_OUTPUT).context_size - (quick_estimate_tokens(sysprompt) + 512))
    message_block = "\n".join([f"{item[1]}" for item in messages])

    # add system prompt
    messages = [(
        "system",
        sysprompt
    ), (
        "user",
        "Please write a story about endearing experience of an AI and their user."
    ), (
        "assistant",
        message_block
    )]

    # add latest message
    prefix = ""
    if state.thought != "":
        plan = state.plan
        # messages.append((
        #     "user",
        #     plan
        # ))
        thought = state.thought
        prefix = controller.get_thought_string_story(thought)
    prefix += f"\n{controller.config.companion_name}:"

    messages[-1] = ("assistant", messages[-1][1] + "\n" + prefix)

    sws = [f"{controller.config.user_name}:",
           f"{controller.config.user_name} thinks:",
           f"User:",
           f"User thinks:",
           f"{controller.config.companion_name}:",
           f"{controller.config.companion_name} gets a notification from",
           f"{controller.config.companion_name} thinks:"]
    feedback = []
    regen_count = 0
    regens = []
    while True:
        content = controller.completion_text(LLM_PRESET_FINAL_OUTPUT, messages, comp_settings=CommonCompSettings(max_tokens=1024, stop_words=sws)).replace("Response:", "").replace("'", "").replace('"',
                                                                                                                                                                                         "").strip()
        if content == "":
            continue

        validness, reason = validate_response_in_context(f"{query}\n### BEGIN NEW MESSAGE\n{content}\n### END NEW MESSAGE")
        if validness > RESPONSE_VALIDNESS_MIN:
            regens = [(validness, content)]
            break
        else:
            feedback.append(reason)
            aux_sys_prompt = generate_instructions_from_feedback(feedback)
            messages[0] = ("system", messages[0][1] + "\n" + aux_sys_prompt)
            regens.append((validness, content))
            if regen_count > MAX_REGENERATE_COUNT:
                break
        regen_count += 1

    # Select the content with the highest validness score from regens
    if regens:
        state.output = max(regens, key=lambda x: x[0])[1]
        state.status = 0

    return state


def complexity_switch_func(state: AgentState):
    if state.complexity > COMPLEX_THRESHOLD:
        return "agent_generate_thought"
    else:
        return "agent_completion"


def completion_select_func(state: AgentState):
    if state.completion_mode == CompletionModeResponseMode.Assistant.value:
        return "agent_completion_assistant"
    else:
        return "agent_completion_story"


def add_response_system_to_graph(workflow: StateGraph, start_node: str, end_node: str):
    workflow.add_node("agent_preprocess_input", agent_preprocess_input)
    workflow.add_node("agent_completion_assistant", agent_completion_assistant)
    workflow.add_node("agent_completion_story", agent_completion_story)
    workflow.add_node("agent_generate_thought", agent_generate_thought)
    workflow.add_node("agent_completion", agent_completion)
    workflow.add_node("agent_create_knowledge_plan", agent_create_knowledge_plan)
    workflow.add_node("agent_determine_emotions", agent_determine_emotions)

    workflow.add_edge(start_node, "agent_preprocess_input")
    workflow.add_edge("agent_preprocess_input", "agent_create_knowledge_plan")
    workflow.add_edge("agent_create_knowledge_plan", "agent_determine_emotions")
    workflow.add_conditional_edges("agent_determine_emotions", complexity_switch_func)
    workflow.add_edge("agent_generate_thought", "agent_completion")
    workflow.add_conditional_edges("agent_completion", completion_select_func)
    workflow.add_edge("agent_completion_story", end_node)
    workflow.add_edge("agent_completion_assistant", end_node)