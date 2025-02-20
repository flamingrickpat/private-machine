from pm.common_prompts.determine_tool import determine_tools
from pm.common_prompts.get_tool_call import get_tool_call
from pm.reply_consideration import generate_reply_consideration
from pm.subsystems.subsystem_base import CogEvent
from pm.system_utils import get_message_to_user, get_internal_contemplation, ActionType, determine_action_type, add_cognitive_event, get_prompt, completion_story_mode, \
    get_recent_messages_block
import time
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Optional, Any, List, Tuple
from pm.controller import controller, log_conversation
from pm.embedding.embedding import get_embedding
from pm.embedding.token import get_token
from pm.database.tables import Event, EventCluster, Cluster, ClusterType, PromptItem, Fact, InterlocusType, AgentMessages, CauseEffectDbEntry
from pm.character import sysprompt, database_uri, cluster_split, companion_name, timestamp_format, sysprompt_addendum


def generate_action_selection(allowed_actions: List[ActionType]) -> Tuple[ActionType, CogEvent]:
    prompt = get_prompt(story_mode=True)
    action_selection = determine_action_type(prompt, allowed_actions)
    content = None
    action_type = ActionType[action_selection.action_type.value]
    if action_type == ActionType.InitiateUserConversation:
        content = f"I will message my user because of the following reasons: {action_selection.reason}"
    elif action_type == ActionType.InitiateIdleMode:
        content = f"I will stay idle until the next heartbeat because of the following reasons: {action_selection.reason}"
    elif action_type == ActionType.InitiateInternalContemplation:
        content = f"I will contemplate for the time being because of the following reasons: {action_selection.reason}"
    elif action_type == ActionType.Reply:
        content = f"I will now reply to the user."
    elif action_type == ActionType.Ignore:
        content = f"I will ignore this input because: {action_selection.reason}"
    elif action_type == ActionType.ToolCall:
        content = f"I will use my AI powers to make an API call."

    event = Event(
        source=f"{companion_name}",
        content=content,
        embedding=controller.get_embedding(content),
        token=get_token(content),
        timestamp=datetime.now(),
        interlocus=InterlocusType.ActionDecision.value
    )
    return action_type, event


def action_init_user_conversation() -> str:
    prompt = get_prompt()
    item = get_message_to_user(prompt)

    content = item.thought
    event = Event(
        source=f"{companion_name}",
        content=content,
        embedding=controller.get_embedding(content),
        token=get_token(content),
        timestamp=datetime.now(),
        interlocus=InterlocusType.Thought.value
    )
    add_cognitive_event(event)

    content = item.message
    event = Event(
        source=f"{companion_name}",
        content=content,
        embedding=controller.get_embedding(content),
        token=get_token(content),
        timestamp=datetime.now(),
        interlocus=InterlocusType.Public.value
    )
    add_cognitive_event(event)
    return content


def action_internal_contemplation() -> str:
    prompt = get_prompt()
    item = get_internal_contemplation(prompt)
    content = item.contemplation
    event = Event(
        source=f"{companion_name}",
        content=content,
        embedding=controller.get_embedding(content),
        token=get_token(content),
        timestamp=datetime.now(),
        interlocus=InterlocusType.Thought.value
    )
    add_cognitive_event(event)
    return ""

def action_tool_call():
    ctx = get_recent_messages_block(6)
    tool_type = determine_tools(ctx)

    res = {}
    tool = get_tool_call(ctx, tool_type)
    tool.execute(res)

    content = f"{companion_name} uses her AI powers to call the tool. The tool response: {res['output']}"
    return "<tool>" + content

def action_reply():
    reply_consideration = generate_reply_consideration()
    add_cognitive_event(reply_consideration)

    return completion_story_mode()