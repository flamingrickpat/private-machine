from typing import Tuple, List

from pm.actions import generate_action_selection, action_tool_call, action_reply, action_init_user_conversation, action_internal_contemplation
from pm.sensation_evaluation import generate_sensation_evaluation
from pm.system_classes import Sensation, SensationType, SystemState
from pm.system_utils import ActionType, add_cognitive_event, generate_thought, check_tasks
from datetime import datetime, timedelta
from pm.controller import controller, log_conversation
from pm.embedding.token import get_token
from pm.database.tables import Event, EventCluster, Cluster, ClusterType, PromptItem, Fact, InterlocusType, AgentMessages, CauseEffectDbEntry
from pm.character import sysprompt, database_uri, cluster_split, companion_name, timestamp_format, sysprompt_addendum, char_card_3rd_person_neutral, user_name


def sensation_to_event(sen: Sensation) -> Event:
    content = sen.payload
    event = None
    if sen.sensation_type == SensationType.UserInput:
        event = Event(
            source=f"{user_name}",
            content=content,
            embedding=controller.get_embedding(content),
            token=get_token(content),
            timestamp=datetime.now(),
            interlocus=InterlocusType.Public.value
        )
        add_cognitive_event(event)
    elif sen.sensation_type == SensationType.ToolResult:
        event = Event(
            source=f"system",
            content=content,
            embedding=controller.get_embedding(content),
            token=get_token(content),
            timestamp=datetime.now(),
            interlocus=InterlocusType.SystemMessage.value
        )
        add_cognitive_event(event)
    return event


def inp_str_to_sensation(inp: str) -> Tuple[Sensation, List]:
    if inp.strip() == "/think":
        allowed_actions = [ActionType.InitiateUserConversation, ActionType.InitiateInternalContemplation, ActionType.ToolCall]
        payload = generate_thought()
        sen = Sensation(
            sensation_type=SensationType.UserInput,
            payload=payload
        )
    elif inp.strip() == "/sleep":
        check_tasks()

        allowed_actions = []
        sen = Sensation(
            sensation_type=SensationType.Empty,
            payload=""
        )
    elif inp.startswith("<tool>"):
        allowed_actions = [ActionType.Reply, ActionType.ToolCall]
        payload = inp.replace("<tool>", "")
        sen = Sensation(
            sensation_type=SensationType.ToolResult,
            payload=payload
        )
    else:
        allowed_actions = [ActionType.Reply, ActionType.ToolCall]
        payload = inp
        sen = Sensation(
            sensation_type=SensationType.UserInput,
            payload=payload
        )

    return sen, allowed_actions

def run_tick(inp: str) -> str:
    #system_state = system_state_load()
    #system_state.decay(0.05)

    # turn input string into formalized sensation with allowed action
    sen, allowed_actions = inp_str_to_sensation(inp)

    # turn sensation into cognitive event and save to sessiion
    sensation_event = sensation_to_event(sen)
    add_cognitive_event(sensation_event)

    # low-level evaluation of sensation
    sensation_evaluation = generate_sensation_evaluation()
    add_cognitive_event(sensation_evaluation)

    # select action
    action_selection, action_event = generate_action_selection(allowed_actions)
    add_cognitive_event(action_event)

    # do action (recursive if necessary)
    if action_selection == ActionType.Ignore:
        return f"{companion_name} chooses to ignore this sensation."
    if action_selection == ActionType.ToolCall:
        tool_result = action_tool_call()
        return run_tick(tool_result)
    elif action_selection == ActionType.Reply:
        return action_reply()
    elif action_selection == ActionType.InitiateIdleMode:
        return f"{companion_name} decides to go into idle mode."
    elif action_selection == ActionType.InitiateUserConversation:
        return action_init_user_conversation()
    elif action_selection == ActionType.InitiateInternalContemplation:
        return action_internal_contemplation()
