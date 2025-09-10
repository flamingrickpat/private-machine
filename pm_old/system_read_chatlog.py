import sys
import traceback

from sqlmodel import select

from pm.character import user_name, commit
from pm.controller import controller
from pm.database.db_utils import init_db
from pm.embedding.token import get_token
from pm.ghost.ghost import Ghost
from pm.subsystem.add_impulse.subsystem_impulse_to_event import SubsystemImpulseToEvent
from pm.subsystem.choose_action.subsystem_action_selection import SubsystemActionSelection
from pm.subsystem.create_action.sleep.sleep_procedures import check_optimize_memory_necessary, optimize_memory, extract_facts_categorize, clusterize, factorize, fact_categorize
from pm.subsystem.create_action.sleep.subsystem_sleep import SubsystemActionSleep
from pm.subsystem.create_action.tool_call.subsystem_tool_call import SubsystemToolCall
from pm.subsystem.create_action.user_reply.subsystem_user_reply import SubsystemUserReply
from pm.subsystem.plan_action.generate_reply_thoughts.subsystem_generate_reply_thoughts import SubsystemGenerateReplyThoughts
from pm.subsystem.reflect_learn.subsystem_narratives import SubsystemNarratives
from pm.subsystem.sensation_evaluation.emotion.subsystem_emotion import SubsystemEmotion
from pm.system_classes import Impulse, ImpulseType, EndlessLoopError
from pm.system_utils import init, res_tag
from pm.utils.chat_utils import parse_chatlog
from pm.character import companion_name
from pm.controller import controller
from pm.database.db_utils import update_database_item
from pm.database.tables import CognitiveTick, ImpulseRegister, Event, InterlocusType
from pm.ghost.ghost_classes import GhostState, PipelineStage, InvalidActionException
from pm.ghost.mental_state import db_to_base_model, EmotionalAxesModel, base_model_to_db, NeedsAxesModel
from pm.subsystem.create_action.sleep.sleep_procedures import check_optimize_memory_necessary
from pm.subsystem.subsystem_base import SubsystemBase
from pm.system_classes import ImpulseType, Impulse, ActionType

def init_from_chatlog(chatlog_path):
    init_db()
    init()

    tick_count = 0
    turns = parse_chatlog(chatlog_path)
    controller.init_db()

    for i in range(0, len(turns), 2):
        t_user = turns[i]
        t_ass = turns[i + 1]

        assert t_user.turn == user_name
        assert t_ass.turn == companion_name

        parent_id = -1

        session = controller.get_session()
        data = session.exec(select(CognitiveTick).order_by(CognitiveTick.id.desc())).all()
        if len(data) > 0:
            parent_id = data[0].id

        tick = CognitiveTick(
            counter=tick_count,
            start_time=t_user.timestamp,
            end_time=t_ass.timestamp,
            parent_id=parent_id
        )
        prev_tick_id = parent_id
        current_tick_id = update_database_item(tick)

        gs = GhostState(
            prev_tick_id=prev_tick_id,
            tick_id=current_tick_id,
            sensation=Impulse(is_input=True, endpoint="Rick", payload=t_user.content, impulse_type=ImpulseType.UserInput),
            ghost=None,
            emotional_state=db_to_base_model(prev_tick_id, EmotionalAxesModel) if prev_tick_id > 0 else EmotionalAxesModel(),
            needs_state=db_to_base_model(prev_tick_id, NeedsAxesModel) if prev_tick_id > 0 else NeedsAxesModel()
        )

        event = Event(
            source=f"{t_user.turn}",
            content=t_user.content,
            embedding=controller.get_embedding(t_user.content),
            token=get_token(t_user.content),
            timestamp=t_user.timestamp,
            interlocus=InterlocusType.Public.value,
            turn_story="assistant",
            turn_assistant="user",
            tick_id=current_tick_id
        )
        update_database_item(event)

        se = SubsystemEmotion()
        se.proces_state(gs)

        event = Event(
            source=f"{t_ass.turn}",
            content=t_ass.content,
            embedding=controller.get_embedding(t_ass.content),
            token=get_token(t_ass.content),
            timestamp=t_ass.timestamp,
            interlocus=InterlocusType.Public.value,
            turn_story="assistant",
            turn_assistant="assistant",
            tick_id=current_tick_id
        )
        update_database_item(event)

        base_model_to_db(current_tick_id, gs.emotional_state)


        sn = SubsystemNarratives()
        sn.proces_state(gs)

        if check_optimize_memory_necessary():
            clusterize()
        factorize()
        fact_categorize()
        controller.commit_db()

    print("ok")

