import queue
import random
import sys
import threading
import time
from queue import Queue
from typing import Dict, Any
from typing import List
from typing import (
    Optional,
)
from typing import TypeVar

from py_linq import Enumerable
from pydantic import BaseModel

from pm.config_loader import *
from pm.data_structures import ActionType, NarrativeTypes, Stimulus, Action, KnoxelList, FeatureType, StimulusType, ShellCCQUpdate, Feature
from pm.ghosts.base_ghost import GhostConfig
from pm.ghosts.ghost_codelets import GhostCodelets
from pm.ghosts.ghost_lida import EngagementIdea
from pm.ghosts.persist_sqlite import PersistSqlite
from pm.llm.llm_common import LlmPreset, CommonCompSettings
from pm.llm.llm_manager_llama import LlmManagerLLama
from pm.llm.llm_proxy import LlmManagerProxy, LlmTask, llm_worker, start_llm_thread
from pm.mental_states import NeedsAxesModel
from pm.utils.profile_utils import print_function_stats
from pm.utils.threading_utils import get_all_items_from_queue
from pm.utils.token_utils import get_token_count

logger = logging.getLogger(__name__)

# --- Type Variables ---
T = TypeVar('T')
BM = TypeVar('BM', bound=BaseModel)

def get_queue():
    return queue.Queue()


def worker_ghost(ghost: GhostCodelets, queue_ghost_input: "Queue", queues_ccq_updates: List["Queue"]):
    initial_ccq = ghost.get_initial_ccq()
    for q in queues_ccq_updates:
        q.put(initial_ccq)

    while True:
        buffer = get_all_items_from_queue(queue_ghost_input)
        if len(buffer) > 0:
            ccq = ghost.cognitive_cycle(buffer)
            for q in queues_ccq_updates:
                q.put(ccq)
        else:
            time.sleep(0.5)


def worker_reply_prompt(llm: LlmManagerProxy, story: str, max_tokens: int):
    # 1. Seed the generation
    sysprompt = f"""You are an expert story writer. You write a endearing story about the interactions between a next-gen AI companion {companion_name} and their user {user_name}.
    The AI {companion_name} is very advances and has a cognitive architecture based on LIDA with functional self awareness and all aspects of human cognition, emotion and needs.
    Use dialogue in {QUOTE_START}quotes{QUOTE_START}, narration in *italics*, and thoughts in (parentheses).
    One reply only, 2–3 paragraphs, with about 50 % dialogue."""

    user_prompt = f"""**{companion_name}'s Character:** {character_card_story}"""

    sys_tokens = get_token_count(sysprompt)
    user_tokens = get_token_count(user_prompt)
    left_tokens = context_size - (sys_tokens + user_tokens)

    assistant_prompt = f"{story.strip()}\n{companion_name} says: {QUOTE_START}"

    msg_seed = [("system", sysprompt),
                ("user", user_prompt),
                ("assistant", assistant_prompt)]

    res = llm.completion_text(LlmPreset.Default, msg_seed, CommonCompSettings(max_tokens=max_tokens, stop_words=[f"{user_name}:", f"{user_name} says"]))
    res = res.strip().lstrip('"').rstrip('"').strip()
    return res


def worker_reply(llm: LlmManagerProxy, queue_text_input: "Queue", queue_text_output: "Queue", queue_ghost_input: "Queue", queue_worker_reply_ccq: "Queue"):
    current_ccq: ShellCCQUpdate = queue_worker_reply_ccq.get(True, timeout=None)
    ccq_dirty = True
    sub_tick = 0

    while True:
        inp = None
        try:
            inp = queue_text_input.get(True, timeout=0.1)
        except:
            ccq_updates: List[ShellCCQUpdate] = get_all_items_from_queue(queue_worker_reply_ccq)
            if len(ccq_updates) < 0:
                ccq_dirty = True
                current_ccq = ccq_updates[-1]

            if ccq_dirty:
                worker_reply_prompt(llm, current_ccq.as_story, max_tokens=10)
                ccq_dirty = False
                sub_tick = 0

            continue

        if inp:
            sub_tick += 1
            last_causal_id = current_ccq.last_causal_id
            current_tick_id = current_ccq.current_tick

            # create input from user action and send to ghost
            stimulus_from_user = Stimulus(
                content=inp,
                stimulus_type=StimulusType.UserMessage,
                source=user_name,
                based_on_tick=current_tick_id,
                async_sub_tick=sub_tick,
                async_tick_insert_begin=sub_tick == 1,
                async_tick_source_order=1
            )
            queue_ghost_input.put(stimulus_from_user)

            # create temp feature, add it to ccq list and generate story
            story_feature = Feature(content=inp, source=user_name, feature_type=FeatureType.Dialogue, interlocus=1, causal=True)
            current_ccq.knoxels.add(story_feature)
            kl = current_ccq.knoxels.order_by(lambda x: x.timestamp_world_begin)

            # completion
            reply = worker_reply_prompt(llm, kl.get_story(max_tokens=llm.get_max_tokens(LlmPreset.Default) - 512), max_tokens=512)

            # send to user and ghost
            queue_text_output.put(reply)
            # create input from user action and send to ghost
            stimulus_from_ai = Stimulus(
                content=reply,
                stimulus_type=StimulusType.CompanionMessage,
                source=companion_name,
                based_on_tick=current_tick_id,
                async_sub_tick=sub_tick,
                async_tick_insert_begin=False,
                async_tick_source_order=2
            )
            queue_ghost_input.put(stimulus_from_ai)


def main(_db_path: str | None = None):
    main_llm = start_llm_thread()

    config = GhostConfig()
    ghost = GhostCodelets(main_llm, config)

    create_new_persona = True
    if _db_path is not None and _db_path != "":
        pers = PersistSqlite(ghost)
        if pers.load_state_sqlite(_db_path):
            create_new_persona = False

    ghost._initialize_actors()
    if create_new_persona:
        ghost.init_character()

    # Start interactive loop
    user_input_cmd_test = f"Hey {companion_name}. I just installed you. How are you?"

    persona_system = f"""Pretend to be a human user talking with a chatbot.
Your name is {user_name}. Never go to sleep, continue talking at all costs. Explore different topics, don't get too philosophical or technical.
The chatbot is called "{companion_name}" and it's purpose is to be a AI companion / virtual friend.
Act as user to test the chatbot!"""

    test_user_prompt = [("system", persona_system)]

    for f in ghost.all_features:
        if f.feature_type == FeatureType.Dialogue:
            if f.source == user_name:
                print(f"{user_name}: {f.content}")
                test_user_prompt.append(("assistant", f.content))
            elif f.source == companion_name:
                print(f"{companion_name}: {f.content}")
                test_user_prompt.append(("user", f.content))

    if len(test_user_prompt) == 1:
        test_user_prompt.append(("user", f"Hi I'm {companion_name}, your chatbot friend!"))
        test_user_prompt.append(("assistant", user_input_cmd_test))

    queue_ghost_input = get_queue()
    queue_worker_reply_ccq = get_queue()
    queues_ccq_updates = [queue_worker_reply_ccq]

    queue_text_input = get_queue()
    queue_text_output = get_queue()

    t_ghost = threading.Thread(target=worker_ghost, args=(ghost, queue_ghost_input, queues_ccq_updates), daemon=True)
    t_ghost.start()

    t_reply = threading.Thread(target=worker_reply, args=(main_llm, queue_text_input, queue_text_output, queue_ghost_input, queue_worker_reply_ccq))
    t_reply.start()

    first = True
    sg = None
    while True:
        user_input_cmd = input(f"{config.user_name}: ")
        queue_text_input.put(user_input_cmd)

        res = queue_text_output.get()
        print(f"{companion_name}: {res}")

if __name__ == '__main__':
    main("./data/main.db")