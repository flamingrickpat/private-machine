import copy
import sys
import os
import traceback
import uuid

from pydantic import BaseModel

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')

from pm.controller import controller, log_conversation
from pm.actions import generate_action_selection, action_tool_call, action_reply, action_init_user_conversation, action_internal_contemplation
from pm.agents_gwt.gwt_layer import pregwt_to_internal_thought
from pm.agents_gwt.schema_main import GwtLayer
from pm.common_prompts.rate_complexity_v2 import rate_complexity_v2
from pm.common_prompts.rewrite_as_thought import rewrite_as_thought
from pm.cosine_clustering.cosine_cluster import get_best_category, cluster_text, TEMPLATE_CATEGORIES
from pm.fact_learners.extract_cause_effect_agent import extract_cas
from pm.sensation_evaluation import generate_sensation_evaluation
from pm.subsystems.subsystem_base import CogEvent
from pm.system import run_tick
from pm.system_classes import Sensation, SensationType, SystemState
from pm.system_utils import init_db, init, res_tag
from pm.validation.validate_directness import validate_directness
from pm.validation.validate_query_fulfillment import validate_query_fulfillment
from pm.validation.validate_response_in_context import validate_response_in_context

TEST_MODE = True

def run_cli() -> str:
    inp = sys.argv[1]
    if inp.strip() == "":
        print("no input")
        exit(1)

    res = run_tick(inp)
    print(res_tag + res)

def run_system_cli():
    init_db()
    init()

    controller.init_db()
    #try:
    if True:
        run_cli()
        if TEST_MODE:
            controller.rollback_db()
        else:
            controller.commit_db()
    #except Exception as e:
    #    print(res_tag)
    #    print(e)
    #    print(traceback.format_exc())
    #    controller.rollback_db()


def run_system_mp(inp: str) -> str:
    init_db()
    init()
    res = ""

    controller.init_db()
    try:
        res = run_tick(inp)
        if TEST_MODE:
            controller.rollback_db()
        else:
            controller.commit_db()
    except Exception as e:
        res = f"{traceback.format_exc()}\n{e}"
        controller.rollback_db()
    return res


if __name__ == '__main__':
    run_system_cli()
