import sys
import traceback

from pm.controller import controller
from pm.database.db_utils import init_db
from pm.ghost.ghost import Ghost
from pm.subsystem.add_impulse.subsystem_impulse_to_event import SubsystemImpulseToEvent
from pm.subsystem.choose_action.subsystem_action_selection import SubsystemActionSelection
from pm.subsystem.create_action.tool_call.subsystem_tool_call import SubsystemToolCall
from pm.subsystem.create_action.user_reply.subsystem_user_reply import SubsystemUserReply
from pm.subsystem.sensation_evaluation.emotion.subsystem_emotion import SubsystemEmotion
from pm.system_classes import Impulse, ImpulseType
from pm.system_utils import init, res_tag

TEST_MODE = False

def is_debug():
    import sys
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace is None:
        return False
    else:
        v = gettrace()
        if v is None:
            return False
        else:
            return True


def run_tick(inp):

    subsystem_response = SubsystemUserReply()
    subsystem_choose_action = SubsystemActionSelection()
    subsystem_impulse_to_event = SubsystemImpulseToEvent()
    subsystem_emotion = SubsystemEmotion()
    subsystem_tools = SubsystemToolCall()

    ghost = Ghost()
    ghost.register_subsystem(subsystem_response)
    ghost.register_subsystem(subsystem_choose_action)
    ghost.register_subsystem(subsystem_impulse_to_event)
    ghost.register_subsystem(subsystem_emotion)
    ghost.register_subsystem(subsystem_tools)

    imp = Impulse(is_input=True, impulse_type=ImpulseType.UserInput, endpoint="Rick", payload=inp)
    res = ghost.tick(imp)
    return res.payload


def run_cli() -> str:
    inp = sys.argv[1]
    if inp.strip() == "":
        print("no input")
        exit(1)

    res = run_tick(inp)
    print(res_tag + res)
    return res

def run_system_cli():
    init_db()
    init()

    controller.init_db()
    if is_debug:
        run_cli()
        if TEST_MODE:
            controller.rollback_db()
        else:
            controller.commit_db()
    else:
        try:
            run_cli()
            if TEST_MODE:
                controller.rollback_db()
            else:
                controller.commit_db()
        except Exception as e:
            print(res_tag)
            print(e)
            print(traceback.format_exc())
            controller.rollback_db()


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
