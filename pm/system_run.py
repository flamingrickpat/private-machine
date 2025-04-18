import sys
import traceback

from pm.character import user_name, commit
from pm.controller import controller
from pm.database.db_utils import init_db
from pm.ghost.ghost import Ghost
from pm.subsystem.add_impulse.subsystem_impulse_to_event import SubsystemImpulseToEvent
from pm.subsystem.choose_action.subsystem_action_selection import SubsystemActionSelection
from pm.subsystem.create_action.sleep.sleep_procedures import check_optimize_memory_necessary, optimize_memory, extract_facts_categorize
from pm.subsystem.create_action.sleep.subsystem_sleep import SubsystemActionSleep
from pm.subsystem.create_action.tool_call.subsystem_tool_call import SubsystemToolCall
from pm.subsystem.create_action.user_reply.subsystem_user_reply import SubsystemUserReply
from pm.subsystem.plan_action.generate_reply_thoughts.subsystem_generate_reply_thoughts import SubsystemGenerateReplyThoughts
from pm.subsystem.sensation_evaluation.emotion.subsystem_emotion import SubsystemEmotion
from pm.system_classes import Impulse, ImpulseType, EndlessLoopError
from pm.system_utils import init, res_tag

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
    subsystem_sleep = SubsystemActionSleep()
    subsystem_reply_thoughts = SubsystemGenerateReplyThoughts()

    ghost = Ghost()
    ghost.register_subsystem(subsystem_response)
    ghost.register_subsystem(subsystem_choose_action)
    ghost.register_subsystem(subsystem_impulse_to_event)
    ghost.register_subsystem(subsystem_emotion)
    ghost.register_subsystem(subsystem_tools)
    ghost.register_subsystem(subsystem_sleep)
    ghost.register_subsystem(subsystem_reply_thoughts)

    extract_facts_categorize()
    if check_optimize_memory_necessary():
        optimize_memory()

    if inp == "/check":
        ghost.run_system_diagnosis()
        return "<SYSTEM OPERATIONAL>"
    else:
        imp = Impulse(is_input=True, impulse_type=ImpulseType.UserInput, endpoint=user_name, payload=inp)
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

    while True:
        controller.init_db()
        if is_debug:
            try:
                run_cli()
            except EndlessLoopError:
                controller.rollback_db()
            except Exception as e:
                raise e

            if not commit:
                print("rolling back...")
                controller.rollback_db()
            else:
                controller.commit_db()

            break
        else:
            try:
                try:
                    run_cli()
                except EndlessLoopError:
                    controller.rollback_db()
                except Exception as e:
                    raise e

                if not commit:
                    print("rolling back...")
                    controller.rollback_db()
                else:
                    controller.commit_db()
            except Exception as e:
                print(res_tag)
                print(e)
                print(traceback.format_exc())
                controller.rollback_db()
                break


def run_system_mp(inp: str) -> str:
    init_db()
    init()
    res = ""

    while True:
        controller.init_db()
        try:
            res = run_tick(inp)
            if not commit:
                print("rolling back...")
                controller.rollback_db()
            else:
                controller.commit_db()
            break
        except EndlessLoopError:
            controller.rollback_db()
        except Exception as e:
            res = f"{traceback.format_exc()}\n{e}"
            controller.rollback_db()
            break
    return res


if __name__ == '__main__':
    run_system_cli()
