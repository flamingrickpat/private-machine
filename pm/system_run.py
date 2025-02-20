import copy
import sys
import os
import traceback
import uuid
from pm.controller import controller, log_conversation
from pm.system import run_tick
from pm.system_utils import init_db, init, res_tag

TEST_MODE = True

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
