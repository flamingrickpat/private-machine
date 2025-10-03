import os
import sys

sys.path.append("..")
os.chdir("..")

from pm.ghosts.base_ghost import GhostConfig
from pm.ghosts.ghost_lida import GhostLida
from pm.ghosts.persist_sqlite import PersistSqlite
from pm.tools.memquery import MemQuery
from pm_lida import main_llm, start_llm_thread

start_llm_thread()

config = GhostConfig()
ghost = GhostLida(main_llm, config)

pers = PersistSqlite(ghost)
if not pers.load_state_sqlite("./data/main.db"):
    raise Exception("create db with data!")

def test_queries():
    mem = MemQuery(ghost, main_llm)
    mem.init_default()

    # show welcome to the LLM (includes schemas + curated queries)
    print(mem.render_welcome_llm())

    # unit tests can iterate and execute each (after substituting placeholders)
    for qid, sql in mem.queries_for_tests():
        sql = sql.replace("{{TEXT}}", "'morning coffee preference and burnout signs'")
        df = mem.query(sql)
