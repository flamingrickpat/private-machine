import os.path
import sqlite3
import json
import time
import traceback
from datetime import datetime, timedelta
from multiprocessing import Process
from typing import List, Dict, Any, Optional
import uuid
from threading import Thread
from time import sleep
import logging
import os

from fastmcp import FastMCP

from pm.utils.singleton_utils import singleton

PORT = 11223

@singleton
class State:
    def __init__(self):
        from pm.tools.memquery import MemQuery
        from pm.ghosts.ghost_lida import GhostLida
        self.ghost: GhostLida = None
        self.mem: MemQuery = None

logger = logging.getLogger(__name__)

# --- Server Initialization ---
mcp = FastMCP("Knowledge MCP Server")

def kill_self():
    while True:
        time.sleep(5)
        os._exit(1)

@mcp.tool()
def get_help(topic: str) -> str:
    """
    Call this for information on how to use knowledge MCP server.
    :param topic: What you need help about.
    :return: help text for sql commands and tools.
    """
    return State().mem.render_welcome_llm()

@mcp.tool()
def report_gathered_knowledge(answer: str) -> str:
    """
    Call this with your final answer to the questions when you're done.
    Or report technical issues with the SQL interface if you encounter any. A technician will fix it!
    There is no length limit to your answer, make it as detailed as possible.
    :return: your supervisors next task
    """
    with open("test.txt", "w", encoding="utf-8") as f:
        f.write(answer)

    t = Thread(target=kill_self)
    t.start()
    return "kill_generation"

@mcp.tool()
def execute_query(query: str) -> str:
    """
    Execute a complex query. Use get_full_content_by_id to get full content of knowledge item
    :param query: dubdb conform sql query
    :return: csv output
    """
    s = State()
    try:
        df = s.mem.query(query)
        out = s.mem.to_csv(df, max_rows=24)
        return out
    except Exception as e:
        tb = traceback.format_exc()
        return str(tb)

@mcp.tool()
def get_full_content_by_id(knowledge_id: int) -> str:
    """
    Every ID of knowledge items is globally unique.
    Use this function to get the full content instead of the shortened version in the table view.
    :param knowledge_id: id (usually first column)
    :return: full content, no cutoff
    """
    s = State()
    knx = s.ghost.get_knoxel_by_id(knowledge_id)
    if knx is None:
        return "Invalid ID, no Knoxel found!"
    return knx.content

def knowledge_mcp_process(db_path):
    os.chdir("../..")
    s = State()

    from pm.ghosts.base_ghost import GhostConfig
    from pm.ghosts.ghost_lida import GhostLida
    from pm.ghosts.persist_sqlite import PersistSqlite
    from pm.tools.memquery import MemQuery
    from pm_lida import main_llm, start_llm_thread

    start_llm_thread()

    config = GhostConfig()
    ghost = GhostLida(main_llm, config)

    pers = PersistSqlite(ghost)
    if not pers.load_state_sqlite(db_path):
        return False

    mem = MemQuery(ghost, main_llm)
    mem.init_default()

    s.ghost = ghost
    s.mem = mem
    mcp.run(transport="streamable-http", host="127.0.0.1", port=PORT, path="/mcp", uvicorn_config={"log_level": "critical"})
    return True

def start_knowledge_mcp_process(ghost):
    db_path = ghost.current_db_path
    p = Process(target=knowledge_mcp_process, args=(db_path,), daemon=True)
    p.start()
    return PORT