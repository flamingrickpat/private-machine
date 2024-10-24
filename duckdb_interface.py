import json
import uuid
from typing import List

import duckdb
from lancedb import LanceDBConnection

from pm.database.db_model import User, Message, Conversation, ConceptualCluster, MessageSummary
from pm.system import AgentState, make_completion
from pm.config.config import read_config_file, MainConfig
from pm.controller import controller
from pm.database.db_helper import init_db, login_user, start_conversation, fetch_conversations, fetch_messages
from pm.utils.token_utils import quick_estimate_tokens

user = controller.db.open_table(User.table).to_lance()
message = controller.db.open_table(Message.table).to_lance()
conversation = controller.db.open_table(Conversation.table).to_lance()
conceptual_cluster = controller.db.open_table(ConceptualCluster.table).to_lance()
message_summary = controller.db.open_table(MessageSummary.table).to_lance()

print("ready...")
while True:
    try:
        query = input()
        res = duckdb.query(query)
        print(res)
    except Exception as e:
        print(e)