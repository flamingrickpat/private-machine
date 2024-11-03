import uuid
import datetime

from pm.clustering.summarize import cluster_and_summarize, high_level_summarize
from pm.consts import RECALC_SUMMARIES_MESSAGES
from pm.controller import controller
from pm.database.db_helper import start_conversation, login_user, insert_object
from pm.database.db_model import Message, MessageInterlocus
from pm.utils.token_utils import quick_estimate_tokens

controller.start()

def create_data(inpath, username, ainame):
    user = login_user("user", "")
    convo_id = start_conversation(user.id, "main").id

    with open(inpath, 'r', encoding="utf-8") as file:
        lines = file.readlines()

    cur_time = datetime.datetime.now() - datetime.timedelta(days=30)
    for cnt, line in enumerate(lines):
        if "#" in line:
            cur_topic = line
            cur_time += datetime.timedelta(hours=3)
        elif ":" in line:
            role = "user" if line.startswith(f"{username}:") else "assistant"
            inp = line.replace(f"{username}: ", "").replace(f"{ainame}: ", "").strip()
            response_message = Message(
                conversation_id=convo_id,
                role=role,
                public=True,
                text=inp,
                embedding=controller.embedder.get_embedding_scalar_float_list(inp),
                tokens=quick_estimate_tokens(inp),
                world_time=cur_time,
                interlocus=MessageInterlocus.Message
            )
            insert_object(response_message)
            cur_time += datetime.timedelta(minutes=5)

        if cnt % RECALC_SUMMARIES_MESSAGES:
            cluster_and_summarize("main")
            high_level_summarize("main")


if __name__ == '__main__':
    file_path = r"./synth_data/synth_data1.txt"
    create_data(file_path, username="Rick", ainame="Nova")