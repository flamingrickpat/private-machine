import uuid
import datetime

from pm.clustering.summarize import cluster_and_summarize, high_level_summarize
from pm.controller import controller
from pm.database.db_helper import start_conversation, login_user
from pm.database.db_model import Message
from pm.utils.token_utils import quick_estimate_tokens

controller.start()

def create_data(inpath, username, ainame):
    user = login_user("user", "")
    convo_id = start_conversation(user.id, "main").id

    with open(inpath, 'r', encoding="utf-8") as file:
        lines = file.readlines()

    cur_time = datetime.datetime.now() - datetime.timedelta(days=30)
    for line in lines:
        if "#" in line:
            cur_topic = line
            cur_time += datetime.timedelta(hours=3)
        elif ":" in line:
            role = "user" if line.startswith(f"{username}:") else "assistant"
            inp = line.replace(f"{username}: ", "").replace(f"{ainame}: ", "").strip()
            response_message = Message(
                id=str(uuid.uuid4()),
                conversation_id=convo_id,
                role=role,
                public=True,
                text=inp,
                embedding=controller.embedder.get_embedding_scalar_float_list(inp),
                tokens=quick_estimate_tokens(inp),
                world_time=cur_time
            )
            controller.db.open_table(Message.table).add([response_message])
            cur_time += datetime.timedelta(minutes=5)

    cluster_and_summarize("main")
    high_level_summarize("main")


if __name__ == '__main__':
    file_path = r"E:\Workspace\Repositories\pm2\more_synth_data.txt"
    create_data(file_path)