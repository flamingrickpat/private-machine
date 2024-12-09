import uuid
from typing import List, Type, Tuple

import pandas as pd
from lancedb.pydantic import LanceModel
from pydantic import BaseModel

from pm.consts import INIT_MESSAGE
from pm.controller import controller
from pm.database.db_model import User, Message, Conversation, MessageSummary, ConceptualCluster, Relation, Fact, Transaction, MessageInterlocus
from pm.utils.token_utils import quick_estimate_tokens


# Initialize the database tables
def init_db():
    controller.db.create_table(Transaction.table, schema=Transaction, exist_ok=True)
    controller.db.create_table(User.table, schema=User, exist_ok=True)
    controller.db.create_table(Message.table, schema=Message, exist_ok=True)
    controller.db.create_table(Conversation.table, schema=Conversation, exist_ok=True)
    controller.db.create_table(MessageSummary.table, schema=MessageSummary, exist_ok=True)
    controller.db.create_table(ConceptualCluster.table, schema=ConceptualCluster, exist_ok=True)
    controller.db.create_table(Relation.table, schema=Relation, exist_ok=True)
    controller.db.create_table(Fact.table, schema=Fact, exist_ok=True)

    try:
        controller.db.open_table(Message.table).create_fts_index("text")
    except Exception as e:
        if "already exists" not in str(e):
            raise e

    try:
        controller.db.open_table(ConceptualCluster.table).create_fts_index("text")
    except Exception as e:
        if "already exists" not in str(e):
            raise e

    try:
        controller.db.open_table(MessageSummary.table).create_fts_index("text")
    except Exception as e:
        if "already exists" not in str(e):
            raise e

    try:
        controller.db.open_table(Fact.table).create_fts_index("text")
    except Exception as e:
        if "already exists" not in str(e):
            raise e

def login_user(username: str, password: str) -> User:
    users_table = controller.db.open_table(User.table)
    user = users_table.search().where(f"username='{username}'", prefilter=True).limit(1).to_pydantic(model=User)

    if user:  # and user[0].password == password:
        return user[0]
    else:
        new_user = User(
            username=username,
            password=password
        )
        users_table.add([new_user])
        return new_user


def start_conversation(user_id: str, override_id: str | None = None) -> Conversation:
    if override_id is None:
        override_id = str(uuid.uuid4())

    conversation = Conversation(
        id=override_id,
        user_id=user_id,
        title=controller.config.companion_name,
        session_id=uuid.uuid4().hex
    )
    controller.db.open_table(Conversation.table).add([conversation])

    init_message = controller.format_str(INIT_MESSAGE)
    msg_init = Message(
        conversation_id=override_id,
        role='system',
        text=init_message,
        public=True,
        embedding=controller.embedder.get_embedding_scalar_float_list(init_message),
        tokens=quick_estimate_tokens(init_message),
        interlocus=MessageInterlocus.MessageSystemInst
    )
    controller.db.open_table(Message.table).add([msg_init])

    for memory in controller.config.initial_character_memory:
        f = Fact(
            conversation_id="main",
            text=memory,
            importance=0.5,
            embedding=controller.embedder.get_embedding_scalar_float_list(memory),
            category=controller.config.companion_name.lower(),
            tokens=quick_estimate_tokens(memory),
        )
        insert_object(f)

    return conversation


def fetch_conversations(user_id: str) -> List[Conversation]:
    return controller.db.open_table(Conversation.table).search().where(f"user_id='{user_id}'",
                                                                       prefilter=True).to_pydantic(
        model=Conversation)


def sql_query(query: str):
    import duckdb
    user = controller.db.open_table(User.table).to_lance()
    message = controller.db.open_table(Message.table).to_lance()
    conversation = controller.db.open_table(Conversation.table).to_lance()
    conceptual_cluster = controller.db.open_table(ConceptualCluster.table).to_lance()
    message_summary = controller.db.open_table(MessageSummary.table).to_lance()
    relation = controller.db.open_table(Relation.table).to_lance()
    res = duckdb.query(query).to_df()
    return res


def df_to_pydantic(df: pd.DataFrame, model: BaseModel) -> List[BaseModel]:
    """
    Convert a DataFrame to a list of Pydantic models.

    Args:
        df (pd.DataFrame): DataFrame containing data.
        model (BaseModel): Pydantic model class to instantiate.

    Returns:
        List[BaseModel]: List of instantiated Pydantic models.
    """
    return [model(**row.to_dict()) for _, row in df.iterrows()]


def fetch_messages(conversation_id: str) -> List[Message]:
    tmp = sql_query(f"select * from message where conversation_id='{conversation_id}' order by created_at asc")
    return df_to_pydantic(tmp, Message)

def role_to_name(x: Message):
    name = controller.config.companion_name if x.role == 'assistant' else (controller.config.user_name if x.role == "user" else "System")
    return name

def fetch_messages_as_string(conversation_id: str, n_thought_plans: int = 1, n_thought: int = 8, n_thought_emotion: int = 4, n_thought_feeling: int = 2) -> str:
    messages = fetch_messages(conversation_id)

    cnt_thought_plans = 0
    cnt_thoughts = 0
    cnt_thought_emotion = 0
    cnt_thought_feeling = 0
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.interlocus == MessageInterlocus.MessageThought:
            cnt_thoughts += 1
            if cnt_thoughts > n_thought:
                del messages[i]
        if msg.interlocus == MessageInterlocus.MessagePlanThought:
            cnt_thought_plans += 1
            if cnt_thought_plans > n_thought_plans:
                del messages[i]
        if msg.interlocus == MessageInterlocus.MessageEmotions:
            cnt_thought_emotion += 1
            if cnt_thought_emotion > n_thought_emotion:
                del messages[i]
        if msg.interlocus == MessageInterlocus.MessageFeeling:
            cnt_thought_feeling += 1
            if cnt_thought_feeling > n_thought_feeling:
                del messages[i]

    message_block = "\n".join([f"{role_to_name(x)}{' thinks' if MessageInterlocus.is_thought(x.interlocus) else ''}: {x.text}" for x in messages])
    return message_block

def fetch_responses(conversation_id: str) -> List[Message]:
    messages = fetch_messages(conversation_id)
    messages = [x for x in messages if not MessageInterlocus.is_thought(x.interlocus)]
    return messages

def fetch_responses_as_string(conversation_id: str) -> str:
    messages = fetch_messages(conversation_id)
    message_block = "\n".join([f"{role_to_name(x)}: {x.text}" for x in messages if not MessageInterlocus.is_thought(x.interlocus)])
    return message_block

def fetch_relations(conversation_id: str) -> List[Relation]:
    tmp = sql_query(f"select * from relation")
    return df_to_pydantic(tmp, Relation)


def fetch_messages_no_summary(conversation_id: str) -> List[Message]:
    query = (f"select distinct m.* from message m "
             f"where m.id not in (select a from relation where rel_ab = 'summarized_by') "
             f"and m.conversation_id = '{conversation_id}' "
             f"order by m.created_at asc")
    tmp = sql_query(query)
    return df_to_pydantic(tmp, Message)


def fetch_summaries(conversation_id: str, level: int) -> List[MessageSummary]:
    tmp = sql_query(f"select * from message_summary "
                    f"where conversation_id='{conversation_id}' "
                    f"and level = {level} order by created_at asc")
    return df_to_pydantic(tmp, MessageSummary)


def fetch_summaries_no_summary(conversation_id: str, level: int) -> List[Message]:
    query = (f"select distinct ms.* from message_summary ms "
             f"where ms.id not in (select a from relation where rel_ab = 'summarized_by') "
             f"and ms.conversation_id = '{conversation_id}' "
             f"and ms.level = {level}"
             f"order by ms.created_at")
    tmp = sql_query(query)
    return df_to_pydantic(tmp, MessageSummary)


def get_messages_by_id(ids: List[str]) -> List[Message]:
    res = []
    for id in ids:
        res.append(controller.db.open_table(Message.table).search().where(f"id='{id}'",
                                                       prefilter=True).limit(1).to_pydantic(model=Message))
    return res

def get_padded_subset(all_messages: List[Message], subset_messages: List[Message], padding: int = 10) -> List[Message]:
    # Extract the IDs of the first and last messages in the subset
    subset_start_id = subset_messages[0].id
    subset_end_id = subset_messages[-1].id

    # Find indices of the first and last subset messages in the full list
    start_idx = next(i for i, msg in enumerate(all_messages) if msg.id == subset_start_id)
    end_idx = next(i for i, msg in enumerate(all_messages) if msg.id == subset_end_id)

    # Calculate the padded range
    padded_start = max(0, start_idx - padding)
    padded_end = min(len(all_messages), end_idx + padding + 1)

    # Return the padded list of messages
    return all_messages[padded_start:padded_end]

def insert_object(obj: LanceModel | List[LanceModel]):
    tablename = obj.__class__.table
    if not isinstance(obj, list):
        obj = [obj]
    controller.db.open_table(tablename).add(obj)

def insert_system_message(conversation_id: str, sysmsg: str):
    msg_init = Message(
        conversation_id=conversation_id,
        role='system',
        text=sysmsg,
        public=True,
        embedding=controller.embedder.get_embedding_scalar_float_list(sysmsg),
        tokens=quick_estimate_tokens(sysmsg),
        interlocus=MessageInterlocus.MessageSystemInst
    )
    insert_object(msg_init)

def rank_table(conversation_id: str, query: str, table: Type[LanceModel]) -> List[Tuple[Message | Fact | MessageSummary, float]]:
    if query.strip() == "":
        return []

    # _relevance_score
    emb = controller.embedder.get_embedding_scalar(query)
    keywords = " ".join(controller.nlp.extract_keywords(query))

    scores = (controller.db.open_table(table.table)
                 .search(query_type="hybrid").vector(emb).text(keywords)
                 .where(f"conversation_id = '{conversation_id}'", prefilter=True)
                 .limit(10000)
                 .to_list()
                 )

    data = controller.db.open_table(table.table).search().limit(100000).to_pydantic(table)
    res = []
    for cur in data:
        _id = cur.id
        for score in scores:
            if score["id"] == _id:
                res.append((
                    cur,
                    score["_relevance_score"]
                ))
                break
    return res

def get_world_time_of_summary(summary_id: str):
    query = (f"select m.* from message m join relation r on m.id = r.a join message_summary ms on ms.id = r.b "
             f"where ms.id = '{summary_id}' "
             f"order by m.created_at limit 1")

    tmp = sql_query(query)
    return df_to_pydantic(tmp, Message)[0].created_at

def get_facts_str(conversation_id: str, query: str, limit: int = 10) -> List[str]:
    scores = get_facts(conversation_id, query, limit)
    return [x.text for x in scores]

def get_facts(conversation_id: str, query: str, limit: int = 10) -> List[Fact]:
    # _relevance_score
    emb = controller.embedder.get_embedding_scalar(query)
    keywords = " ".join(controller.nlp.extract_keywords(query))
    if keywords.strip() != "":
        scores = (controller.db.open_table(Fact.table)
                     .search(query_type="hybrid").vector(emb).text(keywords)
                     .limit(limit)
                     .to_pydantic(Fact))
    else:
        scores = (controller.db.open_table(Fact.table)
                     .search(query_type="hybrid").vector(emb)
                     .limit(limit)
                     .to_pydantic(Fact))

    return [x for x in scores]
