import uuid
from typing import List, Type, Tuple

import pandas as pd
from lancedb.pydantic import LanceModel
from pydantic import BaseModel

from pm.controller import controller
from pm.database.db_model import User, Message, Conversation, MessageSummary, ConceptualCluster, Relation, Fact


# Initialize the database tables
def init_db():
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
            id=str(uuid.uuid4()),
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
        title="New Chat" if override_id is None else override_id.capitalize(),
        session_id=uuid.uuid4().hex
    )
    controller.db.open_table(Conversation.table).add([conversation])
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

def fetch_messages_as_string(conversation_id: str) -> str:
    messages = fetch_messages(conversation_id)
    message_block = "\n".join([f"{controller.config.companion_name if x.role == 'assistant' else controller.config.user_name}: {x.text}" for x in messages])
    return message_block


def fetch_relations(conversation_id: str) -> List[Relation]:
    tmp = sql_query(f"select * from relation")
    return df_to_pydantic(tmp, Relation)


def fetch_messages_no_summary(conversation_id: str) -> List[Message]:
    query = (f"select distinct m.* from message m "
             f"where m.id not in (select a from relation where rel_ab = 'summarized_by') "
             f"and m.conversation_id = '{conversation_id}' "
             f"order by m.created_at")
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

def insert_object(obj: LanceModel):
    tablename = obj.__class__.table
    controller.db.open_table(tablename).add([obj])

def rank_table(conversation_id: str, query: str, table: Type[LanceModel]) -> List[Tuple[LanceModel, float]]:
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

def get_facts(conversation_id: str, query: str, limit: int = 10) -> List[str]:
    # _relevance_score
    emb = controller.embedder.get_embedding_scalar(query)
    keywords = " ".join(controller.nlp.extract_keywords(query))

    scores = (controller.db.open_table(Fact.table)
                 .search(query_type="hybrid").vector(emb).text(keywords)
                 .limit(limit)
                 .to_pydantic(Fact))

    return [x.text for x in scores]
