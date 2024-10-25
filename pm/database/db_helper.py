import uuid
from typing import List
import re

import pandas as pd
from pydantic import BaseModel

from pm.database.db_model import User, Message, Conversation, MessageSummary, ConceptualCluster, Relation
from pm.controller import controller


# Initialize the database tables
def init_db():
    controller.db.create_table(User.table, schema=User, exist_ok=True)
    controller.db.create_table(Message.table, schema=Message, exist_ok=True)
    controller.db.create_table(Conversation.table, schema=Conversation, exist_ok=True)
    controller.db.create_table(MessageSummary.table, schema=MessageSummary, exist_ok=True)
    controller.db.create_table(ConceptualCluster.table, schema=ConceptualCluster, exist_ok=True)
    controller.db.create_table(Relation.table, schema=Relation, exist_ok=True)

    try:
        controller.db.open_table(MessageSummary.table).create_fts_index("text")
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


def start_conversation(user_id: str, override_id: str | None) -> Conversation:
    if override_id is None:
        override_id = str(uuid.uuid4())

    conversation = Conversation(
        id=override_id,
        user_id=user_id,
        title="New Chat",
        session_id=uuid.uuid4().hex
    )
    controller.db.open_table(Conversation.table).add([conversation])
    return conversation


def fetch_conversations(user_id: str) -> List[Conversation]:
    return controller.db.open_table(Conversation.table).search().where(f"user_id='{user_id}'",
                                                                       prefilter=True).to_pydantic(
        model=Conversation)


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
    tmp = sql_query("select * from message order by created_at asc")
    return df_to_pydantic(tmp, Message)

    #return controller.db.open_table(Message.table).search().where(f"conversation_id='{conversation_id}'",
    #                                                   prefilter=True).limit(1000000).to_pydantic(model=Message)


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

def fetch_messages_no_summary(conversation_id: str) -> List[Message]:
    data = sql_query("select distinct m.id as id from message m join relation r on m.id = r.a "
                     f"where r.rel_ab = 'summarized_by' and m.conversation_id = '{conversation_id}' "
                     "order by m.created_at")
    ids = []
    for id in data["id"]:
        ids.append(id)
    return get_messages_by_id(ids)


def get_messages_by_id(ids: List[str]) -> List[Message]:
    res = []
    for id in ids:
        res.append(controller.db.open_table(Message.table).search().where(f"id='{id}'",
                                                       prefilter=True).limit(1).to_pydantic(model=Message))
    return res

def fetch_documentation(guids: List[str]):
    res = []
    for guid in guids:
        doc = controller.db.open_table(Documentation.table).search().where(f"id='{guid}'",
                                                       prefilter=True).to_pydantic(model=Documentation)[0]
        res.append(doc)
    return res

def search_documentation(query: str, max_size: int):
    from stop_words import get_stop_words
    cleaned_text = re.sub(r'[^a-zA-Z0-9]', '', query)
    filtered_words = [word for word in cleaned_text.split() if word not in get_stop_words('english')]
    search_string = " ".join(filtered_words)

    emb = get_embedding(query)
    doc_pages = (controller.db.open_table(Documentation.table)
                 .search(query_type="hybrid").vector(emb).text(search_string)
                 .limit(15).to_pydantic(Documentation))

    token_cnt = 0
    results = []
    for page in doc_pages:
        if page.tokens + token_cnt > max_size and len(results) > 0:
            break
        results.append(page)
    return results


def search_documentation_text(query: str, max_size: int) -> str:
    results = search_documentation(query, max_size)
    return "\n".join([x.text for x in results])