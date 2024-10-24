import uuid
from typing import List
import re
from pm.database.db_model import User, Message, Conversation, MessageSummary, ConceptualCluster
from pm.controller import controller


# Initialize the database tables
def init_db():
    controller.db.create_table(User.table, schema=User, exist_ok=True)
    controller.db.create_table(Message.table, schema=Message, exist_ok=True)
    controller.db.create_table(Conversation.table, schema=Conversation, exist_ok=True)
    controller.db.create_table(MessageSummary.table, schema=MessageSummary, exist_ok=True)
    controller.db.create_table(ConceptualCluster.table, schema=ConceptualCluster, exist_ok=True)

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


def start_conversation(user_id: str) -> Conversation:
    conversation = Conversation(
        id=str(uuid.uuid4()),
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


def fetch_messages(conversation_id: str) -> List[Message]:
    return controller.db.open_table(Message.table).search().where(f"conversation_id='{conversation_id}'",
                                                       prefilter=True).to_pydantic(model=Message)


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