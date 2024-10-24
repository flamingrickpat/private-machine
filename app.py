import json
import uuid

import streamlit as st
from lancedb import LanceDBConnection

from pm.database.db_model import User, Message, Conversation
from pm.system import AgentState, make_completion
from pm.config.config import read_config_file, MainConfig

# Initialize the LanceDB database
cfg = read_config_file("config.json")
db = LanceDBConnection(cfg.db_path)

# Initialize the database tables
def init_db():
    db.create_table("user", schema=User, exist_ok=True)
    db.create_table("message", schema=Message, exist_ok=True)
    db.create_table("conversation", schema=Conversation, exist_ok=True)

def async_handle_llm(conversation_id: str, text: str):
    # Fetch the current agent state for the conversation
    convo_table = db.open_table(Conversation.table)
    convos = convo_table.search().where(f"id='{conversation_id}'", prefilter=True).limit(1).to_pydantic(
        model=Conversation)

    state = {"messages": []}
    convo = convos[0]
    if convo and convo.agent_state:
        state = json.loads(convo.agent_state)

    # Make the completion
    try:
        state["title"] = "Conversation"
        state["conversation_id"] = conversation_id
        state["messages"] = []
        msgs = fetch_messages(conversation_id)
        for msg in msgs:
            state["messages"].append({
                "role": msg.role,
                "content": msg.text
            })
        state = make_completion(state)
    except Exception as e:
        response_message = Message(
            id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            role='assistant',
            text=f"Exception: {repr(e)}"
        )
        db.open_table(Message.table).add([response_message])

    # Save the new state and the LLM response as a new message
    convo.agent_state = json.dumps(state)
    convo.title = state["title"]
    convo_table.update(where=f"id = '{convo.id}'", values=convo.model_dump())

    # Add the LLM response as a new message
    response_message = Message(
        id=str(uuid.uuid4()),
        conversation_id=conversation_id,
        role='assistant',
        text=state['output']
    )
    db.open_table(Message.table).add([response_message])


def login_user(username: str, password: str):
    users_table = db.open_table(User.table)
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


def start_conversation(user_id: str):
    conversation = Conversation(
        id=str(uuid.uuid4()),
        user_id=user_id,
        title="Unnamed",
        session_id=uuid.uuid4().hex
    )
    db.open_table(Conversation.table).add([conversation])
    return conversation.id


def fetch_conversations(user_id: str):
    return db.open_table(Conversation.table).search().where(f"user_id='{user_id}'", prefilter=True).to_pydantic(
        model=Conversation)


def fetch_messages(conversation_id: str):
    return db.open_table(Message.table).search().where(f"conversation_id='{conversation_id}'",
                                                       prefilter=True).to_pydantic(model=Message)


def send_message(conversation_id: str, sender: str, text: str):
    message = Message(
        id=str(uuid.uuid4()),
        conversation_id=conversation_id,
        role=sender,
        text=text
    )
    db.open_table(Message.table).add([message])

    async_handle_llm(conversation_id, text)
    st.rerun()


def login_ui():
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        user = login_user(username, password)
        if user:
            st.session_state['user'] = user
            st.success("Logged in successfully!")
            st.rerun()
        else:
            st.error("Invalid username or password")
            st.rerun()


def chat_ui():
    user = st.session_state['user']
    conversations = fetch_conversations(user.id)

    with st.sidebar:
        st.header("Conversations")
        for convo in conversations:
            if st.button(convo.title, key=convo.id):
                st.session_state['current_conversation_id'] = convo.id
                st.rerun()

        if st.button("Start New"):
            convo_id = start_conversation(user.id)
            st.session_state['current_conversation_id'] = convo_id
            st.rerun()

    if 'current_conversation_id' in st.session_state:
        convo_id = st.session_state.get('current_conversation_id')
        if convo_id:
            convo = db.open_table(Conversation.table).search().where(f"id='{convo_id}'", prefilter=True).limit(
                1).to_pydantic(model=Conversation)[0]
            st.header(f"{convo.title}")
            messages = fetch_messages(convo_id)
            for msg in messages:
                if msg.role == "user":
                    st.success(f"You: {msg.text}")
                else:
                    st.info(f"Companion: {msg.text}")

            new_message = st.text_input("Your message", key=f"new_msg_{convo_id}")
            if st.button("Send", key=f"send_{convo_id}"):
                send_message(convo_id, "user", new_message)
                st.rerun()


init_db()
if 'user' not in st.session_state:
    login_ui()
else:
    chat_ui()

# Style modifications
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        height: 3em;
        background-color: #0e1117;
        color: white;
    }
    .stTextInput>div>div>input {
        color: black;
    }
    .css-18e3th9 {
        padding: 0!important;
    }
</style>
""", unsafe_allow_html=True)