import json
import uuid
from typing import List

import streamlit as st
from duckdb.duckdb import rollback
from lancedb import LanceDBConnection

from pm.consts import THOUGHT_SEP
from pm.database.db_model import User, Message, Conversation, Fact, MessageInterlocus
from pm.architecture.system import AgentState, make_completion
from pm.config.config import read_config_file, MainConfig
from pm.controller import controller
from pm.database.db_helper import init_db, login_user, start_conversation, fetch_conversations, fetch_messages, insert_object
from pm.database.transactions import rollback_transaction
from pm.utils.token_utils import quick_estimate_tokens


def async_handle_llm(conversation_id: str, input: str) -> (int, str):
    # Fetch the current agent state for the conversation
    convo_table = controller.db.open_table(Conversation.table)
    convo = convo_table.search().where(f"id='{conversation_id}'", prefilter=True).limit(1).to_pydantic(
        model=Conversation)[0]

    # create langgraph state
    status = 0
    state = {
        "title": None,
        "input": input,
        "output": None,
        "status": status,
        "conversation_id": conversation_id,
    }

    # Make the completion
    #try:
    state = make_completion(state)
    #except Exception as e:
    #    return -1, repr(e)

    # Save the new state and the LLM response as a new message
    convo.agent_state = json.dumps(state)
    convo.title = f"Conversation {convo.id}"
    convo_table.update(where=f"id = '{convo.id}'", values=convo.model_dump())

    if status == 0:
        # Add the LLM response as a new message
        pass

    return 0, ""


def send_message(conversation_id: str, sender: str, text: str):
    status, msg = async_handle_llm(conversation_id, text)
    if status != 0:
        # CSS to style the "popup" (a div with red background)
        popup_css = """
            <style>
            .error-box {
                background-color: red;
                color: white;
                padding: 20px;
                border-radius: 10px;
                font-size: 18px;
            }
            </style>
            """

        st.markdown(popup_css, unsafe_allow_html=True)
        st.markdown(f'<div class="error-box">An error occurred: {msg}</div>', unsafe_allow_html=True)
    else:
        st.rerun()


def login_ui():
    username = st.text_input("Username", "user")
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

    if len(conversations) == 0:
        start_conversation(user.id, override_id="main")
        conversations = fetch_conversations(user.id)

    with st.sidebar:
        st.header("Conversations")
        for convo in conversations:
            if st.button(convo.title, key=convo.id):
                st.session_state['current_conversation_id'] = convo.id
                st.rerun()

        #if st.button("Start New"):
        #    convo_id = start_conversation(user.id).id
        #    st.session_state['current_conversation_id'] = convo_id
        #    st.rerun()

    if 'current_conversation_id' in st.session_state:
        # Function to display message in colored box
        def display_colored_message(text, color):
            color_class = color.lower()
            st.markdown(f'<div class="box {color_class}">{text}</div>', unsafe_allow_html=True)

        convo_id = st.session_state.get('current_conversation_id')
        if convo_id:
            convo = controller.db.open_table(Conversation.table).search().where(f"id='{convo_id}'", prefilter=True).limit(
                1).to_pydantic(model=Conversation)[0]
            st.header(f"{convo.title}")
            rollback_transaction()
            messages = fetch_messages(convo_id)
            for msg in messages:
                if msg.role == "user":
                    display_colored_message(msg.text, "green")
                elif msg.role == "system":
                    display_colored_message(msg.text, "yellow")
                else:
                    if msg.interlocus == MessageInterlocus.MessageThought:
                        display_colored_message(msg.text, "red")
                    elif msg.interlocus == MessageInterlocus.MessageResponse:
                        display_colored_message(msg.text, "blue")
                    else:
                        display_colored_message(msg.text, "orange")

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
    .box { padding: 10px; border-radius: 5px; margin-bottom: 10px; color: #333; }
    .green { background-color: #DFF0D8; }
    .blue { background-color: #D9EDF7; }
    .yellow { background-color: #FCF8E3; }
    .red { background-color: #F2DEDE; }
    .orange { background-color: #FFA500; }
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