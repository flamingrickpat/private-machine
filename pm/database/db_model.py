from pydantic import Field
from lancedb.pydantic import Vector, LanceModel
from typing import List, Optional
from datetime import datetime

from pygments.lexer import default


# SQLModel classes to define the schema for users, conversations, and messages

class User(LanceModel):
    id: str
    username: str
    password: str

User.table = "user"

class Conversation(LanceModel):
    id: str
    title: str = Field(default_factory=str)
    user_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    session_id: str = Field(default_factory=str)
    agent_state: str = Field(default_factory=str)

Conversation.table = "conversation"

class Message(LanceModel):
    id: str
    conversation_id: str
    sender: str
    text: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

Message.table = "message"
