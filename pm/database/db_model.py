import enum
import uuid
from operator import truediv

from lancedb.pydantic import Vector, LanceModel
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

def get_guid():
    return str(uuid.uuid4())

# Transaction Table to store the start time of each transaction
class Transaction(LanceModel):
    id: str = Field(default_factory=get_guid)
    created_at: datetime = Field(default_factory=datetime.utcnow)  # Timestamp of when the transaction started
    state: str = "active"  # 'active' or 'completed'
Transaction.table = "Transaction"

class User(LanceModel):
    id: str = Field(default_factory=get_guid)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    username: str
    password: str
User.table = "User"

class Conversation(LanceModel):
    id: str = Field(default_factory=get_guid)
    title: str = Field(default_factory=str)
    user_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    session_id: str = Field(default_factory=str)
    agent_state: str = Field(default_factory=str)
Conversation.table = "Conversation"

class MessageInterlocus(enum.IntEnum):
    Undefined = 0
    MessageSystemInst = 5
    MessageEmotions = 7
    MessageFeeling = 8
    MessagePlanThought = 9
    MessageThought = 10
    MessageToolCall = 15
    MessageResponse = 20

    @staticmethod
    def is_thought(il):
        if il == MessageInterlocus.MessageThought or il == MessageInterlocus.MessagePlanThought or il == MessageInterlocus.MessageEmotions or il == MessageInterlocus.MessageFeeling:
            return True
        return False


class Message(LanceModel):
    id: str = Field(default_factory=get_guid)
    conversation_id: str # conversation guid
    created_at: datetime = Field(default_factory=datetime.utcnow)
    world_time: datetime = Field(default_factory=datetime.now)
    role: str # user or assistant
    public: bool # public or internal thought
    text: str # content
    embedding: Vector(1024) # embedding
    tokens: int # rough token count
    interlocus: int # -10 = purely internal, 10 = internal thought about conversation, 20 = message

    def __eq__(self, other):
        if isinstance(other, Message):
            return self.id == other.id
        return False

    def __hash__(self):
        return hash(self.id)

    @property
    def full_text(self):
        from pm.controller import controller
        return f"{controller.config.companion_name if self.role == 'assistant' else 'System' if self.role == 'system' else controller.config.user_name}: {self.text}"

    class Config:
        extra = "allow"
Message.table = "Message"

class MessageSummary(LanceModel):
    """
    Cluster of messages for episodic memory, separated by topic.
    Can consist on n messages or n message clusters of lower level.
    """
    id: str = Field(default_factory=get_guid)
    conversation_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    world_time_begin: datetime = Field(default_factory=datetime.now)
    world_time_end: datetime = Field(default_factory=datetime.now)

    level: int # 0 = topic messages with 1 message extra, > 0 = 4 summaries of lower level with 1 overlap on each side
    text: str
    embedding: Vector(1024)
    tokens: int  # rough token count

    def __eq__(self, other):
        if isinstance(other, MessageSummary):
            return self.id == other.id
        return False

    def __hash__(self):
        return hash(self.id)

    @property
    def world_time(self):
        return self.world_time_begin

    class Config:
        extra = "allow"
MessageSummary.table = "MessageSummary"

class ConceptualCluster(LanceModel):
    """
    Can consist of n messages, will get merged with overlapping clusters.
    """
    id: str = Field(default_factory=get_guid)
    conversation_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    title: str # rough title of content
    text: str # summarized knowledge about that concept
    embedding: Vector(1024)
    tokens: int  # rough token count

    gmm_mean: List[float]  # Mean of the GMM (1 x 1024)
    gmm_covariance: List[List[float]]  # Covariance matrix (1024 x 1024)
    gmm_weight: float  # Weight for the GMM (since we have 1 component)

    def __eq__(self, other):
        if isinstance(other, ConceptualCluster):
            return self.id == other.id
        return False

    def __hash__(self):
        return hash(self.id)
ConceptualCluster.table = "ConceptualCluster"

class Fact(LanceModel):
    id: str = Field(default_factory=get_guid)
    conversation_id: str # conversation guid
    created_at: datetime = Field(default_factory=datetime.utcnow)
    world_time: datetime = Field(default_factory=datetime.now)

    text: str # content
    importance: float
    embedding: Vector(1024) # embedding
    tokens: int # rough token count
    category: str # can be either user, companion or environment

    def __eq__(self, other):
        if isinstance(other, Message):
            return self.id == other.id
        return False

    def __hash__(self):
        return hash(self.id)
    class Config:
        extra = "allow"
Fact.table = "fact"


class Relation(LanceModel):
    id: str = Field(default_factory=get_guid)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    a: str
    b: str
    rel_ab: str
Relation.table = "Relations"

class Reminder(LanceModel):
    id: str = Field(default_factory=get_guid)
    reminder_text: str = Field()
    created_at: datetime = Field(default_factory=datetime.utcnow)
    reminder_time: datetime = Field()
    session_id: str = Field(default_factory=str)
Reminder.table = "Reminder"

class Goal(LanceModel):
    id: str = Field(default_factory=get_guid)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    session_id: str = Field(default_factory=str)
    identifier: str
    goal: str
    description: str
    status: str
    progress: float
Goal.table = "Goal"