import uuid

from lancedb.pydantic import Vector, LanceModel
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

def get_guid():
    return str(uuid.uuid4())

class User(LanceModel):
    id: str
    username: str
    password: str
User.table = "User"

class Conversation(LanceModel):
    id: str
    title: str = Field(default_factory=str)
    user_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    session_id: str = Field(default_factory=str)
    agent_state: str = Field(default_factory=str)
Conversation.table = "Conversation"

class Message(LanceModel):
    id: str # guid
    conversation_id: str # conversation guid
    created_at: datetime = Field(default_factory=datetime.utcnow)

    role: str # user or assistant
    public: bool # public or internal thought
    text: str # content
    embedding: Vector(1024) # embedding
    tokens: int # rough token count
Message.table = "Message"

class MessageSummary(LanceModel):
    """
    Cluster of messages for episodic memory, separated by topic.
    Can consist on n messages or n message clusters of lower level.
    """
    id: str = Field(default_factory=get_guid)
    conversation_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    level: int # 0 = topic messages with 1 message extra, > 0 = 4 summaries of lower level with 1 overlap on each side
    text: str
    embedding: Vector(1024)
    tokens: int  # rough token count
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
ConceptualCluster.table = "ConceptualCluster"

class Relation(LanceModel):
    id: str = Field(default_factory=get_guid)
    a: str
    b: str
    rel_ab: str
Relation.table = "Relations"