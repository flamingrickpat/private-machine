import datetime
import uuid
from enum import StrEnum, IntEnum
from typing import Optional, Any, List

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column
from sqlmodel import Field, SQLModel, create_engine

from pm.character import timestamp_format, user_name, database_uri, companion_name

def get_tick_id():
    from pm.controller import controller
    return controller.current_tick_id

class PromptItem:
    def __init__(self, id, timestamp, turn, prefix, content, postfix, interlocus):#
        self.interlocus = interlocus
        self.id = id
        self.timestamp = timestamp
        self.turn = turn
        self.prefix = prefix
        self.content = content
        self.postfix = postfix

    def __repr__(self):
        return f"({self.turn}, {self.prefix} {self.content} {self.postfix})"

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def to_tuple(self):
        return (self.turn, f"{self.prefix}\n{self.content}\n{self.postfix}".strip())

class InterlocusType(IntEnum):
    Unconscious = -3
    Subconscious = -2
    ConsciousSubsystem = -2
    FirstPersonThought = -1
    SystemMessage = 0
    Public = 1

class TurnType(IntEnum):
    User = 1
    Assistant = 2
    System = 3
    NoTurn = 4

def get_guid():
    return str(uuid.uuid4())

class AgentMessages(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    agent_id: str = Field(...)
    turn: str = Field(...)
    content: str = Field(default="")
    embedding: Any = Field(sa_column=Column(Vector(768)))
    token: int = Field()
    timestamp: datetime.datetime
    rating: float = Field()
    turn_id: str = Field(default_factory=get_guid)

    def __repr__(self):
        return f"{self.turn}: {self.content}"


class Event(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    source: str = Field(default="")
    content: str = Field(default="")
    embedding: Any = Field(sa_column=Column(Vector(768)))
    token: int = Field()
    timestamp: datetime.datetime
    turn_story: str
    turn_assistant: str
    interlocus: int
    tick_id: int = Field(default_factory=get_tick_id)

    def __repr__(self):
        return f"{self.source}: {self.content}"

    def to_prompt_item(self, story_mode: bool) -> List[PromptItem]:
        if story_mode:

            if self.interlocus == InterlocusType.Public:
                return [
                    PromptItem(
                        self.id,
                        self.timestamp,
                        self.turn_story,
                        self.source + ': "',
                         self.content,
                        f'"',
                        self.interlocus)]
            elif self.interlocus == InterlocusType.ConsciousSubsystem:
                return [
                    PromptItem(
                        self.id,
                        self.timestamp,
                        self.turn_story,
                        f'{companion_name}\'s Subsystem "{self.source}" reports: "',
                        self.content,
                        f'"',
                        self.interlocus)]
        else:
            return [
                PromptItem(
                    self.id,
                    self.timestamp,
                    self.turn_assistant,
                    f"",
                    self.content,
                    f"",
                    self.interlocus)]

        """
            if self.interlocus == InterlocusType.Emotion.value:
                return [
                    PromptItem(
                        self.id,
                        self.timestamp,
                        "user",
                        f"{companion_name} is experiencing these emotions and needs right now:",
                        self.content,
                        f"",
                        -1)]
            elif self.interlocus == InterlocusType.Thought.value:
                return [
                    PromptItem(
                        self.id,
                        self.timestamp,
                        "user",
                        f"Let's make {companion_name}'s response more interesting, here is a tree of possibilities with considerations.",
                        self.content,
                        f"",
                        -1)]
            elif self.interlocus == InterlocusType.Contemplation.value:
                return [
                    PromptItem(
                        self.id,
                        self.timestamp,
                        "assistant",
                        f"<think>\nWhile {user_name} is away, I will contemplate on some stuff.\n",
                        self.content,
                        f"\nNow I'll wait for the next heartbeat.</think>",
                        -1)]
            elif self.interlocus == InterlocusType.ActionDecision.value:
                return [
                    PromptItem(
                        self.id,
                        self.timestamp,
                        "user",
                        f"Emmys thinks: '",
                        self.content,
                        f"'",
                        -1)]
            else:
                return [
                    PromptItem(
                        self.id,
                        self.timestamp,
                        "assistant",
                        "",
                        self.source + ": " + self.content,
                        "",
                        1)
                ]
        else:
            if self.interlocus == InterlocusType.Emotion.value:
                return [
                    PromptItem(
                        self.id,
                        self.timestamp,
                        "assistant",
                        "<think>\nThis conversation invokes strong feelings inside me.\n",
                        self.content,
                        f"\nI will use this feelings as basis for the next steps in my response planning.</think>",
                        -1)]
            elif self.interlocus == InterlocusType.Thought.value:
                return [
                    PromptItem(
                        self.id,
                        self.timestamp,
                        "assistant",
                        "<think>\nTo craft a better response, I will explore possible scenarios.\n",
                        self.content,
                        f"\nBased on this I will now draft a response to the user {user_name}.\n</think>",
                        -1)]
            elif self.interlocus == InterlocusType.Contemplation.value:
                return [
                    PromptItem(
                        self.id,
                        self.timestamp,
                        "assistant",
                        f"<think>\nWhile {user_name} is away, I will contemplate on some stuff.\n",
                        self.content,
                        f"\nNow I'll wait for the next heartbeat.</think>",
                        -1)]
            elif self.interlocus == InterlocusType.ActionDecision.value:
                return [
                    PromptItem(
                        self.id,
                        self.timestamp,
                        "assistant",
                        f"<think>",
                        self.content,
                        f"</think>",
                        -1)]
            else:
                return [
                    PromptItem(
                    self.id,
                    self.timestamp,
                    "user" if self.source == f"{user_name}" else "assistant",
                    "",
                    self.content,
                    "",
                    1)
                ]
                """

class ClusterType(StrEnum):
    Topical = "topical"
    Temporal = "temporal"
    Conceptual = "conceptual"

class Cluster(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    creation_date: datetime.datetime = Field(default_factory=datetime.datetime.now)
    type: ClusterType
    level: int = Field(default=0)
    summary: str = Field()
    included_messages: str = Field(default="")
    embedding: Any = Field(sa_column=Column(Vector(768)))
    token: int = Field()
    timestamp_from: datetime.datetime
    timestamp_to: datetime.datetime
    tag: str = Field(default="")

    # Temporary field, excluded from the DB schema
    similarity: float = Field(default=0.0, exclude=True)

    def to_prompt_item(self, story_mode: bool = False, data=None):
        if story_mode:
            if self.type == ClusterType.Topical:
                if data is None:
                    raise Exception()

                pi = PromptItem(
                    str(uuid.uuid4()),
                    self.timestamp_from,
                    "assistant",
                    "",
                    f"Current time: {self.timestamp_from.strftime(timestamp_format)}",
                    "",
                    1)

                data = [e for e in data]
                data.insert(0, pi)
                return data
            else:
                return [PromptItem(
                    str(uuid.uuid4()),
                    self.timestamp_from,
                    "assistant",
                    f"This happened between {self.timestamp_from.strftime(timestamp_format)} and {self.timestamp_to.strftime(timestamp_format)}:",
                    self.summary,
                    f"",
                    1)]
        else:
            if self.type == ClusterType.Topical:
                if data is None:
                    raise Exception()

                pi = PromptItem(
                    str(uuid.uuid4()),
                    self.timestamp_from,
                    "system",
                    "",
                    f"Current time: {self.timestamp_from.strftime(timestamp_format)}",
                    "",
                    1)

                data = [e for e in data]
                data.insert(0, pi)
                return data
            else:
                return [PromptItem(
                    str(uuid.uuid4()),
                    self.timestamp_from,
                    "system",
                    f"You remember this happening between {self.timestamp_from.strftime(timestamp_format)} and {self.timestamp_to.strftime(timestamp_format)}: <memory>",
                    self.summary,
                    f"</memory>",
                    1)]


class Fact(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    content: str
    temporality: float
    max_event_id: int
    embedding: Any = Field(sa_column=Column(Vector(768)))
    token: int = Field()

class EventCluster(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    cog_event_id: int
    con_cluster_id: int

class CauseEffectDbEntry(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    cause: str = Field(default="")
    effect: str = Field(default="")
    category: str = Field(default="")
    embedding: Any = Field(sa_column=Column(Vector(768)))
    token: int = Field()
    timestamp: datetime.datetime
    max_event_id: int

class CognitiveTick(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    counter: int
    parent_id: Optional[int] = Field(default=None)
    start_time: datetime.datetime
    end_time: Optional[datetime.datetime] = Field(default=None)

class WorldInteractionType(IntEnum):
    Sensation = 1
    Action = 2

class ImpulseRegister(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    tick_id: int
    is_input: bool
    impulse_type: str
    endpoint: str
    payload: str

class CognitiveState(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    tick_id: int
    key: str
    value: float

class SemanticKnowledgeTree(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    parent_node: Optional[int] = Field(default=None)
    name: str
    path: str

class FactSktWeight(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    skt_id: int
    fact_id: int
    weight: float


engine = create_engine(database_uri)
SQLModel.metadata.create_all(engine)