import datetime
import uuid
from enum import StrEnum
from typing import Optional, Any, List, Tuple

from sqlmodel import Field, Session, SQLModel, create_engine, select, col
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, text, INTEGER


class CogEventType(StrEnum):
    Thought = "thought"
    Action = "action"
    Observation = "observation"


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

class Event(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    source: str = Field(default="")
    content: str = Field(default="")
    embedding: Any = Field(sa_column=Column(Vector(768)))
    token: int = Field()
    timestamp: datetime.datetime
    interlocus: float = Field(ge=-1, le=1)

    def __repr__(self):
        return f"{self.source}: {self.content}"

    def to_prompt_item(self) -> List[PromptItem]:
        if self.interlocus == -2:
            return [
                PromptItem(
                    self.id,
                    self.timestamp,
                    "assistant",
                    "<think>\nThis conversation invokes strong feelings inside me.\n",
                    self.content,
                    "\nI'm a thinking and feeling being and will include these emotions in my final resposne directed at my user {user_name}..\n</think>",
                    -1)]
        elif self.interlocus == -1:
            return [
                PromptItem(
                    self.id,
                    self.timestamp,
                    "assistant",
                    "<think>\nTo craft a better response, I will now generate a Tree Of Thoughts based on recent events.\n",
                    self.content,
                    "\nAction: Now I'll include all these thoughts into the final resposne directed at my user {user_name}.\n</think>",
                    -1)]
        else:
            return [
                PromptItem(
                self.id,
                self.timestamp,
                "user" if self.source == "{user_name}" else "assistant",
                "",
                self.content,
                "",
                1)
            ]

class ClusterType(StrEnum):
    Topical = "topical"
    Temporal = "temporal"
    Conceptual = "conceptual"

class Cluster(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
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

    def to_prompt_item(self, data=None):
        if self.type == ClusterType.Topical:
            if data is None:
                raise Exception()

            pi = PromptItem(
                str(uuid.uuid4()),
                self.timestamp_from,
                "system",
                "",
                f"Current time: {self.timestamp_from}",
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
                f"You remember this happening between {self.timestamp_from} and {self.timestamp_to}: <memory>",
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
