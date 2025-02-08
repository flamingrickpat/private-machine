import sys
import os
import uuid

from pm.agents.agent import AgentMessage
from pm.database.tables import EventCluster, Event, Cluster, AgentMessages

import time
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Optional, Any, List, Tuple, Union

from sqlmodel import Field, Session, SQLModel, create_engine, select, col
from pm.controller import controller
from pm.embedding.token import get_token
from pm.llm.base_llm import LlmPreset, CommonCompSettings
from rich import print as pprint

from pydantic import BaseModel, Field

from pm.controller import controller


class AgentHistorizedSimpel(BaseModel):
    id: str
    sysprompt: str

    def get_prompt(self, new_input: str, max_tokens: int):
        session = controller.get_session()
        msgs = session.exec(select(AgentMessages).
                            where(AgentMessages.agent_id == self.id).
                            where(AgentMessages.turn == "assistant").
                            order_by(AgentMessages.rating.desc()).
                            order_by(AgentMessages.timestamp.desc()).
                            limit(1000))
        lst = []
        tokens = 0
        for msg in msgs:
            corr_user_turns = session.exec(select(AgentMessages).where(AgentMessages.turn_id == msg.turn_id).where(AgentMessages.turn == "user"))
            corr_user_turn = corr_user_turns.first()
            nt = corr_user_turn.token + msg.token

            if tokens + nt > max_tokens:
                break
            lst.append(msg)
            lst.append(corr_user_turn)

            tokens += nt

        lst.reverse()

        lst = sorted(lst, key=lambda x: x.id)

        prompt = [(x.turn, x.content) for x in lst]

        prompt.insert(0, ("system", self.sysprompt))
        prompt.append(("user", new_input))
        return prompt

    def add_messages(self, user_query: str, agent_response: str, rating: float = 0.5):
        session = controller.get_session()
        f = AgentMessages(
            agent_id=self.id,
            turn="user",
            content=user_query.strip(),
            embedding=controller.get_embedding(user_query),
            token=get_token(user_query),
            timestamp=datetime.now(),
            rating=1
        )
        session.add(f)
        session.flush()
        session.refresh(f)
        turn_id = f.turn_id

        f = AgentMessages(
            agent_id=self.id,
            turn="assistant",
            content=agent_response.strip(),
            embedding=controller.get_embedding(agent_response),
            token=get_token(agent_response),
            timestamp=datetime.now(),
            rating=rating,
            turn_id=turn_id
        )
        session.add(f)
        session.flush()
        session.refresh(f)



if __name__ == '__main__':
    controller.init_db()
    session = controller.get_session()

    a = AgentHistorizedSimpel(id="cat_test", sysprompt="You are a helpful assistant.")

    p = "what is a cat?"
    prompt = a.get_prompt(p, 4000)
    res = controller.completion_text(LlmPreset.Default, prompt, comp_settings=CommonCompSettings(temperature=0.2, max_tokens=1024))
    a.add_messages(p, res, 0.9)

    p = "and the largest?"
    prompt = a.get_prompt(p, 4000)
    res = controller.completion_text(LlmPreset.Default, prompt, comp_settings=CommonCompSettings(temperature=0.2, max_tokens=1024))
    a.add_messages(p, res, 0.9)

    prompt = a.get_prompt("end", 4000)
    for p in prompt:
        print(p)

    controller.rollback_db()