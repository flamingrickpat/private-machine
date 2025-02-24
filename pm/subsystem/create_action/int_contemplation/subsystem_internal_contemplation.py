from datetime import datetime
from typing import List

from pydantic import BaseModel, Field

from pm.character import companion_name, sysprompt
from pm.controller import controller
from pm.database.db_utils import update_database_item
from pm.database.tables import InterlocusType, Event
from pm.embedding.token import get_token
from pm.ghost.ghost_classes import GhostState, PipelineStage
from pm.llm.base_llm import CommonCompSettings, LlmPreset
from pm.subsystem.subsystem_base import SubsystemBase
from pm.system_classes import ImpulseType, Impulse

def get_prompt():
    pass

class InternalContemplationItem(BaseModel):
    contemplation: str = Field(description="your contemplation about anything and stuff")


def get_internal_contemplation(items) -> InternalContemplationItem:
    msgs = [("system", sysprompt)]
    if items is not None:
        for item in items:
            msgs.append(item.to_tuple())

    _, calls = controller.completion_tool(LlmPreset.Conscious, msgs, comp_settings=CommonCompSettings(temperature=1, repeat_penalty=1.11, max_tokens=1024, presence_penalty=1, frequency_penalty=1),
                                          tools=[InternalContemplationItem])
    return calls[0]

class SubsystemInternalContemplation(SubsystemBase):
    def get_subsystem_name(self) -> str:
        pass

    def get_subsystem_description(self) -> str:
        pass

    def get_pipeline_state(self) -> List[PipelineStage]:
        pass

    def get_impulse_type(self) -> List[ImpulseType]:
        pass

    def get_action_types(self):
        pass

    def proces_state(self, state: GhostState):
        prompt = get_prompt()
        item = get_internal_contemplation(prompt)
        content = item.contemplation
        event = Event(
            source=f"{companion_name}",
            content=content,
            embedding=controller.get_embedding(content),
            token=get_token(content),
            timestamp=datetime.now(),
            interlocus=InterlocusType.Thought.value
        )
        update_database_item(event)

        state.output = Impulse(is_input=False, impulse_type=ImpulseType.UserOutput, endpoint="user", payload=content)