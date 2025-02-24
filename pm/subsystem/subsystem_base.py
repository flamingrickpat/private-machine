from abc import ABC, abstractmethod
from typing import List

from pm.ghost.ghost_classes import PipelineStage, GhostState
from pm.system_classes import ImpulseType


class SubsystemBase(ABC):
    @abstractmethod
    def get_subsystem_name(self) -> str:
        pass

    @abstractmethod
    def get_subsystem_description(self) -> str:
        pass

    @abstractmethod
    def get_pipeline_state(self) -> List[PipelineStage]:
        pass

    @abstractmethod
    def get_impulse_type(self) -> List[ImpulseType]:
        pass

    @abstractmethod
    def get_action_types(self):
        pass

    @abstractmethod
    def proces_state(self, state: GhostState):
        pass

