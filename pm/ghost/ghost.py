from datetime import datetime
from enum import StrEnum
from typing import List, Optional

from pydantic import BaseModel, Field

from pm.actions import generate_action_selection
from pm.character import companion_name
from pm.database.db_utils import update_database_item
from pm.database.tables import CognitiveTick, ImpulseRegister, WorldInteractionType, Event
from pm.ghost.ghost_classes import GhostState, GhostAction, PipelineStage
from pm.sensation_evaluation import generate_sensation_evaluation
from pm.subsystem.subsystem_base import SubsystemBase
from pm.system_classes import ImpulseType, Impulse, ActionType
from pm.system_gamemaster import Action


class Ghost:
    def __init__(self):
        self.depth = 0
        self.current_tick_id = 0
        self.current_tick: CognitiveTick | None = None
        self.stage: PipelineStage = PipelineStage.Null
        self.subsystems: List[SubsystemBase] = list()

    def register_subsystem(self, sub: SubsystemBase):
        """
        Add a subsystem to the pipeline.
        """
        self.subsystems.append(sub)

    def tick(self, impulse: Impulse):
        """
        External impulse comes in from environment, internal impulse goes out into environment.
        """
        self.depth = 0
        self._start_tick()
        self._tick_internal(impulse)
        self._end_tick()

    def _tick_internal(self, impulse: Impulse) -> Impulse:
        """
        Main event loop with possible recursion.
        """
        self.depth += 1
        self._start_subtick()

        state = self._add_impulse(impulse)
        self._eval_sensation(state)
        self._choose_action(state)
        while True:
            temp_state = state.model_copy(deep=True)
            self._plan_action(temp_state)
            self._create_action(temp_state)
            status = self._verify_action(temp_state)
            if status:
                state = temp_state
                break
        output = self._publish_output(state)

        self._end_subtick()
        return output

    def _add_impulse(self, impulse: Impulse) -> GhostState:
        self.stage = PipelineStage.AddImpulse
        item = ImpulseRegister(
            tick_id=self.current_tick_id,
            impulse_type=str(impulse.impulse_type.value),
            is_input=impulse.is_input,
            endpoint=impulse.endpoint,
            payload=impulse.payload
        )
        update_database_item(item)
        state = GhostState(
            tick_id=self.current_tick_id,
            sensation=impulse
        )

        self._execute_subsystems(state)
        return state


    def _eval_sensation(self, state: GhostState):
        self.stage = PipelineStage.EvalImpulse
        self._execute_subsystems(state)
        #sensation_evaluation = generate_sensation_evaluation()
        #update_database_item(sensation_evaluation)
        #state.buffer_sensation_preprocessing.append(sensation_evaluation)

    def _choose_action(self, state: GhostState):
        self.stage = PipelineStage.ChooseAction
        self._execute_subsystems(state)

    def _plan_action(self, state: GhostState):
        self.stage = PipelineStage.PlanAction
        pass

    def _create_action(self, state: GhostState):
        self.stage = PipelineStage.CreateAction
        self._execute_subsystems(state)

        action_type = state.action.action_type
        if action_type == ActionType.Ignore:
            return f"{companion_name} chooses to ignore this sensation."
        if action_type == ActionType.ToolCall:
            tool_result = action_tool_call()
            return run_tick(tool_result)
        elif action_type == ActionType.Reply:
            return action_reply()
        elif action_type == ActionType.InitiateIdleMode:
            return f"{companion_name} decides to go into idle mode."
        elif action_type == ActionType.InitiateUserConversation:
            return action_init_user_conversation()
        elif action_type == ActionType.InitiateInternalContemplation:
            return action_internal_contemplation()


    def _verify_action(self, state: GhostState) -> bool:
        self.stage = PipelineStage.VerifyAction
        return True


    def _publish_output(self, state: GhostState) -> Impulse:
        self.stage = PipelineStage.PublishOutput
        if state.output.impulse_type == ImpulseType.ToolResult:
            return self._tick_internal(state.output)
        if state.output.output_type == ImpulseType.UserOutput:
            return state.output

    def _execute_subsystems(self, state):
        systems = []
        for x in self.subsystems:
            add = True
            if self.stage not in x.get_pipeline_state():
                add = False

            if not (state.sensation.impulse_type in x.get_impulse_type() or ImpulseType.All in x.get_impulse_type()):
                add = False

            if state.action is not None and not (state.action.action_type in x.get_action_types() or ActionType.All in x.get_action_types()):
                add = False

            if add:
                systems.append(x)

        for system in systems:
            system.proces_state(state)


    def _start_subtick(self):
        pass

    def _end_subtick(self):
        pass

    def _start_tick(self):
        tick = CognitiveTick(
            counter=self.current_tick.counter + 1,
            start_time=datetime.now()
        )
        self.current_tick_id = update_database_item(tick)
        self.current_tick = tick

    def _end_tick(self):
        self.current_tick.end_time = datetime.now()
        update_database_item(self.current_tick)

