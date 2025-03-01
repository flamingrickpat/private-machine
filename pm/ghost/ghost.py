from datetime import datetime
from typing import List

from sqlmodel import select

from pm.character import companion_name
from pm.controller import controller
from pm.database.db_utils import update_database_item
from pm.database.tables import CognitiveTick, ImpulseRegister
from pm.ghost.ghost_classes import GhostState, PipelineStage, InvalidActionException
from pm.ghost.mental_state import db_to_base_model, EmotionalAxesModel, base_model_to_db, NeedsAxesModel
from pm.subsystem.create_action.sleep.sleep_procedures import check_optimize_memory_necessary
from pm.subsystem.subsystem_base import SubsystemBase
from pm.system_classes import ImpulseType, Impulse, ActionType


class Ghost:
    def __init__(self):
        self.depth = 0
        self.prev_tick_id = None
        self.current_tick_id = 0
        self.current_tick: CognitiveTick | None = None
        self.stage: PipelineStage = PipelineStage.Null
        self.subsystems: List[SubsystemBase] = list()

    def register_subsystem(self, sub: SubsystemBase):
        """
        Add a subsystem to the pipeline.
        """
        self.subsystems.append(sub)

    def tick(self, impulse: Impulse) -> Impulse:
        """
        External impulse comes in from environment, internal impulse goes out into environment.
        """
        self.depth = 0
        self._start_tick()
        res = self._tick_internal(impulse)
        self._end_tick()
        return res

    def run_system_diagnosis(self):
        """Check if self-optimization is necessary and execute it."""
        sleep_necessary = check_optimize_memory_necessary()
        if sleep_necessary:
            imp = Impulse(is_input=True, impulse_type=ImpulseType.SystemMessage, endpoint="System", payload=f"{companion_name}'s system reports that their memory is becoming fragmented. "
                                                                                                        f"Sleeping is required now. Otherwise critical system errors will appear.")
            self.tick(imp)

    def _tick_internal(self, impulse: Impulse) -> Impulse:
        """
        Main event loop with possible recursion.
        """
        self.depth += 1
        self._start_subtick()

        output = None
        state = self._add_impulse(impulse)
        if self._impulse_requires_action(impulse):
            self._eval_sensation(state)
            while True:
                try:
                    with controller.get_session().begin_nested():
                        temp_state = state.model_copy(deep=True)
                        self._choose_action(temp_state)
                        self._plan_action(temp_state)
                        self._create_action(temp_state)
                        status = self._verify_action(temp_state)
                        if status:
                            state = temp_state
                            break
                        else:
                            raise InvalidActionException()
                except InvalidActionException:
                    pass
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
            prev_tick_id=self.prev_tick_id,
            tick_id=self.current_tick_id,
            sensation=impulse,
            ghost=self,
            emotional_state=db_to_base_model(self.current_tick_id - 1, EmotionalAxesModel) if self.current_tick_id > 0 else EmotionalAxesModel(),
            needs_state=db_to_base_model(self.current_tick_id - 1, NeedsAxesModel) if self.current_tick_id > 0 else NeedsAxesModel()
        )

        self._execute_subsystems(state)
        return state


    def _eval_sensation(self, state: GhostState):
        self.stage = PipelineStage.EvalImpulse
        self._execute_subsystems(state)

    def _choose_action(self, state: GhostState):
        self.stage = PipelineStage.ChooseAction
        self._execute_subsystems(state)

    def _plan_action(self, state: GhostState):
        self.stage = PipelineStage.PlanAction
        pass

    def _create_action(self, state: GhostState):
        self.stage = PipelineStage.CreateAction
        self._execute_subsystems(state)

    def _verify_action(self, state: GhostState) -> bool:
        self.stage = PipelineStage.VerifyAction
        return True

    def _publish_output(self, state: GhostState) -> Impulse:
        self.stage = PipelineStage.PublishOutput
        base_model_to_db(self.current_tick.id, state.emotional_state)

        if state.output is None:
            return Impulse(is_input=False, impulse_type=ImpulseType.Empty, endpoint="env", payload="...")

        if state.output.impulse_type == ImpulseType.ToolResult:
            return self._tick_internal(state.output)
        if state.output.impulse_type == ImpulseType.UserOutput:
            return state.output

    def _execute_subsystems(self, state):
        systems = []
        for x in self.subsystems:
            add = True
            if self.stage not in x.get_pipeline_state():
                add = False

            if not (state.sensation.impulse_type in x.get_impulse_type() or ImpulseType.All in x.get_impulse_type()):
                add = False

            at = x.get_action_types()
            if at is not None:
                if state.action is not None and not (state.action.action_type in at or ActionType.All in at):
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
        parent_id = None

        session = controller.get_session()
        data = session.exec(select(CognitiveTick).order_by(CognitiveTick.id.desc())).all()
        if len(data) > 0:
            parent_id = data[0].id

        tick = CognitiveTick(
            counter=self.current_tick.counter + 1 if self.current_tick is not None else 1,
            start_time=datetime.now(),
            parent_id=parent_id
        )
        self.prev_tick_id = parent_id
        self.current_tick_id = update_database_item(tick)
        controller.current_tick_id = self.current_tick_id
        self.current_tick = tick

    def _end_tick(self):
        self.current_tick.end_time = datetime.now()
        update_database_item(self.current_tick)

    def _impulse_requires_action(self, impulse: Impulse):
        """
        Check if the ghost needs to act on the impulse.
        """
        if impulse.impulse_type == ImpulseType.SystemMessageSilent:
            return False

        return True


