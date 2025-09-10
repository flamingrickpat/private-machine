from enum import StrEnum

from pydantic import BaseModel

class EndlessLoopError(Exception):
    pass

class ActionType(StrEnum):
    Sleep = "Sleep"
    All = "All"
    Reply = "Reply"
    ToolCall = "ToolCall"
    Ignore = "Ignore"
    InitiateUserConversation = "InitiateUserConversation"
    InitiateInternalContemplation = "InitiateInternalContemplation"
    InitiateIdleMode = "InitiateIdleMode"
    Think = "Think"
    RecallMemory = "RecallMemory"
    ManageIntent = "ManageIntent"
    Plan = "Plan"
    ManageAwarenessFocus = "ManageAwarenessFocus"
    ReflectThoughts = "ReflectThoughts"

class ImpulseType(StrEnum):
    SystemMessageSilent = "SystemMessageSilent"
    SystemMessage = "SystemMessage"
    WakeUp = "WakeUp"
    All = "All"
    Empty = "Empty"
    ToolResult = "ToolResult"
    Thought = "Thought"
    UserInput = "UserInput"
    UserOutput = "UserOutput"
    VirtualWorld = "VirtualWorld"
    Contemplation = "Contemplation"


class Impulse(BaseModel):
    is_input: bool  # True for sensation/input, False for output
    impulse_type: ImpulseType
    endpoint: str   # Source if input, Target if output
    payload: str

