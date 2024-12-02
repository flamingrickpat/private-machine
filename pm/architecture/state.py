from typing import TypedDict, Dict, List

from pydantic import BaseModel, Field

class EmotionalAxesModel(BaseModel):
    """
    Determine current emotion vector. This will be added to current emotions.
    """
    valence: float = Field(default=0.5, ge=-1.0, le=1.0, description="Ranges from sadness (-1) to happiness (+1), with neutrality (0).")
    safety: float = Field(default=0.5, ge=-1.0, le=1.0, description="Ranges from security (-1) to fear (+1), with a neutral point (0).")
    affinity: float = Field(default=0.5, ge=-1.0, le=1.0, description="Ranges from attraction (-1) to disgust (+1), with indifference (0).")
    calmness: float = Field(default=0.5, ge=-1.0, le=1.0, description="Ranges from calmness (-1) to anger (+1), with passive awareness (0).")
    expectancy: float = Field(default=0.5, ge=-1.0, le=1.0, description="Ranges from predictability (-1) to surprise (+1), with anticipation (0).")
    energy: float = Field(default=0.5, ge=-1.0, le=1.0, description="Ranges from low energy (-1) to high energy (+1), with restfulness (0).")
    pleasantness: float = Field(default=0.5, ge=-1.0, le=1.0, description="Ranges from unpleasant (-1) to pleasant (+1), with neutrality (0).")
    passion: float = Field(default=0.5, ge=0.0, le=1.0, description="From absence of passion (0) to full intensity (1).")
    curiosity: float = Field(default=0.5, ge=0.0, le=1.0, description="From absence of curiosity (0) to full intensity of curiosity (+1).")
    confidence: float = Field(default=0.5, ge=-1.0, le=1.0, description="Ranges from insecurity (-1) to confidence (+1), with neutrality (0).")
    empathy: float = Field(default=0, ge=0.0, le=1.0, description="From absence (0) to full intensity of empathy (+1).")
    jealousy: float = Field(default=0, ge=0.0, le=1.0, description="From absence (0) to full intensity of jealousy (+1).")
    guilt: float = Field(default=0, ge=0.0, le=1.0, description="From absence (0) to full intensity of guilt (+1).")
    gratitude: float = Field(default=0, ge=0.0, le=1.0, description="From absence (0) to full intensity of gratitude (+1).")
    loneliness: float = Field(default=0, ge=0.0, le=1.0, description="From absence (0) to full intensity of loneliness (+1).")

class AgentState(BaseModel):
    input: str = Field(default_factory=str)
    output: str = Field(default_factory=str)

    title: str = Field(default_factory=str)

    conversation_id: str = Field(default_factory=str)

    story_mode: bool = Field(default=False)
    task: List[str] = Field(default_factory=list)
    complexity: float = Field(default=0)
    available_tools: List[str] = Field(default_factory=list)
    completion_mode: str = Field(default_factory=str)

    status: int = Field(default_factory=str)
    next_agent: str = Field(default_factory=str)

    plan: str = Field(default_factory=str)
    knowledge_implicit_facts: str = Field(default_factory=str)
    knowledge_implicit_conversation: str = Field(default_factory=str)

    thought: str = Field(default_factory=str)

    emotional_state: EmotionalAxesModel = Field(default_factory=EmotionalAxesModel)