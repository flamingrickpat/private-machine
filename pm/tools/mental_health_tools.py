from pydantic import BaseModel, Field

class SetReminder(BaseModel):
    time: str = Field(description="can be in iso format or also like 'tomorrow morning', the agent will parse it to correct format")
    reminder_text: str = Field(description="information so you remember what should be done")

class AddGoal(BaseModel):
    identifier: str = Field(description="id of goal")
    goal: str = Field(description="recurrent theme of operation that will be converted into defined goal")
    description: str = Field(description="detail description of goal")

class ProgressGoal(BaseModel):
    identifier: str = Field(description="id of goal")
    status: str = Field(description="current status and information")
    progress: float = Field(description="progress of goal from 0 to 1", ge=0, le=1)

class FinishGoal(BaseModel):
    identifier: str = Field(description="id of goal")

class ListGoals(BaseModel):
    sort_query: str = Field(description="vector search query to order goals")

class ExtendPersonality(BaseModel):
    fact: str = Field(description="new fact about the AI companion character")

class SearchWeb(BaseModel):
    query: str = Field(description="web query from brave search API")
