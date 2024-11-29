from datetime import datetime
from typing import List

from pydantic import BaseModel, Field
from streamlit import progress
from brave import Brave

from pm.controller import controller
from pm.database.db_helper import insert_object, insert_system_message
from pm.database.db_model import Reminder, Goal, Fact
from pm.utils.token_utils import quick_estimate_tokens


class SetReminder(BaseModel):
    """
    Set a reminder to yourself at some point in the future.
    """
    time: str = Field(description="can be in iso format or also like 'tomorrow morning', the agent will parse it to correct format")
    reminder_text: str = Field(description="information so you remember what should be done")

    def execute(self, state):
        time = datetime.now() #agent_convert_text_to_datetime(self.time)
        rem = Reminder(
            reminder_text=self.reminder_text,
            reminder_time=time
        )
        insert_object(rem)
        insert_system_message(state["conversation_id"], f"Reminder has been successfully set for {time}!")


class AddGoal(BaseModel):
    """
    Create a goal that you can track. It can be anything, from goal from user to personal goal!
    """
    identifier: str = Field(description="id of goal")
    goal: str = Field(description="recurrent theme of operation that will be converted into defined goal")
    description: str = Field(description="detail description of goal")

    def execute(self, state):
        goal = Goal(
            identifier=self.identifier,
            goal=self.goal,
            description=self.description,
            status="",
            progress=0
        )
        insert_object(goal)
        insert_system_message(state["conversation_id"], f"Goal '{self.goal}' has been created!")

class ProgressGoal(BaseModel):
    """
    Progress a goal by adding to the status and setting progression value.
    """
    identifier: str = Field(description="id of goal")
    status: str = Field(description="current status and information")
    progress: float = Field(description="progress of goal from 0 to 0.9 because 1 means finished", ge=0, le=0.9)

    def execute(self, state):
        goals = controller.db.open_table(Goal.table)
        where = f"identifier='{self.identifier}'"
        goal = goals.search().where(where, prefilter=True).limit(1).to_pydantic()
        if len(goal) == 0:
            insert_system_message(state["conversation_id"], f"Goal with identifier '{self.identifier}' not found!")
        else:
            new_status = goal[0].status + f"\nStatus update from {datetime.now()}:\n" + self.status
            goals.update(where=where, values={"status": new_status, "progress": progress})
            insert_system_message(state["conversation_id"], f"Goal with ID '{self.identifier}' has been progressed!")


class FinishGoal(BaseModel):
    """
    Finish a goal to not see it in the list of goals anymore.
    """
    identifier: str = Field(description="id of goal")
    finish_reason: str = Field(description="why the goal is to be finished")

    def execute(self, state):
        goals = controller.db.open_table(Goal.table)
        where = f"identifier='{self.identifier}'"
        goal = goals.search().where(where, prefilter=True).limit(1).to_pydantic()
        if len(goal) == 0:
            insert_system_message(state["conversation_id"], f"Goal with identifier '{self.identifier}' not found!")
        else:
            goals.update(where=where, values={"progress": 1})
            insert_system_message(state["conversation_id"], f"Goal with ID '{self.identifier}' has been finished!")

class ListGoals(BaseModel):
    """
    See all ongoing goals, sorted by a string query.
    """
    sort_query: str = Field(description="vector search query to order goals")

    def execute(self, state):
        goals = controller.db.open_table(Goal.table)
        goal: List[Goal] = goals.search(controller.embedder.get_embedding_scalar_float_list(self.sort_query)).limit(64).to_pydantic()

        lst = "\n".join([f"Identifier: {x.identifier} Goal: {x.goal} Progress: {x.progress}" for x in goal])
        insert_system_message(state["conversation_id"], f"Goal list:\n{lst}")

class SeeGoalDetails(BaseModel):
    """
    List all information about an ongoing goal.
    """
    identifier: str

    def execute(self, state):
        goals = controller.db.open_table(Goal.table)
        where = f"identifier='{self.identifier}'"
        goal = goals.search().where(where, prefilter=True).limit(1).to_pydantic()
        if len(goal) == 0:
            insert_system_message(state["conversation_id"], f"Goal with identifier '{self.identifier}' not found!")
        else:
            insert_system_message(state["conversation_id"], goal[0].model_dump_json())

class ExtendPersonality(BaseModel):
    """
    If you learn something about yourself and want to remember it in the future, use this to write a fact about yourself into your memory.
    """
    fact: str = Field(description="new fact about the AI companion character")

    def execute(self, state):
        fact = self.fact
        f = Fact(
            conversation_id=state["conversation_id"],
            text=fact,
            importance=1,
            embedding=controller.embedder.get_embedding_scalar_float_list(fact),
            category="self_inserted",
            tokens=quick_estimate_tokens(fact),
        )
        insert_object(f)

        insert_system_message(state["conversation_id"], f"Fact successfully saved!")

class SearchWeb(BaseModel):
    """
    Search the brave web search API for external information about ongoing events not in your training data.
    """
    query: str = Field(description="web query from brave search API")

    def execute(self, state):
        if not controller.config.brave_api_key or controller.config.brave_api_key == "":
            insert_system_message(state["conversation_id"], f"Brave Web Search not set up, please inform your user!")
        else:
            brave = Brave(api_key=controller.config.brave_api_key)
            query = "current us president"
            num_results = 4
            search_results = brave.search(q=query, count=num_results)

            def extract_text_from_results(json_response):
                if 'web' in json_response and 'results' in json_response['web']:
                    results = json_response['web']['results']
                    extracted_text = [
                        f"Title: {result.get('title', '')}\nDescription: {result.get('description', '')}"
                        for result in results
                    ]
                    return "\n\n".join(extracted_text)
                return "No relevant results found."

            insert_system_message(state["conversation_id"], f"Search Results: {extract_text_from_results(search_results.model_dump())}")

class InsertDiaryEntry(BaseModel):
    """
    Document something from the user's life.
    """
    entry: str = Field(description="")