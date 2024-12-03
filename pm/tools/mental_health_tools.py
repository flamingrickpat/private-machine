from datetime import datetime
from typing import List

from nltk.corpus import words
from pydantic import BaseModel, Field
from streamlit import progress
from brave import Brave
from tavily import TavilyClient

from pm.controller import controller
from pm.database.db_helper import insert_object, insert_system_message
from pm.database.db_model import Reminder, Goal, Fact
from pm.utils.string_utils import get_last_n_messages_or_words_from_string, clip_to_sentence_by_words
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
        state["output"] = f"Reminder has been successfully set for {time}!"


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
        state["output"] = f"Goal '{self.goal}' has been created!"

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
            state["output"] = f"Goal with identifier '{self.identifier}' not found!"
        else:
            new_status = goal[0].status + f"\nStatus update from {datetime.now()}:\n" + self.status
            goals.update(where=where, values={"status": new_status, "progress": progress})
            state["output"] = f"Goal with ID '{self.identifier}' has been progressed!"


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
            state["output"] = f"Goal with identifier '{self.identifier}' not found!"
        else:
            goals.update(where=where, values={"progress": 1})
            state["output"] = f"Goal with ID '{self.identifier}' has been finished!"

class ListGoals(BaseModel):
    """
    See all ongoing goals, sorted by a string query.
    """
    sort_query: str = Field(description="vector search query to order goals")

    def execute(self, state):
        goals = controller.db.open_table(Goal.table)
        goal: List[Goal] = goals.search(controller.embedder.get_embedding_scalar_float_list(self.sort_query)).limit(64).to_pydantic()

        lst = "\n".join([f"Identifier: {x.identifier} Goal: {x.goal} Progress: {x.progress}" for x in goal])
        state["output"] = f"Goal list:\n{lst}"

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
            state["output"] = f"Goal with identifier '{self.identifier}' not found!"
        else:
            state["output"] = goal[0].model_dump_json()

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

        state["output"] = f"Fact successfully saved!"

class SearchWeb(BaseModel):
    """
    Search the brave web search API for external information about ongoing events not in your training data.
    """
    query: str = Field(description="web query from brave search API")

    def execute(self, state):
        def format_tavily_result(tavily_result):
            query = tavily_result.get("query", "No query provided")
            results = tavily_result.get("results", [])

            compact_result = f"Query: {query}\nTop Results:\n"
            for i, result in enumerate(results[:2], start=1):
                title = result.get("title", "No title")
                url = result.get("url", "No URL")
                content = result.get("content", "No content").replace("\n", "")
                compact_result += f"{i}. {title}\n   Summary: {content}\n\n"

            return compact_result.strip()

        if not controller.config.tavily_api_key or controller.config.tavily_api_key == "":
            state["output"] = f"Tavily Web Search not set up, please inform your user!"
        else:
            tavily_client = TavilyClient(api_key=controller.config.tavily_api_key)
            response = format_tavily_result(tavily_client.search(self.query))
            state["output"] = f"Search Results: {response}"

class InsertDiaryEntry(BaseModel):
    """
    Document something from the user's life.
    """
    entry: str = Field(description="entry about events")

    def execute(self, state):
        print(self.entry)