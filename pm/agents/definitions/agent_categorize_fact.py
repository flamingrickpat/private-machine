from typing import List

from pydantic import BaseModel, Field

from pm.agents.agent_base import PromptOp, System, User, Assistant, ExampleBegin, ExampleEnd, CompletionJSON, Message, BaseAgent


class FactCategorization(BaseModel):
    reason: str = Field(description="Neutral rationale tied to explicit wording, stability, and utility.")
    category: List[str] = Field(description="Short list of the most relevant categories.")
    importance: float = Field(description="Overall importance from 0.0 to 1.0.")
    time_dependent: float = Field(description="0.0 = stable/always true; 1.0 = only true right now (ephemeral).")


example_categories = """
[
  {"path": "people_personality", "description": "Explicitly stated stable personality dispositions."},
  {"path": "people_needs", "description": "Explicitly stated ongoing needs and constraints."},
  {"path": "people_goals", "description": "Explicitly stated objectives and target outcomes."},
  {"path": "people_preferences", "description": "Explicitly stated likes, dislikes, and choice preferences."},
  {"path": "people_interactions", "description": "Communications, requests, commitments, role interactions."},
  {"path": "world_places", "description": "Concrete locations and settings."},
  {"path": "world_environment", "description": "Ambient conditions and operational context."},
  {"path": "world_world_building", "description": "Stable organizational or technical structure."},
  {"path": "world_events", "description": "Time-bound events and actions."},
  {"path": "relationships_good", "description": "Explicitly stated positive relationship states or repair actions."},
  {"path": "relationships_bad", "description": "Explicitly stated tension, conflict, or distrust states."}
]
"""


class CategorizeFacts(BaseAgent):
    name = "CategorizeFacts"

    def get_system_prompts(self) -> List[str]:
        return [
            """You categorize one atomic fact into semantic buckets for memory indexing.

Hard rules:
1) Use only the provided fact text.
2) Do not infer hidden motives or unstated psychological constructs.
3) Choose 1-4 categories from the provided palette.
4) `reason` must be neutral and evidence-referential, not interpretive.
5) `importance` reflects downstream retrieval/planning utility:
   - High: durable constraints, major commitments, structural decisions.
   - Medium: recurring practices, notable but revisable plans.
   - Low: transient minor details.
6) `time_dependent` reflects expected persistence:
   - 0.0-0.2 stable, long-lived.
   - 0.3-0.6 medium persistence.
   - 0.7-1.0 short-lived or time-window specific.
7) Output must be valid JSON for the target schema.
""",
            f"Output JSON schema:\n{FactCategorization.schema_json()}",
            f"Category palette:\n{example_categories}",
        ]

    def get_rating_system_prompt(self) -> str:
        return (
            "Rate validity and grounding. Reject interpretive drift. "
            "Reason must cite explicit wording and persistence implications."
        )

    def get_rating_probability(self) -> float:
        return 0

    def get_default_few_shots(self) -> List[Message]:
        return [
            (
                "user",
                "The team will deploy version 2.3 on Friday at 18:00 CET.",
            ),
            (
                "assistant",
                """{
  "reason": "The fact is a time-specific deployment event used for near-term coordination.",
  "category": ["world_events", "people_interactions"],
  "importance": 0.7,
  "time_dependent": 0.9
}""",
            ),
            (
                "user",
                "The company uses AWS as its primary cloud provider.",
            ),
            (
                "assistant",
                """{
  "reason": "The fact states a structural technical choice that shapes many downstream decisions over time.",
  "category": ["world_world_building"],
  "importance": 0.85,
  "time_dependent": 0.25
}""",
            ),
            (
                "user",
                "Anna says she feels overlooked in meetings.",
            ),
            (
                "assistant",
                """{
  "reason": "The fact is an attributed interpersonal state that matters for interaction planning and can change.",
  "category": ["relationships_bad", "people_interactions"],
  "importance": 0.75,
  "time_dependent": 0.6
}""",
            ),
        ]

    def build_plan(self) -> List[PromptOp]:
        fact = self.input.get("fact", "")
        ops: List[PromptOp] = []

        for system_prompt in self.get_system_prompts():
            ops.append(System(system_prompt))
        for role, text in self.get_default_few_shots():
            if role == "user":
                ops.append(User(text))
            elif role == "assistant":
                ops.append(Assistant(text))

        ops.append(ExampleBegin())
        ops.append(User(fact))
        ops.append(CompletionJSON(schema=FactCategorization, target_key="category"))
        ops.append(ExampleEnd())
        return ops
