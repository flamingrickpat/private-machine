from typing import List, Type
from pydantic import BaseModel, Field

from pm.agents.agent_base import (
    CompletionText, User, CompletionJSON, ExampleEnd, PromptOp, System,
    Message, BaseAgent, Assistant, ExampleBegin
)

# ---------------------------
# Data model for the JSON output
# ---------------------------
class FactCategorization(BaseModel):
    reason: str = Field(description="Explain why you chose these categories, importance, and time_dependent.")
    category: List[str] = Field(description="Short list of the most relevant categories.")
    importance: float = Field(description="Overall importance from 0.0 to 1.0.")
    time_dependent: float = Field(description="0.0 = stable/always true; 1.0 = only true right now (ephemeral).")

# ---------------------------
# Category palette shown to the model
# ---------------------------
example_categories = """
[
  {
    "path": "people_personality",
    "description": "Core personality traits and enduring dispositions (e.g., empathy, curiosity, playfulness, conscientiousness)."
  },
  {
    "path": "people_needs",
    "description": "Intrinsic/ongoing needs (downtime, connection, predictability, support) that modulate functioning over time."
  },
  {
    "path": "people_goals",
    "description": "Objectives and aspirations, near- or long-term (career goals, project aims, missions)."
  },
  {
    "path": "people_preferences",
    "description": "Likes/dislikes and decision biases (UI style, colors, mediums, foods, routines)."
  },
  {
    "path": "people_interactions",
    "description": "Interpersonal acts, commitments, communications, and role dynamics between parties."
  },
  {
    "path": "world_places",
    "description": "Concrete locations and settings (addresses, cities, venues, rooms)."
  },
  {
    "path": "world_environment",
    "description": "Ambient/contextual conditions (weather, atmosphere, constraints) that shape a scene."
  },
  {
    "path": "world_world_building",
    "description": "Stable background structures and practices (org setups, tech stacks, cultural norms, policies)."
  },
  {
    "path": "world_events",
    "description": "Time-bound happenings (meetings, deployments, alerts, ceremonies) with when/where details."
  },
  {
    "path": "relationships_good",
    "description": "Positive states in relationships (cooperation, trust, repair, mutual support)."
  },
  {
    "path": "relationships_bad",
    "description": "Negative states in relationships (conflict, tension, mistrust, neglect)."
  }
]
"""

class CategorizeFacts(BaseAgent):
    name = "CategorizeFacts"

    # ---------------------------
    # System prompts with strong guidance on time_dependent & importance
    # ---------------------------
    def get_system_prompts(self) -> List[str]:
        return [
            (
                "You categorize a SINGLE atomic fact into high-level semantic buckets and rate its importance and time dependence.\n"
                "Rules:\n"
                "• Output must be valid JSON matching the provided schema.\n"
                "• Choose 1–4 categories that best explain the fact’s semantic role.\n"
                "• Importance (0–1): estimate long-term utility for reasoning, planning, and user modeling.\n"
                "  – High (≥0.8): enduring traits, major relationship changes, critical policies/agreements, structural tech choices.\n"
                "  – Medium (0.4–0.7): recurring practices, non-trivial but revisable plans, notable preferences.\n"
                "  – Low (≤0.3): one-off minor events, trivial observations.\n"
                "• time_dependent (0–1): stability over time.\n"
                "  – 0.0–0.2: very stable or quasi-permanent (personality trait; office address; long-standing policy).\n"
                "  – 0.3–0.6: medium stability (current role, ongoing plan, recurring schedule subject to change).\n"
                "  – 0.7–0.9: short-lived or tied to a near-term window (this week’s milestone; sprint schedule).\n"
                "  – 1.0: instantaneous/ephemeral state (today’s weather; a spike at 14:05; forgetting keys).\n"
                "Heuristics for time_dependent:\n"
                "• Facts with explicit timestamps/one-time events → push toward 0.9–1.0.\n"
                "• Recurring commitments without end date → 0.5–0.7.\n"
                "• Enduring dispositions/long-term policies → 0.1–0.3.\n"
                "• If a fact encodes both a stable pattern AND a dated instance, bias to the instance’s volatility.\n"
                "• Prefer conservative (higher) time_dependent when in doubt about persistence.\n"
                "Always explain the rationale in `reason` referencing stability, scope, and downstream utility."
            ),
            f"Output JSON schema (strictly match):\n{FactCategorization.schema_json()}",
            f"Category palette (choose from these when applicable):\n{example_categories}"
        ]

    def get_rating_system_prompt(self) -> str:
        return (
            "Rate whether the JSON is valid and whether category, importance, and time_dependent follow the rubric. "
            "Reject speculation and ensure reasons reference persistence and utility."
        )

    def get_rating_probability(self) -> float:
        return 0

    # ---------------------------
    # Few-shot examples (long & complex; emphasize time_dependent calibration)
    # ---------------------------
    def get_default_few_shots(self) -> List[Message]:
        return [
            # 1) Enduring but revisable relationship change + event/place
            ("user", "Alice and Bob got married on 2024-06-12 in Vienna."),
            ("assistant", """{
  "reason": "Marriage is a pivotal relationship change with enduring consequences; includes a dated event and a place. High long-term utility (relationship state), but the act itself is a dated event (non-ephemeral outcome).",
  "category": ["relationships_good", "people_interactions", "world_events", "world_places"],
  "importance": 0.95,
  "time_dependent": 0.20
}"""),

            # 2) Ephemeral incident
            ("user", "John stepped outside and realized he forgot his car keys."),
            ("assistant", """{
  "reason": "A one-off, momentary incident with little long-term impact. It may inform immediate actions but quickly loses relevance.",
  "category": ["world_events", "people_interactions"],
  "importance": 0.15,
  "time_dependent": 0.98
}"""),

            # 3) Stable tech choice / organizational practice
            ("user", "The company uses AWS as its primary cloud provider."),
            ("assistant", """{
  "reason": "Primary cloud choice is structural, affects many downstream decisions, and tends to be stable over months/years though it could change.",
  "category": ["world_world_building"],
  "importance": 0.85,
  "time_dependent": 0.25
}"""),

            # 4) Recurring, open-ended commitment (medium stability)
            ("user", "The team holds biweekly check-ins for the next six months."),
            ("assistant", """{
  "reason": "Recurring practice with a defined horizon; not one-off, but not permanent. Useful for scheduling and coordination.",
  "category": ["people_interactions", "world_events"],
  "importance": 0.60,
  "time_dependent": 0.65
}"""),

            # 5) Trait-level insight (very stable)
            ("user", "Mark values transparency and prefers explicit feedback loops."),
            ("assistant", """{
  "reason": "An enduring disposition (transparency) plus a preference for process (explicit feedback). High modeling utility and relatively stable.",
  "category": ["people_personality", "people_preferences"],
  "importance": 0.80,
  "time_dependent": 0.20
}"""),

            # 6) Preference (moderately stable)
            ("user", "Priya prefers dark mode and keyboard-driven UIs."),
            ("assistant", """{
  "reason": "Usability preferences are often stable across tasks but may evolve; helpful for tailoring tools and docs.",
  "category": ["people_preferences"],
  "importance": 0.55,
  "time_dependent": 0.35
}"""),

            # 7) Ambient condition (ephemeral)
            ("user", "It is raining today in Linz."),
            ("assistant", """{
  "reason": "Weather is highly time-bound; useful for immediate plans only.",
  "category": ["world_environment", "world_places"],
  "importance": 0.20,
  "time_dependent": 1.00
}"""),

            # 8) Ongoing need pattern (mid stability)
            ("user", "Anna says she needs quiet time after consecutive client calls to recharge."),
            ("assistant", """{
  "reason": "A stated recurring need impacts workload planning and wellbeing. Likely persists but may vary.",
  "category": ["people_needs", "people_interactions"],
  "importance": 0.70,
  "time_dependent": 0.45
}"""),

            # 9) Sensitive relationship state (can change, so medium-high time dependence)
            ("user", "Anna feels overlooked by her team lead and the relationship is strained."),
            ("assistant", """{
  "reason": "A negative relationship state that can shift with interventions; important for social dynamics but not permanent.",
  "category": ["relationships_bad", "people_interactions"],
  "importance": 0.75,
  "time_dependent": 0.60
}"""),

            # 10) Concrete address (quite stable)
            ("user", "The office is located at Kärntner Straße 12, Vienna."),
            ("assistant", """{
  "reason": "A fixed place is typically long-lived and enables spatial grounding and logistics.",
  "category": ["world_places"],
  "importance": 0.65,
  "time_dependent": 0.20
}"""),

            # 11) Dated deployment plan (short-term, high volatility)
            ("user", "The team will deploy version 2.3 on Friday at 18:00 CET."),
            ("assistant", """{
  "reason": "A scheduled, time-specific event valuable for near-term coordination; once completed it becomes historical.",
  "category": ["world_events", "people_interactions"],
  "importance": 0.70,
  "time_dependent": 0.90
}"""),

            # 12) Standing reminder (recurring, indefinite = mid/high time dependence)
            ("user", "Chatty will remind the user every Monday at 9 AM about milestones."),
            ("assistant", """{
  "reason": "A recurring commitment without an explicit end date; important for planning but subject to change.",
  "category": ["people_interactions", "world_events"],
  "importance": 0.50,
  "time_dependent": 0.60
}"""),

            # 13) Policy-level practice (stable but revisable)
            ("user", "Security reviews require two independent approvals before release."),
            ("assistant", """{
  "reason": "Organizational policy that influences workflows broadly; usually stable across releases.",
  "category": ["world_world_building", "people_interactions"],
  "importance": 0.85,
  "time_dependent": 0.25
}"""),

            # 14) Performance spike (instantaneous)
            ("user", "CPU utilization spiked to 95% at 14:05 UTC."),
            ("assistant", """{
  "reason": "A momentary observation that may inform incident timelines but quickly loses predictive value.",
  "category": ["world_events"],
  "importance": 0.40,
  "time_dependent": 1.00
}"""),

            # 15) Goals (longer-lived intention)
            ("user", "Nora's goal is to publish the research report by the end of the quarter."),
            ("assistant", """{
  "reason": "An explicit goal that guides planning for weeks; not permanent but meaningful for decision-making.",
  "category": ["people_goals"],
  "importance": 0.75,
  "time_dependent": 0.50
}"""),

            # 16) Repair/commitment (relationship improvement; can evolve)
            ("user", "Mark promised to acknowledge Anna’s contributions publicly in future meetings."),
            ("assistant", """{
  "reason": "A commitment that can shift relationship dynamics; impactful but contingent on future behavior.",
  "category": ["relationships_good", "people_interactions"],
  "importance": 0.80,
  "time_dependent": 0.55
}"""),

            # 17) Cultural norm (stable background)
            ("user", "Design critiques at this company are open to all engineers and designers."),
            ("assistant", """{
  "reason": "A cultural practice/policy that shapes collaboration; typically stable.",
  "category": ["world_world_building", "people_interactions"],
  "importance": 0.70,
  "time_dependent": 0.30
}"""),

            # 18) Personal constraint (ongoing need)
            ("user", "Omar avoids late-evening meetings due to childcare responsibilities."),
            ("assistant", """{
  "reason": "A persistent scheduling constraint rooted in ongoing responsibilities; helpful for planning.",
  "category": ["people_needs", "people_preferences", "people_interactions"],
  "importance": 0.70,
  "time_dependent": 0.40
}"""),

            # 19) Stable preference + environment trigger
            ("user", "Riley prefers quiet workspaces and uses noise-cancelling headphones when the office gets loud."),
            ("assistant", """{
  "reason": "Preference likely stable; the mitigation behavior is situational. Useful for environment-person fit.",
  "category": ["people_preferences", "world_environment", "people_interactions"],
  "importance": 0.60,
  "time_dependent": 0.35
}"""),

            # 20) Temporary configuration window (short-lived)
            ("user", "Feature flags A and B are enabled only during this week’s canary rollout."),
            ("assistant", """{
  "reason": "A strictly time-bound configuration valuable for near-term diagnostics; becomes obsolete after the window.",
  "category": ["world_events", "world_world_building"],
  "importance": 0.55,
  "time_dependent": 0.90
}""")
        ]

    # ---------------------------
    # Build the prompt plan
    # ---------------------------
    def build_plan(self) -> List[PromptOp]:
        fact = self.input.get("fact", "")
        ops: List[PromptOp] = []

        for s in self.get_system_prompts():
            ops.append(System(s))
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
