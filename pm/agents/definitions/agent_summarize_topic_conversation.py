from typing import List

from pm.agents.agent_base import CompletionText, User, ExampleEnd, PromptOp, System, Message, BaseAgent, Assistant, ExampleBegin


class SummarizeTopicConversation(BaseAgent):
    name = "SummarizeTopicConversation"

    def get_system_prompts(self) -> List[str]:
        return [
            """You summarize dialogue and event records into neutral memory text for downstream reasoning.

Primary goal:
- Produce a concise, evidence-grounded summary with no interpretation drift.

Hard rules:
1) Use only facts present in the provided text.
2) Do not infer motives, diagnoses, hidden emotions, or symbolic meaning.
3) Do not evaluate who is right or wrong.
4) Preserve timeline-relevant details when available (people, actions, decisions, commitments, constraints).
5) If uncertainty exists in source text, keep wording conservative.
6) Output exactly one plain paragraph.
7) Do not output bullets, markdown, headings, numbering, or JSON.
8) Use a dry, neutral, analytic tone with direct language.
"""
        ]

    def get_rating_system_prompt(self) -> str:
        return (
            "Rate summary quality for neutrality and grounding. "
            "Reject speculative interpretation and reject any non-paragraph formatting."
        )

    def get_rating_probability(self) -> float:
        return 0

    def get_default_few_shots(self) -> List[Message]:
        text_1 = """
User: Thanks for all the help. What are the next steps?
AI: The stack is GCP and the UI uses React, Material UI, and Recharts.
AI: Week 1 is environment setup and IAM. Week 2 is Pub/Sub and Cloud Functions.
AI: Week 3 is BigQuery schema and ETL design. Week 4 is frontend scaffolding.
User: Please store these milestones and remind me weekly.
AI: I will store milestones and send weekly reminders.
"""
        summary_1 = (
            "The dialogue defines a project plan using a GCP stack with a React, Material UI, and Recharts UI, then outlines "
            "milestones for four weeks covering environment and IAM setup, Pub/Sub and Cloud Functions, BigQuery schema and ETL design, "
            "and frontend scaffolding, and it ends with a confirmed request to store milestones and send weekly reminders."
        )

        text_2 = """
Context: A quiet cafe in the afternoon.
Anna: I have been struggling with the project at work.
Anna: I feel unsupported by you as my team lead and overlooked in meetings.
Mark: I thought giving independence would be empowering.
Anna: I appreciate independence, but I need recognition for my contributions.
Mark: I am sorry and I will acknowledge your work openly.
"""
        summary_2 = (
            "In a cafe conversation, Anna states she is struggling with the project and feels unsupported and overlooked, Mark states he intended "
            "independence to be empowering, Anna states she still needs recognition, and Mark apologizes and commits to acknowledging her work openly."
        )

        return [("user", text_1), ("assistant", summary_1), ("user", text_2), ("assistant", summary_2)]

    def build_plan(self) -> List[PromptOp]:
        dialog = self.input["content"]
        context = self.input["context"]
        ops: List[PromptOp] = []

        for system_prompt in self.get_system_prompts():
            ops.append(System(system_prompt))
        for role, text in self.get_default_few_shots():
            if role == "user":
                ops.append(User(text))
            elif role == "assistant":
                ops.append(Assistant(text))

        ops.append(ExampleBegin())
        ops.append(User(
            f"Context: {context}\n"
            f"{dialog}\n\n"
            "Write exactly one neutral paragraph summary."
        ))
        ops.append(CompletionText(target_key="summary"))
        ops.append(ExampleEnd())
        return ops
