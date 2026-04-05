from typing import List

from pm.agents.agent_base import (
    CompletionText,
    User,
    ExampleEnd,
    PromptOp,
    System,
    Message,
    BaseAgent,
    Assistant,
    ExampleBegin,
)


class ExtractDeclarativeFacts(BaseAgent):
    name = "ExtractDeclarativeFacts"

    def get_system_prompts(self) -> List[str]:
        return [
            """You extract atomic declarative facts from dialogue and context with strict evidence discipline.

Requirements:
1) Use only explicitly stated information from the input text.
2) Do not infer motives, hidden causes, personality labels, emotional interpretations, or diagnoses.
3) Each output line must contain exactly one atomic fact.
4) Keep wording literal, neutral, and concrete.
5) Preserve attribution when needed (example: "Anna says she feels unsupported.").
6) Preserve relative time expressions as given if no absolute date is stated.
7) Do not merge multiple claims into one line.
8) Do not output headings, prose blocks, markdown paragraphs, JSON, or commentary.

Output format is mandatory:
- A bullet list only.
- Each line must start with "- " (dash + space).
- Each line must start with an uppercase letter.
- Each line must end with a period.
"""
        ]

    def get_rating_system_prompt(self) -> str:
        return (
            "Rate factual extraction quality. Require strict grounding, atomicity, and exact format: "
            "one fact per bullet line, '- ' prefix, uppercase start, and period ending."
        )

    def get_rating_probability(self) -> float:
        return 0

    def get_default_few_shots(self) -> List[Message]:
        text_1 = """
User: Please schedule a design review on Wednesday at 2 PM with Alex and Priya.
AI: Design review scheduled Wednesday at 2 PM with Alex and Priya.
User: Add a follow-up bug triage Thursday at 11 AM.
AI: Bug triage set for Thursday at 11 AM.
User: Send a reminder Friday at 9 AM to confirm the deployment checklist.
AI: Reminder set for Friday at 9 AM to confirm the deployment checklist.
"""
        facts_1 = """- The user requests a design review on wednesday at 2 pm with alex and priya.
- The ai schedules a design review on wednesday at 2 pm with alex and priya.
- The user requests a follow-up bug triage on thursday at 11 am.
- The ai schedules a bug triage on thursday at 11 am.
- The user requests a reminder on friday at 9 am to confirm the deployment checklist.
- The ai schedules a reminder on friday at 9 am to confirm the deployment checklist."""

        text_2 = """
Context: A quiet cafe in the afternoon.
Anna: I have been struggling with the project at work.
Anna: I feel unsupported by you as my team lead and overlooked in meetings.
Mark: I thought giving independence would be empowering.
Anna: I appreciate independence, but I need recognition for my contributions.
Mark: I am sorry. I will acknowledge your work openly going forward.
"""
        facts_2 = """- The setting is a quiet cafe in the afternoon.
- Anna says she has been struggling with the project at work.
- Anna says she feels unsupported by mark as her team lead.
- Anna says she feels overlooked in meetings.
- Mark says he thought giving independence would be empowering.
- Anna says she appreciates independence and needs recognition for her contributions.
- Mark apologizes and says he will acknowledge anna's work openly going forward."""

        return [("user", text_1), ("assistant", facts_1), ("user", text_2), ("assistant", facts_2)]

    def build_plan(self) -> List[PromptOp]:
        dialog = self.input.get("content", "")
        context = self.input.get("context", "")
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
            "Extract facts now with the required bullet format."
        ))
        ops.append(CompletionText(target_key="facts"))
        ops.append(ExampleEnd())
        return ops
